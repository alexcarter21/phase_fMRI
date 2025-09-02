#%%
'''
What's going on here? 
We are using nipype to create a pipeline that will take a 4D fMRI image and a T1 image and register them to the MNI152 2mm template.
The pipeline will:  
1. Perform motion correction on the 4D fMRI image
2. Compute the mean EPI image
3. Perform brain extraction on the T1 image
4. Perform rigid registration of the mean EPI to the T1 image
5. Perform nonlinear registration of the mean EPI to the T1 image
6. Perform rigid and affine registration of the T1 image to the MNI152 2mm template
7. Merge the transforms from steps 5 and 6
8. Apply the merged transforms to the motion-corrected 4D fMRI image
9. Save the output image in the MNI152 2mm template space
  
Using nipype adds a bit of overhead a the start but I think it's worth it:
It makes the code more modular and easier to understand
It makes it easier to visualize the workflow
It makes it easier to debug
It makes it easier to run the workflow on a cluster
It makes it easier to parallelize the workflow
It keeps track of what has been run and what hasn't - so when you re-run the pipeline it only runs the bits that need to be re-run

Some gotchas that I found along the way that others might find useful:
Merging the transforms was a pain. In particular, merging separate affine, nonlinear and T1->MNI transforms was tricky. I had to use the Merge node and then connect the output of the Merge node to the ApplyTransforms node. I also had to set the 'collapse_output_transforms' and 'write_composite_transform' options to True in the ants.Registration() nodes.
That write_composite thing was key. Note that for these data the final xform is both critical and hard because the data are so high-resolution (both in time and space) so running
out of memory is always a threat.

The datasink stuff was recommended for debugging and I left it in as an option but in the end I found it didn't really help much

Getting the alignments done well is hard. The affine from T2* to native T1 provides a start point that the nonlinear ANTS routine then refines. 
The parameters in the nonlinear stage are carefully chosen to work with these highres (and high temporal resolution) data that only cover a small part of the brain.
Note that I restrict the amount of deformation in the nonlinear stage to 1x1x1 voxels. This is a bit of a hack but it seems to work well.

I've run this on D2 (64GB, multi-core) and it takes about 10 mins. I'm restricting ANTS to 12 cores so that it does not take over the whole machine.

The next steps are to run the alignment on the remaining runs from a single session (so first mcflirt them to the average T2* image), then apply the composite
transform to the remaining runs. This will give us a set of aligned runs for each session. Then we can run the L1 fits on each session.

I think L1 etc can also happen in the same pipeline! 

ARW 032625
'''

#%%
import os
import nipype.interfaces.ants as ants
import nipype.interfaces.fsl as fsl
from nipype.pipeline import engine as pe
from nipype.pipeline import MapNode
from nipype.interfaces.io import DataSink
from nipype import config
from nipype.interfaces.utility import Function, Merge, IdentityInterface
import networkx as nx
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn import plotting
from nilearn.reporting import make_glm_report
import nibabel as nib
import numpy as np
import pandas as pd

os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '2'  # or the number of threads you want

#%%
# Don't need this monkey patch for the 'correct' nipype environment
#if not hasattr(nx, 'to_scipy_sparse_matrix'):  # Fix for newer versions of NetworkX 'monkey patch'!
#    nx.to_scipy_sparse_matrix = nx.to_scipy_sparse_array

DO_DATASINK_DEBUG=True

#config.enable_debug_mode()

# Define base directory
all_pptIDs = ['R5619','R6068','R6159','R6376','R6380','R6425','R6502','R6591','R6611','R6619','R6669','R6688','R6702','R6770','R6694','R6804']

for pptID in all_pptIDs:
    base_dir = '/scratch/groups/Projects/P1490'
    data_dir = 'Data' # directory of all participants' data
    nifti_dir='nifti' # A subdirectory of the data_dir in each ppt folder
    t1_dir='freesurfer' # directory where recon-all T1 is
    t1_folder = f'{pptID}_fs7/mri'

    # Get fslDir from the FSLDIR environment variable
    fslDir = os.environ['FSLDIR']
    fslDataDir=os.path.join(fslDir,'data/standard/')
    mni_template = os.path.join(fslDataDir,'MNI152_T1_1mm.nii.gz')

    t1_image = os.path.join(base_dir, t1_dir, t1_folder, 'T1.nii.gz')

    # These are the fMRI files we want to process
    if pptID == 'R5619':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '6_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '12_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz']
    elif pptID == 'R6068':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '12_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz']
    elif pptID == 'R6159':
        all_fmri_files= ['6_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz',
                        '12_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz',
                        '14_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN8_XPQ400.nii.gz']
    elif pptID == 'R6376':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '6_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '12_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz'] 
    elif pptID == 'R6380':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '14_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz']
    elif pptID == 'R6425':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '6_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '12_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz']
    elif pptID == 'R6502':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '14_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz']
    elif pptID == 'R6591':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '6_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '12_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz']
    elif pptID == 'R6611':
        all_fmri_files= ['6_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz',
                        '14_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN8_XPQ400.nii.gz']
    elif pptID == 'R6619':
        all_fmri_files= ['9_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '11_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz',
                        '13_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz',
                        '15_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz', 
                        '17_MB_1_13_ISO_COLPHASE_RUN8_XPQ400.nii.gz']
    elif pptID == 'R6669':
        all_fmri_files= ['4_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '12_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz',
                        '14_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz']
    elif pptID == 'R6688':
        all_fmri_files= ['2_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '4_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '14_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz']
    elif pptID == 'R6694':
        all_fmri_files= ['4_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '6_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '12_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz']
    elif pptID == 'R6702':
        all_fmri_files= ['4_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '6_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '12_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN7_XPQ400.nii.gz']
    elif pptID == 'R6770':
        all_fmri_files= ['6_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '10_MB_1_13_ISO_COLPHASE_RUN3_XPQ400.nii.gz',
                        '14_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz']
    elif pptID == 'R6804':
        all_fmri_files= ['6_MB_1_13_ISO_COLPHASE_RUN1_XPQ400.nii.gz',
                        '8_MB_1_13_ISO_COLPHASE_RUN2_XPQ400.nii.gz',
                        '12_MB_1_13_ISO_COLPHASE_RUN4_XPQ400.nii.gz',
                        '14_MB_1_13_ISO_COLPHASE_RUN5_XPQ400.nii.gz', 
                        '16_MB_1_13_ISO_COLPHASE_RUN6_XPQ400.nii.gz']

    #%%
    all_fmri_files = [os.path.join(base_dir, data_dir, pptID, nifti_dir, f) for f in all_fmri_files]  # This is your Python list

    # We pick off the first element in the list as the 'reference' 
    # All other files will be aligned to this one.
    reference_fmri_file = all_fmri_files[0] 

    # Create workflow
    wf = pe.Workflow(name='complete_ants_pipeline_fresh')
    wf.base_dir = os.path.join(base_dir,data_dir, pptID, 'nipype_workdir_1mm')

    # The way nipype works is that we define 'nodes' as below - then link up the inputs and outputs of those nodes

    # ---------------------------- NODE DEFINITION SECTION ------------------------
    # --- Input Node ---
    inputnode = pe.Node(IdentityInterface(fields=['t1_image', 'fmri_4d', 'mni_template']),
                        name='inputnode')
    inputnode.inputs.t1_image = t1_image
    inputnode.inputs.fmri_4d = reference_fmri_file
    inputnode.inputs.mni_template = mni_template

    # Here we set up a list of inputs a map node - the goal is to first compute the transform for one image,
    # Then apply that transform to all the images in the list. 
    listnode = pe.Node(IdentityInterface(fields=['fmri_files']), name='listnode')
    listnode.inputs.fmri_files = all_fmri_files  # This is your Python list

    # --- MCFLIRT for motion correction ---
    mcflirt = pe.Node(fsl.MCFLIRT(), name='mcflirt')
    mcflirt.inputs.cost = 'mutualinfo'
    mcflirt.inputs.output_type = 'NIFTI_GZ'
    mcflirt.inputs.save_mats = True
    mcflirt.config = {'execution': {'remove_unnecessary_outputs': False}}

    # --- Mean EPI calculation ---
    mean_epi = pe.Node(fsl.ImageMaths(), name='mean_epi')
    mean_epi.inputs.op_string = '-Tmean'
    mean_epi.inputs.output_type = 'NIFTI_GZ'

    # --- Brain Extraction (FSL BET) ---
    bet_t1 = pe.Node(fsl.BET(), name='bet_t1')
    bet_t1.inputs.robust = True
    bet_t1.inputs.mask = True
    bet_t1.inputs.output_type = 'NIFTI_GZ'

    # --- Rigid registration: EPI->T1 ---
    rigid_reg = pe.Node(ants.Registration(), name='rigid_reg')
    rigid_reg.inputs.transforms = ['Rigid']
    rigid_reg.inputs.transform_parameters = [(0.2,)]
    rigid_reg.inputs.metric = ['MI']
    rigid_reg.inputs.metric_weight = [1]
    rigid_reg.inputs.radius_or_number_of_bins = [32]
    rigid_reg.inputs.sampling_strategy = ['Regular']
    rigid_reg.inputs.sampling_percentage = [0.25]
    rigid_reg.inputs.number_of_iterations = [[1000, 500, 250, 150]]
    rigid_reg.inputs.convergence_threshold = [1e-6]
    rigid_reg.inputs.convergence_window_size = [10]
    rigid_reg.inputs.shrink_factors = [[8,4,2,1]]
    rigid_reg.inputs.smoothing_sigmas = [[3,2,1,0]]
    rigid_reg.inputs.sigma_units = ['vox']
    rigid_reg.inputs.use_histogram_matching = [True]
    rigid_reg.inputs.output_warped_image = True
    rigid_reg.inputs.output_transform_prefix = "rigid_"
    rigid_reg.inputs.verbose = True

    # --- Nonlinear registration: EPI->T1 ---
    nonlinear_reg = pe.Node(ants.Registration(), name='nonlinear_reg')
    nonlinear_reg.inputs.transforms = ['SyN']
    nonlinear_reg.inputs.transform_parameters = [(0.1, 3.0, 0.0)]
    nonlinear_reg.inputs.metric = ['CC']
    nonlinear_reg.inputs.metric_weight = [1.0]
    nonlinear_reg.inputs.radius_or_number_of_bins = [4]
    nonlinear_reg.inputs.number_of_iterations = [[200,100, 50, 10]]
    nonlinear_reg.inputs.convergence_threshold = [1e-6]
    nonlinear_reg.inputs.convergence_window_size = [10]
    nonlinear_reg.inputs.smoothing_sigmas = [[4,2,1,0]]
    nonlinear_reg.inputs.sigma_units = ['vox']
    nonlinear_reg.inputs.shrink_factors = [[8,4,2,1]]
    nonlinear_reg.inputs.use_histogram_matching = [True]
    nonlinear_reg.inputs.output_warped_image = True
    nonlinear_reg.inputs.output_transform_prefix = "nonlinear_"
    nonlinear_reg.inputs.winsorize_lower_quantile = 0.005
    nonlinear_reg.inputs.winsorize_upper_quantile = 0.995
    nonlinear_reg.inputs.verbose = True
    nonlinear_reg.inputs.args = '--restrict-deformation 1x1x0.2'
    nonlinear_reg.inputs.collapse_output_transforms = True
    nonlinear_reg.inputs.write_composite_transform = True

    # --- T1->MNI Registration ---
    t1_to_mni_reg = pe.Node(ants.Registration(), name='t1_to_mni_reg')
    t1_to_mni_reg.inputs.transforms = ['Rigid','Affine']
    t1_to_mni_reg.inputs.transform_parameters = [(0.1,),(0.1,)]
    t1_to_mni_reg.inputs.metric = ['MI','MI']
    t1_to_mni_reg.inputs.metric_weight = [1.0,1.0]
    t1_to_mni_reg.inputs.radius_or_number_of_bins = [32,32]
    t1_to_mni_reg.inputs.sampling_strategy = ['Regular','Regular']
    t1_to_mni_reg.inputs.sampling_percentage = [0.25,0.25]
    t1_to_mni_reg.inputs.number_of_iterations = [[1000,500,250,100],[1000,500,250,100]]
    t1_to_mni_reg.inputs.convergence_threshold = [1e-6,1e-6]
    t1_to_mni_reg.inputs.convergence_window_size = [10,10]
    t1_to_mni_reg.inputs.smoothing_sigmas = [[3,2,1,0],[3,2,1,0]]
    t1_to_mni_reg.inputs.sigma_units = ['vox','vox']
    t1_to_mni_reg.inputs.shrink_factors = [[8,4,2,1],[8,4,2,1]]
    t1_to_mni_reg.inputs.use_histogram_matching = [True,True]
    t1_to_mni_reg.inputs.output_warped_image = True
    t1_to_mni_reg.inputs.output_transform_prefix = "t1_to_mni_"
    t1_to_mni_reg.inputs.winsorize_lower_quantile = 0.005
    t1_to_mni_reg.inputs.winsorize_upper_quantile = 0.995
    t1_to_mni_reg.inputs.verbose = True
    t1_to_mni_reg.inputs.collapse_output_transforms = True
    t1_to_mni_reg.inputs.write_composite_transform = True

    # Here is a map node to perform mcflirt on each fmri input again
    mcflirt_mapnode = MapNode(
        fsl.MCFLIRT(),
        iterfield=['in_file'],   # only the "in_file" is iterated
        name='mcflirt_mapnode'
    )

    mcflirt_mapnode.inputs.cost = 'mutualinfo'
    mcflirt_mapnode.inputs.output_type = 'NIFTI_GZ'
    mcflirt_mapnode.inputs.save_mats = True

    # Here we set up a map node to apply the transform to all the images in the list
    apply_multi = MapNode(
        ants.ApplyTransforms(),
        name='apply_multi',
        iterfield=['input_image']  # We'll apply the same transforms to multiple images
    )

    apply_multi.inputs.dimension = 3
    apply_multi.inputs.input_image_type = 3     # 4D time-series are 3D + t
    apply_multi.inputs.interpolation = 'Linear'
    apply_multi.inputs.float = True
    apply_multi.inputs.reference_image = mni_template

    # ---------------------------- END NODE DEFINITION SECTION ------------------------
    #%%
    # ---------------------------- CONNECTION DEFINITION SECTION ------------------------
    # --- Connections ---
    wf.connect(inputnode, 'fmri_4d', mcflirt, 'in_file')
    wf.connect(inputnode, 't1_image', bet_t1, 'in_file')
    wf.connect(inputnode, 'mni_template', t1_to_mni_reg, 'fixed_image')

    # Motion correction
    wf.connect(mcflirt, 'out_file', mean_epi, 'in_file')

    # Rigid step: EPI->T1
    wf.connect(bet_t1, 'out_file', rigid_reg, 'fixed_image')
    wf.connect(mean_epi, 'out_file', rigid_reg, 'moving_image')

    # Nonlinear: EPI->T1
    wf.connect(inputnode, 't1_image', nonlinear_reg, 'fixed_image')
    wf.connect(bet_t1, 'mask_file', nonlinear_reg, 'fixed_image_masks')
    wf.connect(mean_epi, 'out_file', nonlinear_reg, 'moving_image')
    wf.connect(rigid_reg, 'forward_transforms', nonlinear_reg, 'initial_moving_transform')

    # T1->MNI
    wf.connect(inputnode, 't1_image', t1_to_mni_reg, 'moving_image')

    # T1->MNI composite first
    merge_transforms = pe.Node(Merge(2), name='merge_transforms')
    wf.connect(t1_to_mni_reg, 'composite_transform', merge_transforms, 'in1')

    # EPI->T1 composite second
    wf.connect(nonlinear_reg, 'composite_transform', merge_transforms, 'in2')

    # Connect the list of fMRI files to the apply_multi node
    wf.connect(merge_transforms, 'out', apply_multi, 'transforms')
    wf.connect(mean_epi, 'out_file', mcflirt_mapnode, 'ref_file')
    wf.connect(listnode, 'fmri_files', mcflirt_mapnode, 'in_file')
    wf.connect(mcflirt_mapnode, 'out_file', apply_multi, 'input_image')

    # Connect the output of the apply_multi node to the output node
    def save_apply_multi_output(output_images, base_dir):
        import os
        # Save the names of the output images in a file as well as printing them
        print("Apply Multi Output Images:")
        for img in output_images:
            print(img)
        # Save to a text file inside the base directory. How do we access the base_dir from inside this function?

        # Assuming wf is the workflow object, you can access its base_dir
        # by using wf.base_dir. However, you need to pass the workflow object to this function.
        # Alternatively, you can set the base_dir as a global variable or pass it as an argument.
        # Here we are assuming that wf is the workflow object
        # and we are using its base_dir attribute to save the output images.
        # Note: You might want to use os.path.join to create the full path for the output file
        # to avoid issues with different operating systems.

        output_file = os.path.join(base_dir, 'output_images.txt')
        with open(output_file, 'w') as f:
            for img in output_images:
                f.write(f"{img}\n")

        return output_images  # Important: Return the images for the pipeline to continue

    print_node = pe.Node(Function(input_names=['output_images', 'base_dir'],
                                output_names=['output_images'],
                                function=save_apply_multi_output),
                        name='save_apply_multi_output')
    print_node.inputs.base_dir = wf.base_dir

    wf.connect(apply_multi, 'output_image', print_node, 'output_images')

    outputnode = pe.Node(IdentityInterface(fields=['output_image']), name='outputnode')
    wf.connect(apply_multi, 'output_image', outputnode, 'output_image')

    if DO_DATASINK_DEBUG: # We probably don't need this very much - in fact it's a potential source of run time errors!

        #--- DataSink for Debug ---
        datasink = pe.Node(DataSink(), name='datasink')
        datasink.inputs.base_directory = os.path.join(base_dir,'debug_outputs')
        # --- DataSink connections ---
        wf.connect(mcflirt, 'out_file', datasink, 'mcflirt')
        wf.connect(mean_epi, 'out_file', datasink, 'mean_epi')
        wf.connect(bet_t1, 'out_file', datasink, 'bet_t1.@brain')
        wf.connect(bet_t1, 'mask_file', datasink, 'bet_t1.@mask')
        wf.connect(rigid_reg, 'warped_image', datasink, 'rigid_reg.@warped')
        wf.connect(nonlinear_reg, 'warped_image', datasink, 'nonlinear_reg.@warped')
        wf.connect(t1_to_mni_reg, 'warped_image', datasink, 't1_to_mni_reg.@warped')
        
    # ---------------------------- END CONNECTION DEFINITION SECTION ------------------------

    # --- Write graph and run ---
    wf.write_graph(graph2use='exec', format='png', simple_form=False)
    #wf.run(plugin='MultiProc', plugin_args={'n_procs': 2})
    wf.run(plugin='Linear') 
