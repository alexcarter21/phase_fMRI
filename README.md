# phase_fMRI
Code to analyse fMRI data collected with binocular phase stimuli.


In this experiment participants were presented with 6Hz flickering disks in:
  - Chromaticity - luminance, L-M or S
  - Spatial frequency - disk or grating
  - Phase - in or anti-phase across eyes
For anti-phase conditions you would expect small responses if binocular as the inputs in each eye will cancel out.
Trials are 12s with 12s ITI, and presented in a random order.

Condition codes:
1) Achromatic Disk, 6Hz, in phase
2) Achromatic Disk, 6Hz, antiphase
3) L-M Disk, 6Hz in phase
4) L-M Disk, 6Hz, antiphase
5) S Disk, 6Hz in phase
6) S Disk, 6Hz, antiphase
7) Achromatic grat, 6Hz in phase
8) Achromatic grat, 6Hz, antiphase
9) L-M grat, 6Hz in phase
10) L-M grat, 6Hz, antiphase
11) S grat, 6Hz in phase
12) S grat, 6Hz, antiphase

phase1 code -> written in collaboration with supervisor. This preprocesses data for each participant (e.g. motion correction, alignment etc.) using nipype.
phase2 code -> This runs GLMs for each participant and saves these beta values. Also check alignment and beta maps per ppt.
phase3 code -> group level analysis, apply ROI templates, statistical analysis and visualising the data.
atlas_prep -> make sure the atlases are correct for what we need (V1 - restricted eccentricity to stimulus size, LGN - combine hemispheres).
