This is an example submit file that runs Freesurfer's recon all from a Docker container within HTCondor

## freesurfer_recon-all.sub
This is the HTCondor submit file. You will need to consider the following:
 1. You will need to specify a license.txt file for the Freesurfer license.
 2. You will need to specify input files (as Nifti or anything that Freesurfer can consume) and output files as a .tar.gz file by modifying the .sub file for your particular application.
 3. Check that you are using the desired Freesurfer version, e.g., freesurfer/freesurfer:7.4.1
 4. You may need to adjust the cpu, memory, and disk requests as needed.

 ## freesurfer_run_script_recon-all.bash
 This is the script that runs Freesurfer from the Docker container. You probably do not need to modify this.