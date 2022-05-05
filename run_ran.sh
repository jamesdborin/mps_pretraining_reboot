#!/bin/bash -l

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:00

# Request 1 gigabyte of RAM for each core/thread 
# (must be an integer followed by M, G, or T)
#$ -l mem=100G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=10G

# Set the name of the job.
#$ -N RAN

# Request 16 cores.
#$ -pe smp 36

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/ucapjmd/code/mps_pretraining_reboot

# load the python module
module load python/3.8.0

# 8. Run the application.
python /home/ucapjmd/code/mps_pretraining_reboot/variance_experiments.py ran