#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testPoseMambaB    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Starting eval job for PoseMamba on MPI-INF-3DHP"


python train.py --config checkpoint/pose3d/PoseMamba_B/config.yaml --evaluate checkpoint/pose3d/PoseMamba_B/best_epoch.bin --checkpoint eval/checkpointB