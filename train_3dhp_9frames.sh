#!/bin/bash
#
#SBATCH --partition=gpu_min24gb     # Reserved partition
#SBATCH --qos=gpu_min24gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train3dhpS    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

# Training script for PoseMamba-S on MPI-INF-3DHP with 9-frame temporal window

python train_3dhp.py --config configs/pose3d/PoseMamba_train_3dhp_S_9frames.yaml --checkpoint checkpoint/pose3d/PoseMamba_train_3dhp_S_9frames
