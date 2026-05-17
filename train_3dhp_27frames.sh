#!/bin/bash
#
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train3dhpS27Frames    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
echo "Starting training job for PoseMamba on MPI-INF-3DHP"

python train_3dhp.py --config configs/pose3d/PoseMamba_train_3dhp_S_27.yaml --checkpoint checkpoint/pose3d/PoseMamba_train_3dhp_S_27 