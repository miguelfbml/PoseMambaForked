#!/bin/bash
#
#SBATCH --partition=gpu_min24gb     # Reserved partition
#SBATCH --qos=gpu_min24gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testPoseMambaB    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
echo "Starting training job for PoseMamba on Human3.6M"

python train.py --config configs/pose3d/PoseMamba_train_h36m_S.yaml --checkpoint checkpoint/pose3d/PoseMamba_train_h36m_S 