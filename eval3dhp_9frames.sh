#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=testPoseMamba3DHP_S_9    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Starting eval job for PoseMamba on MPI-INF-3DHP (S, 9 frames)"


python train_3dhp.py --config configs/pose3d/testing/PoseMamba_train_3dhp_S_9.yaml --evaluate checkpoint/pose3d/PoseMamba_train_3dhp_S_9/best_epoch.bin --checkpoint eval/checkpoint_S_9
