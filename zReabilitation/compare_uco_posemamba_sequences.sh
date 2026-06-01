#!/bin/bash
#
#SBATCH --partition=gpu_min32gb     # Reserved partition
#SBATCH --qos=gpu_min32gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=compareUcoPosemamba
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running UCO PoseMamba sequence rendering"
echo "Folders: 0-5 | Subfolders: 09-16 | Cameras: cam0-cam4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_LAUNCH_BLOCKING=1
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"

export TORCH_USE_CUDA_DSA=1

python3 "$SCRIPT_DIR/compare_uco_posemamba_sequences.py" \
    --yolo-model "../zdemo/weights/yolo/best.pt" \
    --posemamba-config "../configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "../zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --output-dir "uco_posemamba_sequence_output" \
    --img-size "640" \
    --batch-size "1" \
    --device "cuda:0" \
    --folders 0 1 2 3 4 5 \
    --subfolders 9 10 11 12 13 14 15 16 \
    --cameras cam0 cam1 cam2 cam3 cam4

echo "✓ Completed UCO PoseMamba rendering run"