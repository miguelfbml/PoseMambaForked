#!/bin/bash
#
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=compareUcoAngleSummary
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Running UCO GT vs PoseMamba angle summary"

export CUDA_LAUNCH_BLOCKING=1

python3 compare_uco_gt_pred_posemamba_angle_summary.py \
    --yolo-model "../zdemo/weights/yolo/best.pt" \
    --posemamba-config "../configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "../zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --output-file "uco_angle_summary.txt" \
    --img-size "640" \
    --batch-size "16" \
    --device "cuda:0" \
    --window-size "5" \
    --folders 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 \
    --subfolders 9 10 11 12 13 14 15 16 \
    --cameras cam0 cam1 cam2 cam3 cam4

echo "✓ Completed UCO GT vs PoseMamba angle summary"