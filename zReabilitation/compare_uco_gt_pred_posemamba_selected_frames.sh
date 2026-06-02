#!/bin/bash
#
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=compareUcoGtPoseMamba
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Running selected-frame UCO GT vs PoseMamba comparison"

export CUDA_LAUNCH_BLOCKING=1

YOLO_MODEL="../zdemo/weights/yolo/best.pt"
POSEMAMBA_CONFIG="../configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml"
POSEMAMBA_CHECKPOINT="../zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin"
OUTPUT_DIR="uco_gt_pred_selected_frames"
CAMERA="cam0"
FRAMES=(0 30 60 90 120 150 180 210 240 270)

python3 compare_uco_gt_pred_posemamba_selected_frames.py \
    --sequence "0/01" \
    --camera "$CAMERA" \
    --frames "${FRAMES[@]}" \
    --yolo-model "$YOLO_MODEL" \
    --posemamba-config "$POSEMAMBA_CONFIG" \
    --posemamba-checkpoint "$POSEMAMBA_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "640" \
    --batch-size "16" \
    --device "cuda:0"

python3 compare_uco_gt_pred_posemamba_selected_frames.py \
    --sequence "0/02" \
    --camera "$CAMERA" \
    --frames "${FRAMES[@]}" \
    --yolo-model "$YOLO_MODEL" \
    --posemamba-config "$POSEMAMBA_CONFIG" \
    --posemamba-checkpoint "$POSEMAMBA_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "640" \
    --batch-size "16" \
    --device "cuda:0"

python3 compare_uco_gt_pred_posemamba_selected_frames.py \
    --sequence "1/01" \
    --camera "$CAMERA" \
    --frames "${FRAMES[@]}" \
    --yolo-model "$YOLO_MODEL" \
    --posemamba-config "$POSEMAMBA_CONFIG" \
    --posemamba-checkpoint "$POSEMAMBA_CHECKPOINT" \
    --output-dir "$OUTPUT_DIR" \
    --img-size "640" \
    --batch-size "16" \
    --device "cuda:0"

echo "✓ Completed selected-frame UCO GT vs PoseMamba comparison"