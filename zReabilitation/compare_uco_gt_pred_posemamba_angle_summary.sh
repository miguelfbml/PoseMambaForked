#!/bin/bash
#
#SBATCH --partition=gpu_min32gb
#SBATCH --qos=gpu_min32gb
#SBATCH --job-name=compareUcoGtPoseMamba
#SBATCH --output=slurm_%x.%j.out
#SBATCH --error=slurm_%x.%j.err

echo "Running sequence angle-summary UCO GT vs PoseMamba comparison"

export CUDA_LAUNCH_BLOCKING=1

YOLO_MODEL="../zdemo/weights/yolo/best.pt"
POSEMAMBA_CONFIG="../configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml"
POSEMAMBA_CHECKPOINT="../zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin"
OUTPUT_DIR="uco_gt_pred_selected_frames"
CAMERA="cam0"
FRAMES=(0 30 60 90 120 150 180 210 240 270)
SEQUENCES=("1/09" "1/10" "1/11" "1/12" "1/13" "1/14" "1/15" "1/16")

for sequence in "${SEQUENCES[@]}"; do
    echo "Starting sequence processing: $sequence | $CAMERA"
    python3 compare_uco_gt_pred_posemamba_angle_summary.py \
        --sequence "$sequence" \
        --camera "$CAMERA" \
        --frames "${FRAMES[@]}" \
        --yolo-model "$YOLO_MODEL" \
        --posemamba-config "$POSEMAMBA_CONFIG" \
        --posemamba-checkpoint "$POSEMAMBA_CHECKPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --img-size "640" \
        --batch-size "16" \
        --device "cuda:0" \
        --show-angle-diff
done

echo "✓ Completed sequence angle-summary UCO GT vs PoseMamba comparison"