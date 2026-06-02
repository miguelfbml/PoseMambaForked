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
CAMERAS=("cam0" "cam1" "cam2" "cam3" "cam4")
FRAMES=(0 30 60 90 120 150 180 210 240 270)

declare -A CAMERA_ERROR_SUM
declare -A CAMERA_ERROR_COUNT
declare -A EXERCISE_ERROR_SUM
declare -A EXERCISE_ERROR_COUNT

for camera in "${CAMERAS[@]}"; do
    for folder in $(seq 0 26); do
        for sub in 9 10 11 12 13 14 15 16; do
            subf=$(printf "%02d" "$sub")
            sequence="${folder}/${subf}"
            echo "Starting sequence processing: $sequence | $camera"
            run_output=$(python3 compare_uco_gt_pred_posemamba_angle_summary.py \
                --sequence "$sequence" \
                --camera "$camera" \
                --frames "${FRAMES[@]}" \
                --yolo-model "$YOLO_MODEL" \
                --posemamba-config "$POSEMAMBA_CONFIG" \
                --posemamba-checkpoint "$POSEMAMBA_CHECKPOINT" \
                --output-dir "$OUTPUT_DIR" \
                --img-size "640" \
                --batch-size "16" \
                --device "cuda:0" \
                --show-angle-diff 2>&1)
            printf '%s\n' "$run_output"

            result_line=$(printf '%s\n' "$run_output" | grep '^RESULT ' | tail -n 1)
            if [ -z "$result_line" ]; then
                echo "⚠ No summary line found for $sequence | $camera"
                continue
            fi

            mean_value=$(printf '%s\n' "$result_line" | sed -n 's/.*mean_angle_error_deg=\([^ ]*\).*/\1/p')
            valid_frames=$(printf '%s\n' "$result_line" | sed -n 's/.*valid_angle_frames=\([^ ]*\).*/\1/p')

            if [ -n "$mean_value" ] && [ "$mean_value" != "nan" ] && [ -n "$valid_frames" ] && [ "$valid_frames" -gt 0 ]; then
                camera_sum_key="$camera"
                exercise_sum_key="$subf"

                CAMERA_ERROR_SUM["$camera"]=$(awk -v a="${CAMERA_ERROR_SUM["$camera"]:-0}" -v b="$mean_value" -v n="$valid_frames" 'BEGIN { printf "%.10f", a + b * n }')
                CAMERA_ERROR_COUNT["$camera"]=$(awk -v a="${CAMERA_ERROR_COUNT["$camera"]:-0}" -v b="$valid_frames" 'BEGIN { printf "%d", a + b }')

                EXERCISE_ERROR_SUM["$subf"]=$(awk -v a="${EXERCISE_ERROR_SUM["$subf"]:-0}" -v b="$mean_value" -v n="$valid_frames" 'BEGIN { printf "%.10f", a + b * n }')
                EXERCISE_ERROR_COUNT["$subf"]=$(awk -v a="${EXERCISE_ERROR_COUNT["$subf"]:-0}" -v b="$valid_frames" 'BEGIN { printf "%d", a + b }')
            fi
        done
    done
done

echo
echo "=== Mean angle error per camera ==="
for camera in "${CAMERAS[@]}"; do
    count=${CAMERA_ERROR_COUNT["$camera"]:-0}
    sum=${CAMERA_ERROR_SUM["$camera"]:-0}
    if [ "$count" -gt 0 ]; then
        mean=$(awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.4f", s / c }')
        echo "$camera: $mean deg over $count valid frames"
    else
        echo "$camera: no valid angle frames"
    fi
done

echo
echo "=== Mean angle error per exercise ==="
for subf in 09 10 11 12 13 14 15 16; do
    count=${EXERCISE_ERROR_COUNT["$subf"]:-0}
    sum=${EXERCISE_ERROR_SUM["$subf"]:-0}
    if [ "$count" -gt 0 ]; then
        mean=$(awk -v s="$sum" -v c="$count" 'BEGIN { printf "%.4f", s / c }')
        echo "Exercise $subf: $mean deg over $count valid frames"
    else
        echo "Exercise $subf: no valid angle frames"
    fi
done

echo "✓ Completed sequence angle-summary UCO GT vs PoseMamba comparison"