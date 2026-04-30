#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=compareSelected  # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running selected-frame GT vs YOLO comparison on UCO dataset"

# UCO Dataset format: folder/subfolder (e.g., 0/01)
# Camera options: cam0, cam1, cam2, cam3, cam4


# Process folder 0, subfolder 03 with cam1
python3 compare_gt_yolo_selected_frames.py \
    --sequence "0/03" \
    --camera "cam1" \
    --frames 0 10 20 \
    --model-path "weights/YOLO/best.pt" \
    --output-dir "comparison_selected_frames_uco" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"



echo "✓ Completed UCO dataset comparison"