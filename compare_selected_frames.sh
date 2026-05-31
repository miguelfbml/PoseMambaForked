#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=compareSelected  # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running selected-frame GT vs YOLO comparison"


python3 compare_gt_yolo_selected_frames.py \
    --sequence "TS1" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_yolo_selected_frames.py \
    --sequence "TS2" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"



python3 compare_gt_yolo_selected_frames.py \
    --sequence "TS3" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 5800 \
    --model-path "runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"



python3 compare_gt_yolo_selected_frames.py \
    --sequence "TS4" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"




python3 compare_gt_yolo_selected_frames.py \
    --sequence "TS5" \
    --frames 30 60 90 120 150 180 210 240 270 300 \
    --model-path "runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"




python3 compare_gt_yolo_selected_frames.py \
    --sequence "TS6" \
    --frames 50 100 150 200 250 300 350 400 450 490 \
    --model-path "runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt" \
    --output-dir "comparison_selected_frames" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"