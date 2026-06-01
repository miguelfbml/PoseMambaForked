#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=countUcoYolo     # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running YOLO pose counting across UCO videos"
echo "Folders: 0-26 | Subfolders: 09-16 | Cameras: cam0-cam4"

python3 count_uco_yolo_sequences.py \
    --model-path "weights/YOLO/best.pt" \
    --output-dir "uco_yolo_sequence_output" \
    --img-size "640" \
    --batch-size "16" \
    --confidence "0.35" \
    --device "cuda:0" \
    --no-save-videos \
    --cameras cam0 cam1 cam2 cam3 cam4

echo "✓ Completed UCO YOLO counting run"