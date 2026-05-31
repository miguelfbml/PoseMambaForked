#!/bin/bash
#
#SBATCH --partition=gpu_min11gb     # Reserved partition
#SBATCH --qos=gpu_min11gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=compareSelected  # Job name
#SBATCH --output=slurm_%x.%j.out    # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err     # File containing STDERR output. If ommited, use STDOUT.


echo "Running selected-frame GT vs PoseMamba 3D comparison"

python3 compare_gt_pred_posemamba_selected_frames.py \
    --sequence "TS1" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "zdemo/weights/yolo/best.pt" \
    --output-dir "comparison_posemamba_selected_frames" \
    --posemamba-config "configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --image-root "/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set" \
    --gt-data "motion3d/GT/data_test_3dhp.npz" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_pred_posemamba_selected_frames.py \
    --sequence "TS2" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "zdemo/weights/yolo/best.pt" \
    --output-dir "comparison_posemamba_selected_frames" \
    --posemamba-config "configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --image-root "/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set" \
    --gt-data "motion3d/GT/data_test_3dhp.npz" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_pred_posemamba_selected_frames.py \
    --sequence "TS3" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 5800 \
    --model-path "zdemo/weights/yolo/best.pt" \
    --output-dir "comparison_posemamba_selected_frames" \
    --posemamba-config "configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --image-root "/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set" \
    --gt-data "motion3d/GT/data_test_3dhp.npz" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_pred_posemamba_selected_frames.py \
    --sequence "TS4" \
    --frames 600 1200 1800 2400 3000 3600 4200 4800 5400 6000 \
    --model-path "zdemo/weights/yolo/best.pt" \
    --output-dir "comparison_posemamba_selected_frames" \
    --posemamba-config "configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --image-root "/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set" \
    --gt-data "motion3d/GT/data_test_3dhp.npz" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_pred_posemamba_selected_frames.py \
    --sequence "TS5" \
    --frames 30 60 90 120 150 180 210 240 270 300 \
    --model-path "zdemo/weights/yolo/best.pt" \
    --output-dir "comparison_posemamba_selected_frames" \
    --posemamba-config "configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --image-root "/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set" \
    --gt-data "motion3d/GT/data_test_3dhp.npz" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"


python3 compare_gt_pred_posemamba_selected_frames.py \
    --sequence "TS6" \
    --frames 50 100 150 200 250 300 350 400 450 490 \
    --model-path "zdemo/weights/yolo/best.pt" \
    --output-dir "comparison_posemamba_selected_frames" \
    --posemamba-config "configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml" \
    --posemamba-checkpoint "zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin" \
    --image-root "/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set" \
    --gt-data "motion3d/GT/data_test_3dhp.npz" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"
