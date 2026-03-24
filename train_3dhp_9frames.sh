#!/bin/bash
# Training script for PoseMamba-S on MPI-INF-3DHP with 9-frame temporal window

python train_3dhp.py --config configs/pose3d/PoseMamba_train_3dhp_S_9frames.yaml --checkpoint checkpoint/pose3d/PoseMamba_train_3dhp_S_9frames
