import argparse
import os
from dataclasses import dataclass
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

from utils.data import read_pkl
from data.const import H36M_TO_MPI
from data.reader.motion_dataset import MPI3DHP, Fusion

# MPI-INF-3DHP skeleton connections
connections = [
    (10, 9), (9, 8), (8, 11), (8, 14), (14, 15), (15, 16),
    (11, 12), (12, 13), (8, 7), (7, 0), (0, 4), (0, 1),
    (1, 2), (2, 3), (4, 5), (5, 6)
]

def convert_h36m_to_mpi_connection():
    global connections
    new_connections = []
    for connection in connections:
        new_connection = (H36M_TO_MPI[connection[0]], H36M_TO_MPI[connection[1]])
        new_connections.append(new_connection)
    connections = new_connections

def read_h36m_gt(args):
    cam2real = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=np.float32)
    scale_factor = 0.298

    sample_joint_seq = read_pkl(f'../motion3d/H36M-243/test/%08d.pkl' % args.sequence_number)['data_label']
    sample_joint_seq = sample_joint_seq.transpose(1, 0, 2)  # (17, T, 3)
    sample_joint_seq = (sample_joint_seq / scale_factor) @ cam2real
    convert_h36m_to_mpi_connection()
    return sample_joint_seq

def read_mpi_gt(args):
    """Load ground truth exactly as in train_3dhp.py evaluation"""
    @dataclass
    class DatasetArgs:
        data_root: str
        n_frames: int
        stride: int
        flip: bool
        test_augmentation: bool
        data_augmentation: bool
        reverse_augmentation: bool
        out_all: int
        test_batch_size: int

    # Use same parameters as your model evaluation
    dataset_args = DatasetArgs(
        data_root='../motion3d/', 
        n_frames=27,  # Same as your model's n_frames
        stride=9,     # Same as your model's stride
        flip=False,
        test_augmentation=False,
        data_augmentation=False,
        reverse_augmentation=False,
        out_all=1,
        test_batch_size=1
    )
    
    dataset = Fusion(dataset_args, train=False)  # Use Fusion like train_3dhp.py
    
    print(f"Dataset length: {len(dataset)}")
    
    if args.sequence_number >= len(dataset):
        print(f"ERROR: Sequence {args.sequence_number} is out of range! Dataset has {len(dataset)} sequences.")
        return None

    # Get multiple samples to build a sequence
    sequence_data = []
    target_seq_name = None
    processed_samples = 0
    
    # Available sequence names in MPI-INF-3DHP test set
    available_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
    
    # If sequence_name is provided, use it; otherwise use sequence_number
    if args.sequence_name:
        target_seq_name = args.sequence_name
    else:
        target_seq_name = available_sequences[args.sequence_number % len(available_sequences)]
    
    print(f"Looking for sequence: {target_seq_name}")
    
    for i in range(len(dataset)):
        try:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = dataset[i]
            
            # Extract sequence name properly
            if isinstance(seq, (list, tuple)):
                current_seq_name = seq[0]
            elif isinstance(seq, torch.Tensor):
                current_seq_name = seq.item() if seq.numel() == 1 else str(seq)
            else:
                current_seq_name = str(seq)
            
            # Filter by target sequence
            if current_seq_name != target_seq_name:
                continue
            
            print(f"Found matching sequence: {current_seq_name} (sample {i})")
            
            # Process exactly like train_3dhp.py
            if isinstance(gt_3D, torch.Tensor):
                gt_3D = gt_3D.clone()
            else:
                gt_3D = torch.tensor(gt_3D)
                
            gt_3D = gt_3D.view(1, -1, 17, 3)  # N=1, T, 17, 3
            gt_3D[:, :, 14] = 0  # Set root joint to 0
            
            # Extract center frame (same as evaluation)
            center_frame = gt_3D.shape[1] // 2
            center_pose = gt_3D[0, center_frame]  # (17, 3)
            
            # Make root-relative (same as evaluation)
            center_pose = center_pose - center_pose[14:15, :]
            
            # Convert to numpy
            if hasattr(center_pose, 'cpu'):
                center_pose = center_pose.cpu().numpy()
            elif hasattr(center_pose, 'numpy'):
                center_pose = center_pose.numpy()
            else:
                center_pose = np.array(center_pose)
            
            sequence_data.append(center_pose)
            processed_samples += 1
            
            # Limit the number of samples for visualization
            if processed_samples >= 100:  # Collect up to 100 frames
                break
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Processed {processed_samples} samples for sequence {target_seq_name}")
    
    if not sequence_data:
        print("No valid ground truth data found")
        return None
    
    # Stack frames: (17, T, 3)
    sequence_3d = np.stack(sequence_data, axis=1)
    print(f"Stacked sequence shape: {sequence_3d.shape}")
    
    # Apply camera transformation
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    sequence_3d = sequence_3d @ cam2real
    
    convert_h36m_to_mpi_connection()
    print(f"Ground truth sequence: {target_seq_name}, shape: {sequence_3d.shape}")
    return sequence_3d, target_seq_name

def load_predictions(args, gt_seq_name):
    """Load predictions from inference_data.mat, matching the GT sequence"""
    data = scio.loadmat(args.inference_file)
    
    available_seqs = [key for key in data.keys() if not key.startswith('__')]
    print(f"Available sequences in inference file: {available_seqs}")
    
    # Use the ground truth sequence name
    seq_name = gt_seq_name
    
    if seq_name not in data:
        print(f"Sequence {seq_name} not found in inference file")
        # Try to match by index
        if args.sequence_number < len(available_seqs):
            seq_name = available_seqs[args.sequence_number]
            print(f"Using sequence by index: {seq_name}")
        else:
            raise ValueError(f"No matching sequence found")
    
    pred_3d = data[seq_name]
    print(f"Loaded prediction for sequence: {seq_name}, shape: {pred_3d.shape}")
    
    if pred_3d.ndim == 4:  # (3, 17, 1, T)
        pred_3d = pred_3d[:, :, 0, :]  # (3, 17, T)
        pred_3d = pred_3d.transpose(1, 2, 0)  # (17, T, 3)
    elif pred_3d.ndim == 3:  # (3, 17, T)
        pred_3d = pred_3d.transpose(1, 2, 0)  # (17, T, 3)
    
    # Apply camera transformation
    cam2real = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    pred_3d = pred_3d @ cam2real
    
    # The predictions from inference_data.mat are already processed correctly
    # They should already be in the right coordinate system
    
    return pred_3d
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence-number', type=int, default=0, help='Sequence index')
    parser.add_argument('--sequence-name', type=str, default=None, help='Specific sequence name (TS1, TS2, TS3, TS4, TS5, TS6)')
    parser.add_argument('--dataset', choices=['h36m', 'mpi'], default='mpi')
    parser.add_argument('--inference-file', type=str, default='../../checkpoint_mpi/inference_data.mat',
                        help='Path to inference_data.mat file')
    parser.add_argument('--frame-start', type=int, default=0, help='Starting frame for comparison')
    parser.add_argument('--num-frames', type=int, default=50, help='Number of frames to compare')
    args = parser.parse_args()

    print(f"Comparing sequence {args.sequence_number} of {args.dataset} dataset")

    # Load ground truth with the same method as train_3dhp.py
    if args.dataset == 'mpi':
        gt_result = read_mpi_gt(args)
        if gt_result is None:
            print("Failed to load ground truth data. Exiting.")
            return
        gt_joint_seq, gt_seq_name = gt_result
    else:
        gt_joint_seq = read_h36m_gt(args)
        gt_seq_name = f"h36m_seq_{args.sequence_number}"
    
    print(f"Ground truth shape: {gt_joint_seq.shape}")

    # Load predictions using the same sequence name
    pred_joint_seq = load_predictions(args, gt_seq_name)
    print(f"Prediction shape: {pred_joint_seq.shape}")

    # Now both GT and predictions should be synchronized
    # Both represent the same temporal sampling as used in evaluation
    
    # Ensure same number of frames
    num_frames = min(gt_joint_seq.shape[1], pred_joint_seq.shape[1])
    gt_joint_seq = gt_joint_seq[:, :num_frames, :]
    pred_joint_seq = pred_joint_seq[:, :num_frames, :]
    
    # Limit visualization frames if needed
    if num_frames > args.num_frames:
        start_frame = args.frame_start
        end_frame = min(start_frame + args.num_frames, num_frames)
        gt_joint_seq = gt_joint_seq[:, start_frame:end_frame, :]
        pred_joint_seq = pred_joint_seq[:, start_frame:end_frame, :]
        num_frames = end_frame - start_frame
        print(f"Limited visualization to frames {start_frame}-{end_frame} ({num_frames} frames)")
    
    print(f"Final synchronized frames: {num_frames}")

    # Calculate overall MPJPE for verification
    overall_error = np.mean(np.linalg.norm(gt_joint_seq - pred_joint_seq, axis=2))
    print(f"Overall MPJPE: {overall_error:.2f} mm")

    # Rest of visualization code remains the same...
    valid_gt = gt_joint_seq[~np.isnan(gt_joint_seq) & ~np.isinf(gt_joint_seq)]
    valid_pred = pred_joint_seq[~np.isnan(pred_joint_seq) & ~np.isinf(pred_joint_seq)]
    if valid_gt.size == 0:
        all_poses = valid_pred
    else:
        all_poses = np.vstack([valid_gt.reshape(-1, 3), valid_pred.reshape(-1, 3)])
    min_value = np.min(all_poses, axis=0)
    max_value = np.max(all_poses, axis=0)
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': '3d'})
    seq_title = gt_seq_name
    fig.suptitle(f'Ground Truth vs Prediction - {seq_title}\n(Synchronized Evaluation Frames: {num_frames})', fontsize=14)

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Set limits
        for ax in [ax1, ax2]:
            ax.set_xlim3d([min_value[0], max_value[0]])
            ax.set_ylim3d([min_value[1], max_value[1]])
            ax.set_zlim3d([min_value[2], max_value[2]])
            ax.set_xlabel('X', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_zlabel('Z', fontsize=10)

        # Plot ground truth
        ax1.set_title(f'Ground Truth\n(Frame {frame + 1}/{num_frames})', fontsize=12)
        x_gt = gt_joint_seq[:, frame, 0]
        y_gt = gt_joint_seq[:, frame, 1]
        z_gt = gt_joint_seq[:, frame, 2]
        
        # Draw GT connections
        for connection in connections:
            start = gt_joint_seq[connection[0], frame, :]
            end = gt_joint_seq[connection[1], frame, :]
            ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'b-', linewidth=2, alpha=0.8)
        
        ax1.scatter(x_gt, y_gt, z_gt, c='blue', s=60, alpha=0.9, edgecolors='darkblue', linewidth=0.5)
        ax1.scatter(x_gt[14], y_gt[14], z_gt[14], c='green', s=120, marker='*', alpha=1.0, edgecolors='darkgreen', linewidth=1)

        # Plot prediction
        ax2.set_title(f'Prediction\n(Frame {frame + 1}/{num_frames})', fontsize=12)
        x_pred = pred_joint_seq[:, frame, 0]
        y_pred = pred_joint_seq[:, frame, 1]
        z_pred = pred_joint_seq[:, frame, 2]
        
        # Draw prediction connections
        for connection in connections:
            start = pred_joint_seq[connection[0], frame, :]
            end = pred_joint_seq[connection[1], frame, :]
            ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', linewidth=2, alpha=0.8)
        
        ax2.scatter(x_pred, y_pred, z_pred, c='red', s=60, alpha=0.9, edgecolors='darkred', linewidth=0.5)
        ax2.scatter(x_pred[14], y_pred[14], z_pred[14], c='green', s=120, marker='*', alpha=1.0, edgecolors='darkgreen', linewidth=1)

        # Calculate frame-wise error
        frame_error = np.mean(np.linalg.norm(gt_joint_seq[:, frame, :] - pred_joint_seq[:, frame, :], axis=1))
        fig.suptitle(f'Ground Truth vs Prediction - {seq_title}\n(Frame {frame + 1}/{num_frames}, Error: {frame_error:.1f}mm)', fontsize=14)

        return ax1, ax2

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=150, repeat=True, blit=False)
    output_path = f'../mpi_comparison_{seq_title.lower()}_synced.gif'
    ani.save(output_path, writer='pillow', fps=8, dpi=100)
    print(f"Synchronized comparison GIF saved to: {output_path}")

    # Save static image
    update(0)
    plt.tight_layout()
    plt.savefig(output_path.replace('.gif', '.png'), dpi=150, bbox_inches='tight')
    print(f"Static comparison image saved to: {output_path.replace('.gif', '.png')}")

if __name__ == '__main__':
    main()