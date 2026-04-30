"""
Compare Ground Truth and YOLO 2D poses for selected frames only.

Supports both MPI 3DHP and UCO Physical Rehabilitation datasets.

For MPI 3DHP:
python compare_gt_yolo_selected_frames.py --sequence TS1 --frames 0 10 20 --model-path runs/pose/mpi_yolo11x_pose_corrected3/weights/best.pt

For UCO dataset (folder/subfolder format):
python compare_gt_yolo_selected_frames.py --sequence 0/01 --camera cam0 --frames 0 10 20 --model-path runs/pose/yolo_model/weights/best.pt
"""

import argparse
import gc
import glob
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from compare_gt_yolo_2d import (  # noqa: E402
    CONNECTIONS_2D,
    check_gpu_availability,
    convert_coordinates_to_pixels,
    estimate_yolo_poses,
    load_test_3d_data_from_dataset,
    make_root_relative_2d_pixel,
)


# Dataset paths
MPI_TEST_PATHS = [
    '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set',
]

UCO_DATASET_PATH = '/nas-ctm01/datasets/public/UCO Physical Rehabilitation/dataset/clips_mp4'


def load_uco_ground_truth_2d(folder, subfolder, camera):
    """Load 2D ground truth poses from UCO _p2d.txt file
    
    Format: Each line contains x/y coordinate pairs for the visible keypoints.
    All keypoints are always visible.
    Maps 5 keypoint values to a 17-joint pose array for compatibility.
    """
    subfolder_name = f'{int(subfolder):02d}'
    gt_file = os.path.join(UCO_DATASET_PATH, str(folder), subfolder_name, f'{camera}_p2d.txt')
    
    if not os.path.exists(gt_file):
        print(f"❌ Ground truth file not found: {gt_file}")
        return None
    
    try:
        poses = []
        with open(gt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                values = [float(x) for x in line.split()]

                # Handle coordinate pairs per line: x1 y1 x2 y2 ...
                if len(values) >= 2:
                    # Create a 17-joint pose array for compatibility.
                    pose = np.zeros((17, 2))

                    keypoint_pairs = len(values) // 2
                    joint_indices = [1, 2, 3, 4, 5]

                    for i in range(min(keypoint_pairs, len(joint_indices))):
                        joint_idx = joint_indices[i]
                        pose[joint_idx, 0] = values[i * 2]
                        pose[joint_idx, 1] = values[i * 2 + 1]
                    
                    poses.append(pose)
        
        if poses:
            return np.array(poses)
        else:
            print(f"⚠ No valid poses found in {gt_file}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading ground truth from {gt_file}: {e}")
        return None


def load_uco_frames_from_video(folder, subfolder, camera, frame_indices=None):
    """Load frames from UCO video file"""
    subfolder_name = f'{int(subfolder):02d}'
    video_path = os.path.join(UCO_DATASET_PATH, str(folder), subfolder_name, f'{camera}.mp4')
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if frame_indices is None:
            # Load all frames
            frame_indices = list(range(total_frames))
        
        for frame_idx in frame_indices:
            if frame_idx >= total_frames:
                print(f"⚠ Frame {frame_idx} out of range (0-{total_frames-1})")
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠ Failed to read frame {frame_idx}")
                frames.append(None)
            else:
                frames.append(frame)
        
        cap.release()
        return frames, total_frames
        
    except Exception as e:
        print(f"❌ Error loading video: {e}")
        return None, 0


def load_sequence_image_paths(sequence_name):
    """Load image paths for MPI 3DHP dataset"""
    for base_path in MPI_TEST_PATHS:
        image_folder = os.path.join(base_path, sequence_name, 'imageSequence')
        if os.path.exists(image_folder):
            image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
            image_files.extend(glob.glob(os.path.join(image_folder, '*.png')))
            image_files.sort()

            if image_files:
                print(f"✓ Found images at: {os.path.abspath(image_folder)}")
                return image_files

    return None


def load_selected_frames(sequence_name, frame_indices, args=None):
    """Load selected frames from either MPI or UCO dataset"""
    
    # Check if this is UCO format (contains "/" for folder/subfolder)
    if '/' in sequence_name:
        parts = sequence_name.split('/')
        if len(parts) == 2:
            folder, subfolder = parts
            camera = args.camera if args and hasattr(args, 'camera') else 'cam0'
            
            # Load from UCO dataset
            frames_result = load_uco_frames_from_video(folder, subfolder, camera, frame_indices)
            if frames_result is None:
                return None, None
            
            frames, total_frames = frames_result
            
            # Filter out None frames and track valid indices
            valid_frames = []
            valid_indices = []
            for i, frame in enumerate(frames):
                if frame is not None:
                    valid_frames.append(frame)
                    if i < len(frame_indices):
                        valid_indices.append(frame_indices[i])
            
            if valid_frames:
                return valid_frames, valid_indices
            return None, None
    
    # Load from MPI 3DHP dataset
    image_files = load_sequence_image_paths(sequence_name)
    if image_files is None:
        print(f"❌ Could not find image folder for sequence {sequence_name}")
        return None, None

    frames = []
    valid_indices = []

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(image_files):
            print(f"⚠ Skipping frame {frame_idx}: out of range (0-{len(image_files) - 1})")
            continue

        frame = cv2.imread(image_files[frame_idx])
        if frame is None:
            print(f"⚠ Skipping frame {frame_idx}: failed to read image")
            continue

        frames.append(frame)
        valid_indices.append(frame_idx)

    if not frames:
        return None, None

    return frames, valid_indices


def draw_pose_on_image(image, pose, title, line_color, point_color, missing_text):
    output = image.copy()

    cv2.putText(
        output,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if pose is None or np.all(pose == 0):
        cv2.putText(
            output,
            missing_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    for joint1, joint2 in CONNECTIONS_2D:
        if joint1 < len(pose) and joint2 < len(pose):
            p1 = pose[joint1]
            p2 = pose[joint2]
            if np.allclose(p1, 0.0) or np.allclose(p2, 0.0):
                continue
            cv2.line(
                output,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                line_color,
                2,
                cv2.LINE_AA,
            )

    for joint_idx, joint_xy in enumerate(pose):
        if np.allclose(joint_xy, 0.0):
            continue
        center = (int(round(joint_xy[0])), int(round(joint_xy[1])))
        cv2.circle(output, center, 4, point_color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            str(joint_idx),
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def save_frame_comparison(image, gt_frame, yolo_frame, sequence_name, frame_idx, output_dir):
    gt_image = draw_pose_on_image(
        image,
        gt_frame,
        'Ground Truth',
        line_color=(255, 140, 0),
        point_color=(255, 80, 0),
        missing_text='No GT Data',
    )

    pred_image = draw_pose_on_image(
        image,
        yolo_frame,
        'YOLO Prediction',
        line_color=(0, 180, 255),
        point_color=(0, 255, 255),
        missing_text='No YOLO Detection',
    )

    frame_err = float('nan')
    if gt_frame is not None and yolo_frame is not None and not np.all(gt_frame == 0) and not np.all(yolo_frame == 0):
        frame_err = float(np.mean(np.linalg.norm(gt_frame - yolo_frame, axis=1)))

    if not np.isnan(frame_err):
        cv2.putText(
            pred_image,
            'Frame MPJPE: {:.2f}px'.format(frame_err),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            pred_image,
            'Frame MPJPE: N/A',
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    side_by_side = np.concatenate([gt_image, pred_image], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'frame_{:06d}_gt_vs_yolo.png'.format(frame_idx))
    cv2.imwrite(output_path, side_by_side)
    return output_path


def process_selected_frames(model, sequence_name, frame_indices, args, device='cpu'):
    # Check if this is UCO format (contains "/")
    is_uco_dataset = '/' in sequence_name
    
    if is_uco_dataset:
        # Load UCO ground truth
        parts = sequence_name.split('/')
        folder, subfolder = parts[0], parts[1]
        camera = args.camera if hasattr(args, 'camera') else 'cam0'
        
        gt_poses_2d = load_uco_ground_truth_2d(folder, subfolder, camera)
        if gt_poses_2d is None:
            print(f"❌ Failed to load ground truth data for {sequence_name}")
            return None
    else:
        # Load MPI 3DHP ground truth
        gt_poses_2d, _, _ = load_test_3d_data_from_dataset(sequence_name)
        if gt_poses_2d is None:
            print(f"❌ Failed to load ground truth data for {sequence_name}")
            return None

    frames, valid_indices = load_selected_frames(sequence_name, frame_indices, args)
    if frames is None:
        print(f"❌ Failed to load selected frames for {sequence_name}")
        return None

    selected_gt = []
    selected_frames = []
    selected_frame_indices = []

    for frame_idx, frame in zip(valid_indices, frames):
        if frame_idx >= len(gt_poses_2d):
            print(f"⚠ Skipping frame {frame_idx}: GT data not available")
            continue
        selected_gt.append(gt_poses_2d[frame_idx])
        selected_frames.append(frame)
        selected_frame_indices.append(frame_idx)

    if not selected_frames:
        print('❌ No valid frame pairs were found')
        return None

    print(f"✓ Processing {len(selected_frames)} selected frames for sequence {sequence_name}")

    yolo_poses_2d, _, performance_metrics = estimate_yolo_poses(
        model,
        selected_frames,
        args.img_size,
        device,
        batch_size=args.batch_size,
    )

    gt_poses_2d_pixel = convert_coordinates_to_pixels(np.array(selected_gt), selected_frames)
    yolo_poses_2d_pixel = convert_coordinates_to_pixels(yolo_poses_2d, selected_frames)

    sequence_output_dir = os.path.join(args.output_dir, sequence_name)
    os.makedirs(sequence_output_dir, exist_ok=True)

    saved_files = []
    for local_idx, frame_idx in enumerate(selected_frame_indices):
        saved_path = save_frame_comparison(
            selected_frames[local_idx],
            gt_poses_2d_pixel[local_idx],
            yolo_poses_2d_pixel[local_idx],
            sequence_name,
            frame_idx,
            sequence_output_dir,
        )
        saved_files.append(saved_path)
        print(f"✓ Saved comparison for frame {frame_idx} -> {saved_path}")

    print(f"\n✓ Done. Output folder: {os.path.abspath(sequence_output_dir)}")
    print(f"✓ Processed frames: {selected_frame_indices}")
    print(f"✓ Mean inference time: {performance_metrics['mean_inference_time'] * 1000:.2f} ms")
    print(f"✓ FPS: {performance_metrics['fps']:.2f}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Save GT vs YOLO comparisons for selected frames only')
    parser.add_argument('--sequence', type=str, required=True, 
                        help='Sequence to compare. For MPI 3DHP: TS1-TS6. For UCO dataset: folder/subfolder (e.g., 0/01)')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='Frame indices to process (0-based, space-separated)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained YOLO model (.pt file)')
    parser.add_argument('--camera', type=str, default='cam0', help='Camera to use for UCO dataset (cam0-cam4, default: cam0)')
    parser.add_argument('--output-dir', type=str, default='comparison_selected_frames', help='Directory to save output images')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for YOLO inference')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for YOLO inference (default: 32 on CUDA, 16 on CPU)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    args = parser.parse_args()

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')

    print('🎯 Selected-frame GT vs YOLO comparison')
    print('=' * 80)
    print(f'Sequence: {args.sequence}')
    print(f'Frames: {args.frames}')
    print(f'Model: {args.model_path}')
    print(f'Output dir: {args.output_dir}')
    print(f'Input size: {args.img_size}')
    if '/' in args.sequence:
        print(f'Camera: {args.camera} (UCO dataset)')
    if args.batch_size is not None:
        print(f'Inference batch size (override): {args.batch_size}')
    else:
        print('Inference batch size: auto (32 on CUDA, 16 on CPU)')
    print('=' * 80)

    if not os.path.exists(args.model_path):
        print(f'❌ Model not found: {args.model_path}')
        return

    if args.device == 'auto':
        device, gpu_available = check_gpu_availability()
    else:
        device = args.device
        gpu_available = device.startswith('cuda') and torch.cuda.is_available()

    print(f'🤖 Loading YOLO model from {args.model_path}...')
    try:
        model = YOLO(args.model_path)
        if gpu_available:
            print(f'📦 Moving model to {device}...')
            model.to(device)
        print('✓ YOLO model loaded successfully')
    except Exception as e:
        print(f'❌ Error loading YOLO model: {e}')
        return

    try:
        process_selected_frames(model, args.sequence, args.frames, args, device)
    finally:
        if gpu_available:
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()