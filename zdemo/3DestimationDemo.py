"""
Real-time 3D Pose Estimation using YOLO + TCPFormer
Collects 9 frames of 2D poses and predicts 3D poses with live 3D visualization

Usage:
python 3Destimation.py
python 3Destimation.py --camera 0 --conf 0.5
python 3Destimation.py --show-fps
python 3Destimation.py --no-viz  # Disable 3D visualization for faster performance
"""

import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path to import TCPFormer modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.learning import load_model_TCPFormer
from utils.tools import get_config

# MPI-INF-3DHP joint names and connections for visualization
JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand', 
    'RShoulder', 'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle',
    'RHip', 'RKnee', 'RAnkle', 'Sacrum', 'Spine', 'Neck'
]

CONNECTIONS_3D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

# Default paths
DEFAULT_YOLO_MODEL_PATH = 'weights/yolo/best.pt'
DEFAULT_TCPFORMER_CONFIG = '../configs/mpi/TCPFormer_mpi_9.yaml'
DEFAULT_TCPFORMER_CHECKPOINT = 'weights/frames9/best_epoch.pth.tr'

def normalize_screen_coordinates(X, w, h):
    """Normalize 2D keypoints to [-1, 1] range"""
    assert X.shape[-1] == 2
    return X / w * 2 - [1, h / w]

def apply_upright_correction(poses_3d):
    """Rotate poses around X-axis by 90 degrees to make figures upright, then around Z-axis by 90 degrees, then another 45 degrees."""
    rotation_x_90 = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)
    rotation_z_90 = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float32)
    # Rotation by 45 degrees around Z axis
    theta = np.deg2rad(45)
    rotation_z_45 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=np.float32)
    # Apply all rotations: X 90°, Z 90°, then Z 45°
    return poses_3d @ rotation_x_90.T @ rotation_z_90.T @ rotation_z_45.T

def make_root_relative_3d(poses_3d, root_joint_idx=14):
    """Subtract root joint position from all joints."""
    root_pos = poses_3d[root_joint_idx]
    root_relative_poses = poses_3d - root_pos[np.newaxis, :]
    root_relative_poses[root_joint_idx] = [0.0, 0.0, 0.0]
    return root_relative_poses

def scale_pose_to_max(pose_3d, max_value=900):
    """Scale pose so that the maximum absolute coordinate is max_value"""
    # Find maximum absolute value across all coordinates
    max_coord = np.max(np.abs(pose_3d))
    
    # Avoid division by zero
    if max_coord < 1e-6:
        return pose_3d
    
    # Calculate scaling factor
    scale_factor = max_value / max_coord
    
    # Apply scaling
    scaled_pose = pose_3d * scale_factor
    
    return scaled_pose

def check_gpu():
    """Check if GPU is available"""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        device = 'cuda:0'
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠ GPU not available, using CPU")
    return device

def setup_3d_plot(coord_range=1000):
    """Setup matplotlib 3D plot with reusable plot objects"""
    plt.ion()  # Enable interactive mode
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up static elements (axes, labels, limits)
    ax.set_xlim3d([-coord_range, coord_range])
    ax.set_ylim3d([-coord_range, coord_range])
    ax.set_zlim3d([-coord_range, coord_range])
    ax.set_xlabel('X (mm, right)', fontsize=12)
    ax.set_ylabel('Y (mm, forward)', fontsize=12)
    ax.set_zlabel('Z (mm, up)', fontsize=12)
    
    # Create empty plot objects that will be updated
    skeleton_lines = []
    for _ in CONNECTIONS_3D:
        line, = ax.plot([], [], [], 'b-', linewidth=3, alpha=0.8)
        skeleton_lines.append(line)
    
    # Joint scatter plots
    joint_scatter = ax.scatter([], [], [], c='blue', s=80, alpha=0.9,
                              edgecolors='darkblue', linewidth=2)
    
    # Root joint (static green star at origin)
    root_scatter = ax.scatter([0], [0], [0], c='green', s=200, marker='*', 
                             alpha=1.0, edgecolors='darkgreen', linewidth=3)
    
    # Coordinate axes (static)
    ax.plot([0, 100], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.7)
    ax.plot([0, 0], [0, 100], [0, 0], 'g-', linewidth=2, alpha=0.7)
    ax.plot([0, 0], [0, 0], [0, 100], 'b-', linewidth=2, alpha=0.7)
    ax.text(100, 0, 0, 'X(R)', fontsize=10, color='red')
    ax.text(0, 100, 0, 'Y(F)', fontsize=10, color='green')
    ax.text(0, 0, 100, 'Z(U)', fontsize=10, color='blue')
    
    # Joint text labels
    joint_texts = []
    for _ in range(17):
        text = ax.text(0, 0, 0, '', fontsize=11, color='blue', weight='bold')
        joint_texts.append(text)
    
    ax.view_init(elev=15, azim=45)
    
    # Store plot objects for updates
    plot_objects = {
        'skeleton_lines': skeleton_lines,
        'joint_scatter': joint_scatter,
        'root_scatter': root_scatter,
        'joint_texts': joint_texts
    }
    
    return fig, ax, plot_objects

def update_3d_plot_fast(fig, ax, plot_objects, pose_3d, frame_num):
    """Fast update of 3D plot by only changing data, not recreating objects"""
    
    # Update title
    ax.set_title(f'Real-time 3D Pose Estimation - Frame {frame_num}', fontsize=14, pad=20)
    
    # Update skeleton lines
    for idx, connection in enumerate(CONNECTIONS_3D):
        joint1, joint2 = connection
        if joint1 < len(pose_3d) and joint2 < len(pose_3d):
            x1, y1, z1 = pose_3d[joint1]
            x2, y2, z2 = pose_3d[joint2]
            plot_objects['skeleton_lines'][idx].set_data([x1, x2], [y1, y2])
            plot_objects['skeleton_lines'][idx].set_3d_properties([z1, z2])
        else:
            # Hide line if joints don't exist
            plot_objects['skeleton_lines'][idx].set_data([], [])
            plot_objects['skeleton_lines'][idx].set_3d_properties([])
    
    # Update joint positions (exclude root joint at index 14)
    non_root_joints = [i for i in range(len(pose_3d)) if i != 14]
    if non_root_joints:
        joint_positions = pose_3d[non_root_joints]
        plot_objects['joint_scatter']._offsets3d = (
            joint_positions[:, 0],
            joint_positions[:, 1],
            joint_positions[:, 2]
        )
    
    # Update joint text labels
    for joint_idx, (x, y, z) in enumerate(pose_3d):
        if joint_idx != 14:  # Skip root joint
            plot_objects['joint_texts'][joint_idx].set_position((x+30, y+30))
            plot_objects['joint_texts'][joint_idx].set_3d_properties(z+30)
            plot_objects['joint_texts'][joint_idx].set_text(str(joint_idx))
        else:
            plot_objects['joint_texts'][joint_idx].set_text('')
    
    # Redraw only what's necessary
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def print_3d_keypoints(prediction_num, keypoints_3d, print_output=True):
    """Print 3D keypoint coordinates in the same format as generate3D.py"""
    # Apply upright correction and make root-relative
    keypoints_3d_corrected = apply_upright_correction(keypoints_3d)
    keypoints_3d_root_rel = make_root_relative_3d(keypoints_3d_corrected, root_joint_idx=14)
    
    # Scale pose to max 900
    keypoints_3d_scaled = scale_pose_to_max(keypoints_3d_root_rel, max_value=900)
    
    if print_output:
        print(f"\n{'='*80}")
        print(f"3D Pose Prediction #{prediction_num}")
        print(f"{'='*80}")
        
        # Print header
        print(f"{'Joint':<20} {'X (mm, right)':<20} {'Y (mm, forward)':<20} {'Z (mm, up)':<20}")
        print(f"{'-'*80}")
        
        # Print each joint's 3D coordinates (17 joints)
        for idx in range(17):
            if idx < len(keypoints_3d_scaled):
                x, y, z = keypoints_3d_scaled[idx]
                joint_name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Joint_{idx}"
                
                # Highlight root joint
                if idx == 14:
                    print(f"{joint_name:<20} {x:<20.4f} {y:<20.4f} {z:<20.4f}  ← ROOT")
                else:
                    print(f"{joint_name:<20} {x:<20.4f} {y:<20.4f} {z:<20.4f}")
            else:
                joint_name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f"Joint_{idx}"
                print(f"{joint_name:<20} {0.0:<20.4f} {0.0:<20.4f} {0.0:<20.4f}")
        
        print(f"{'='*80}\n")
    
    return keypoints_3d_scaled

def main():
    parser = argparse.ArgumentParser(description='Real-time 3D Pose Estimation with YOLO + TCPFormer')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size for YOLO inference (default: 640)')
    parser.add_argument('--show-fps', action='store_true',
                       help='Show FPS counter in terminal')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable 3D visualization (faster performance)')
    parser.add_argument('--no-print', action='store_true',
                       help='Disable printing 3D coordinates (faster performance)')
    parser.add_argument('--fill-missing', action='store_true',
                       help='When YOLO fails, repeat last pose into the buffer to keep it full')
    parser.add_argument('--pose-skip', type=int, default=1,
                       help='Subsample factor for 2D poses (1=use every frame, 2=every 2nd frame, etc.)')
    parser.add_argument('--tcpformer-config', type=str, default=DEFAULT_TCPFORMER_CONFIG,
                       help='Path to TCPFormer config file')
    parser.add_argument('--tcpformer-checkpoint', type=str, default=DEFAULT_TCPFORMER_CHECKPOINT,
                       help='Path to TCPFormer checkpoint file')
    parser.add_argument('--coord-range', type=int, default=1000,
                       help='3D visualization coordinate range (default: 1000mm)')
    args = parser.parse_args()
    
    print("="*60)
    print("Real-time 3D Pose Estimation with YOLO + TCPFormer")
    print("="*60)
    print(f"YOLO Model: {DEFAULT_YOLO_MODEL_PATH}")
    print(f"TCPFormer Config: {args.tcpformer_config}")
    print(f"TCPFormer Checkpoint: {args.tcpformer_checkpoint}")
    print(f"Camera: {args.camera}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Image size: {args.img_size}")
    print(f"3D Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print(f"Print 3D Coordinates: {'Disabled' if args.no_print else 'Enabled'}")
    print("="*60)
    if not args.no_viz:
        print("Close 3D visualization window or press Ctrl+C to quit")
    else:
        print("Press Ctrl+C to quit")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists(DEFAULT_YOLO_MODEL_PATH):
        print(f"❌ Error: YOLO model not found at {DEFAULT_YOLO_MODEL_PATH}")
        return
    
    if not os.path.exists(args.tcpformer_config):
        print(f"❌ Error: TCPFormer config not found at {args.tcpformer_config}")
        return
    
    if not os.path.exists(args.tcpformer_checkpoint):
        print(f"❌ Error: TCPFormer checkpoint not found at {args.tcpformer_checkpoint}")
        return
    
    # Check GPU availability
    device = check_gpu()
    
    # Load YOLO model
    print(f"Loading YOLO model...")
    try:
        yolo_model = YOLO(DEFAULT_YOLO_MODEL_PATH)
        yolo_model.to(device)
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    # Load TCPFormer model
    print(f"Loading TCPFormer model...")
    try:
        tcpformer_args = get_config(args.tcpformer_config)
        tcpformer_model = load_model_TCPFormer(tcpformer_args)
        
        # Load checkpoint
        checkpoint = torch.load(args.tcpformer_checkpoint, map_location='cpu')
        
        # Handle DataParallel prefix in checkpoint
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # Load state dict
        tcpformer_model.load_state_dict(new_state_dict, strict=False)
        
        if torch.cuda.is_available():
            tcpformer_model = torch.nn.DataParallel(tcpformer_model)
            tcpformer_model = tcpformer_model.cuda()
        
        tcpformer_model.eval()
        print("✓ TCPFormer model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading TCPFormer model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual camera dimensions
    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Camera opened successfully ({camera_width}x{camera_height})")
    
    # Setup 3D plot only after we have enough 2D poses (buffer full)
    fig, ax, plot_objects = None, None, None
    viz_initialized = False
    
    print("Starting real-time detection...")
    
    # Sliding window buffer for collecting 9 frames
    n_frames = 9
    pose_buffer = []

    # Subsample 2D poses from YOLO frames (1 = every frame, 2 = every 2nd frame, etc.)
    pose_subsample = max(1, args.pose_skip)
    
    # FPS calculation variables for YOLO inference
    yolo_inference_times = []
    yolo_window_size = 30
    current_yolo_fps = 0
    
    # FPS calculation variables for TCPFormer inference (3D pose estimation)
    tcpformer_inference_times = []
    tcpformer_window_size = 30
    current_tcpformer_fps = 0
    
    # FPS calculation variables for 3D visualization update
    viz_update_times = []
    viz_window_size = 30
    current_viz_fps = 0
    
    # FPS calculation variables for end-to-end processing (per 3D prediction update)
    end_to_end_times = []
    end_to_end_window_size = 30
    current_end_to_end_fps = 0
    last_prediction_time = None
    
    frame_count = 0
    prediction_count = 0
    buffer_full_once = False  # Flag to start end-to-end timing after first full buffer
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Measure YOLO inference time
            yolo_start = time.time()
            
            # Run YOLO inference
            results = yolo_model.predict(
                frame,
                verbose=False,
                imgsz=args.img_size,
                conf=args.conf,
                device=device
            )
            
            yolo_end = time.time()
            yolo_time = yolo_end - yolo_start
            
            # Store YOLO inference time and calculate FPS
            yolo_inference_times.append(yolo_time)
            if len(yolo_inference_times) > yolo_window_size:
                yolo_inference_times.pop(0)
            
            avg_yolo_time = np.mean(yolo_inference_times)
            current_yolo_fps = 1.0 / avg_yolo_time if avg_yolo_time > 0 else 0
            
            # Process results
            if (results and len(results) > 0 and 
                hasattr(results[0], 'keypoints') and 
                results[0].keypoints is not None and 
                len(results[0].keypoints.xy) > 0):
                
                # Get keypoints for first detection
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                confidences = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)
                
                # Create pose_2d array (17, 3) with x, y, confidence
                pose_2d = np.zeros((17, 3), dtype=np.float32)
                n_kpts = min(17, keypoints.shape[0])
                pose_2d[:n_kpts, :2] = keypoints[:n_kpts]
                pose_2d[:n_kpts, 2] = confidences[:n_kpts]
                
# Add to buffer (sliding window) with subsampling
                # (e.g., pose_subsample=2 will store every 2nd detected pose)
                if frame_count % pose_subsample == 0:
                    pose_buffer.append(pose_2d)

                    # Maintain sliding window of 27 frames
                    if len(pose_buffer) > n_frames:
                        pose_buffer.pop(0)  # Remove oldest frame
                
                # Show buffer status with all FPS metrics
                if args.show_fps:
                    base_str = f"Buffer: {len(pose_buffer)}/{n_frames} | YOLO: {current_yolo_fps:.1f} FPS"
                    if len(tcpformer_inference_times) > 0:
                        base_str += f" | TCPFormer: {current_tcpformer_fps:.1f} FPS"
                    if not args.no_viz and len(viz_update_times) > 0:
                        base_str += f" | 3D Viz: {current_viz_fps:.1f} FPS"
                    if current_end_to_end_fps > 0:
                        base_str += f" | End-to-End: {current_end_to_end_fps:.1f} FPS"
                    print(base_str, end='\r')
                
                # (prediction step moved below so it runs even when YOLO fails)
            
            else:
                # Optionally keep buffer full by repeating the last pose when detection fails
                if args.fill_missing and len(pose_buffer) > 0:
                    pose_buffer.append(pose_buffer[-1])
                    if len(pose_buffer) > n_frames:
                        pose_buffer.pop(0)

                if args.show_fps:
                    base_str = f"No person detected | Frame: {frame_count} | YOLO: {current_yolo_fps:.1f} FPS"
                    if len(tcpformer_inference_times) > 0:
                        base_str += f" | TCPFormer: {current_tcpformer_fps:.1f} FPS"
                    if not args.no_viz and len(viz_update_times) > 0:
                        base_str += f" | 3D Viz: {current_viz_fps:.1f} FPS"
                    if current_end_to_end_fps > 0:
                        base_str += f" | End-to-End: {current_end_to_end_fps:.1f} FPS"
                    print(base_str, end='\r')

            # Run TCPFormer prediction when we have exactly 27 frames (even if YOLO missed)
            if len(pose_buffer) == n_frames:
                if not buffer_full_once:
                    buffer_full_once = True  # Start end-to-end timing after first full buffer

                    # Initialize 3D visualization only after buffer is full
                    if not args.no_viz and not viz_initialized:
                        fig, ax, plot_objects = setup_3d_plot(args.coord_range)
                        viz_initialized = True
                        print("✓ 3D visualization window opened")

                prediction_count += 1

                # Stack poses: (27, 17, 3) -> normalize only x,y
                poses_2d = np.stack(pose_buffer, axis=0)  # (27, 17, 3)

                # Normalize x, y coordinates to [-1, 1]
                poses_2d_normalized = poses_2d.copy()
                poses_2d_normalized[:, :, :2] = normalize_screen_coordinates(
                    poses_2d[:, :, :2], 
                    camera_width, 
                    camera_height
                )

                # Prepare input for TCPFormer: (batch=1, T=27, J=17, C=3)
                input_2d = torch.from_numpy(poses_2d_normalized).unsqueeze(0).float()

                if torch.cuda.is_available():
                    input_2d = input_2d.cuda()

                # Measure TCPFormer inference time
                tcpformer_start = time.time()

                # Run TCPFormer inference
                with torch.no_grad():
                    pred_3d = tcpformer_model(input_2d)  # (1, 27, 17, 3)

                tcpformer_end = time.time()
                tcpformer_time = tcpformer_end - tcpformer_start

                # Store TCPFormer inference time and calculate FPS
                tcpformer_inference_times.append(tcpformer_time)
                if len(tcpformer_inference_times) > tcpformer_window_size:
                    tcpformer_inference_times.pop(0)

                avg_tcpformer_time = np.mean(tcpformer_inference_times)
                current_tcpformer_fps = 1.0 / avg_tcpformer_time if avg_tcpformer_time > 0 else 0

                # Get middle frame prediction (frame 4, index 4)
                middle_frame_idx = n_frames // 2
                pred_3d_middle = pred_3d[0, middle_frame_idx].cpu().numpy()  # (17, 3)

                # Print 3D keypoints with upright correction, root-relative, and scaling
                pose_3d_viz = print_3d_keypoints(prediction_count, pred_3d_middle, print_output=not args.no_print)

                # Update end-to-end FPS (based on the rate of new 3D predictions)
                now = time.time()
                if last_prediction_time is not None:
                    delta = now - last_prediction_time
                    end_to_end_times.append(delta)
                    if len(end_to_end_times) > end_to_end_window_size:
                        end_to_end_times.pop(0)
                    avg_delta = np.mean(end_to_end_times)
                    current_end_to_end_fps = 1.0 / avg_delta if avg_delta > 0 else 0
                last_prediction_time = now

                # Update 3D visualization only if enabled and we have a valid end-to-end update rate
                if not args.no_viz and current_end_to_end_fps > 0:
                    viz_start = time.time()
                    update_3d_plot_fast(fig, ax, plot_objects, pose_3d_viz, prediction_count)
                    viz_end = time.time()
                    viz_time = viz_end - viz_start

                    # Store visualization update time and calculate FPS
                    viz_update_times.append(viz_time)
                    if len(viz_update_times) > viz_window_size:
                        viz_update_times.pop(0)

                    avg_viz_time = np.mean(viz_update_times)
                    current_viz_fps = 1.0 / avg_viz_time if avg_viz_time > 0 else 0

            # Check if matplotlib window is still open (only if visualization is enabled)
            if not args.no_viz and fig is not None:
                if not plt.fignum_exists(fig.number):
                    print("\n3D visualization window closed. Exiting...")
                    break
            
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if not args.no_viz:
            plt.close('all')
        print("\n✓ Camera released and windows closed")
        print(f"\n{'='*60}")
        print("Performance Summary:")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total 3D predictions: {prediction_count}")
        print(f"Average YOLO FPS: {current_yolo_fps:.1f}")
        if len(tcpformer_inference_times) > 0:
            print(f"Average TCPFormer FPS: {current_tcpformer_fps:.1f} (throughput after 27 frames)")
        if not args.no_viz and len(viz_update_times) > 0:
            print(f"Average 3D Visualization FPS: {current_viz_fps:.1f}")
        if current_end_to_end_fps > 0:
            print(f"Average End-to-End FPS: {current_end_to_end_fps:.1f}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()