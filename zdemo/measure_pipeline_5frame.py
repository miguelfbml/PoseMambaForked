"""
Pipeline Performance Measurement for 5-Frame 3D Pose Estimation

Measures real timings for:
- Frame capture
- YOLO 2D detection
- PoseMamba 3D prediction (on center frame #3 of 5-frame buffer)
- Visualization
- Total end-to-end latency

Usage:
    python measure_pipeline_5frame.py --camera 0
    python measure_pipeline_5frame.py --camera 0 --show-results-every 30
    python measure_pipeline_5frame.py --camera 0 --no-viz
"""

import argparse
import os
import sys
import time
from collections import deque
from typing import Optional, Tuple
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project to path
ZDEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ZDEMO_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from ultralytics import YOLO
except ImportError:
    from ultralytics.models.yolo.model import YOLO

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from lib.utils.utils_data import flip_data


# ============================================================================
# Timing Measurement Classes
# ============================================================================

class TimingStats:
    """Container for timing measurements."""
    
    def __init__(self, name: str, window_size: int = 100):
        self.name = name
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
    
    def add(self, time_ms: float) -> None:
        """Add a measurement in milliseconds."""
        self.times.append(time_ms)
    
    def stats(self) -> dict:
        """Get statistics (mean, p95, max)."""
        if not self.times:
            return {'mean': 0, 'p95': 0, 'max': 0, 'count': 0}
        
        times_arr = np.array(list(self.times))
        return {
            'mean': float(np.mean(times_arr)),
            'p95': float(np.percentile(times_arr, 95)),
            'max': float(np.max(times_arr)),
            'min': float(np.min(times_arr)),
            'std': float(np.std(times_arr)),
            'count': len(self.times),
        }
    
    def print(self) -> None:
        """Print formatted statistics."""
        s = self.stats()
        if s['count'] > 0:
            print(
                f"{self.name:25} | "
                f"mean: {s['mean']:7.2f}ms | "
                f"p95: {s['p95']:7.2f}ms | "
                f"max: {s['max']:7.2f}ms"
            )


class PipelineTimer:
    """Tracks timing for the entire 5-frame pipeline."""
    
    def __init__(self):
        self.frame_capture_time = None
        self.yolo_start_time = None
        self.yolo_end_time = None
        self.buffer_full_time = None
        self.posemamba_start_time = None
        self.posemamba_end_time = None
        self.viz_start_time = None
        self.viz_end_time = None
    
    def mark_frame_capture(self) -> None:
        """Mark when frame was captured from camera."""
        self.frame_capture_time = time.time()
    
    def mark_yolo_start(self) -> None:
        """Mark start of YOLO 2D detection."""
        self.yolo_start_time = time.time()
    
    def mark_yolo_end(self) -> None:
        """Mark end of YOLO 2D detection."""
        self.yolo_end_time = time.time()
    
    def mark_buffer_full(self) -> None:
        """Mark when 5-frame buffer becomes full."""
        self.buffer_full_time = time.time()
    
    def mark_posemamba_start(self) -> None:
        """Mark start of PoseMamba 3D prediction."""
        self.posemamba_start_time = time.time()
    
    def mark_posemamba_end(self) -> None:
        """Mark end of PoseMamba 3D prediction."""
        self.posemamba_end_time = time.time()
    
    def mark_viz_start(self) -> None:
        """Mark start of visualization update."""
        self.viz_start_time = time.time()
    
    def mark_viz_end(self) -> None:
        """Mark end of visualization update."""
        self.viz_end_time = time.time()
    
    def get_timings(self) -> dict:
        """
        Extract timing measurements.
        
        Returns:
            Dictionary with timing measurements in milliseconds:
            - yolo_time: YOLO inference time
            - buffer_wait_time: Time from YOLO to buffer full
            - posemamba_time: PoseMamba inference time
            - viz_time: Visualization update time
            - end_to_end_time: Camera capture to viz end
        """
        timings = {}
        
        if self.yolo_start_time and self.yolo_end_time:
            timings['yolo_time'] = (self.yolo_end_time - self.yolo_start_time) * 1000
        
        if self.yolo_end_time and self.buffer_full_time:
            timings['buffer_wait_time'] = (self.buffer_full_time - self.yolo_end_time) * 1000
        
        if self.posemamba_start_time and self.posemamba_end_time:
            timings['posemamba_time'] = (self.posemamba_end_time - self.posemamba_start_time) * 1000
        
        if self.viz_start_time and self.viz_end_time:
            timings['viz_time'] = (self.viz_end_time - self.viz_start_time) * 1000
        
        if self.frame_capture_time and self.viz_end_time:
            timings['end_to_end_time'] = (self.viz_end_time - self.frame_capture_time) * 1000
        
        return timings


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_screen_coordinates(points, width, height):
    """Normalize 2D keypoints to [-1, 1] range."""
    assert points.shape[-1] == 2
    return points / width * 2 - [1, height / width]


def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print('GPU not available, using CPU')
    return device


def load_posemamba_model(config_path, checkpoint_path, device):
    """Load PoseMamba-S model and checkpoint."""
    config = get_config(config_path)
    model_backbone = load_backbone(config)

    if device.startswith('cuda'):
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.to(device)

    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
    state_dict = checkpoint['model_pos']
    model_is_dp = isinstance(model_backbone, nn.DataParallel)
    ckpt_has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

    if ckpt_has_module_prefix and not model_is_dp:
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    elif (not ckpt_has_module_prefix) and model_is_dp:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    model_backbone.load_state_dict(state_dict, strict=True)
    model_backbone.eval()
    return model_backbone, config


def setup_visualization():
    """Setup 3D matplotlib figure."""
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    coord_range = 1000
    ax.set_xlim3d([-coord_range, coord_range])
    ax.set_ylim3d([-coord_range, coord_range])
    ax.set_zlim3d([-coord_range, coord_range])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    return fig, ax


def update_3d_plot(ax, pose_3d, frame_num):
    """Update 3D plot with pose."""
    ax.clear()
    
    coord_range = 1000
    ax.set_xlim3d([-coord_range, coord_range])
    ax.set_ylim3d([-coord_range, coord_range])
    ax.set_zlim3d([-coord_range, coord_range])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Plot joints
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='blue', s=50)
    
    # Plot connections
    connections = [
        (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
    ]
    for c1, c2 in connections:
        if c1 < len(pose_3d) and c2 < len(pose_3d):
            ax.plot(
                [pose_3d[c1, 0], pose_3d[c2, 0]],
                [pose_3d[c1, 1], pose_3d[c2, 1]],
                [pose_3d[c1, 2], pose_3d[c2, 2]],
                'b-', linewidth=1.5
            )
    
    ax.set_title(f'Prediction #{frame_num}')
    plt.pause(0.001)


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Measure 5-frame pipeline performance')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO input size')
    parser.add_argument('--conf', type=float, default=0.5, help='YOLO confidence threshold')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--show-results-every', type=int, default=30, help='Print stats every N predictions')
    parser.add_argument('--posemamba-config', type=str, 
                        default=os.path.join(PROJECT_ROOT, 'configs', 'pose3d', 'testing', 'notestaug', 'PoseMamba_train_3dhp_S_5.yaml'))
    parser.add_argument('--posemamba-checkpoint', type=str,
                        default=os.path.join(ZDEMO_DIR, 'weights', 'PoseMamba', 'ModelS', 'best_epoch_5.bin'))
    parser.add_argument('--yolo-model', type=str,
                        default=os.path.join(ZDEMO_DIR, 'weights', 'yolo', 'best.pt'))
    args = parser.parse_args()

    print("=" * 80)
    print("5-FRAME PIPELINE PERFORMANCE MEASUREMENT")
    print("=" * 80)
    print(f"Camera: {args.camera}")
    print(f"YOLO Model: {args.yolo_model}")
    print(f"PoseMamba Config: {args.posemamba_config}")
    print(f"PoseMamba Checkpoint: {args.posemamba_checkpoint}")
    print(f"Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print("=" * 80)

    # Check files exist
    for path in [args.yolo_model, args.posemamba_config, args.posemamba_checkpoint]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            return

    # Setup GPU
    device = check_gpu()

    # Load models
    print("Loading YOLO model...")
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(device)

    print("Loading PoseMamba model...")
    posemamba_model, posemamba_config = load_posemamba_model(
        args.posemamba_config,
        args.posemamba_checkpoint,
        device
    )
    
    no_conf = bool(getattr(posemamba_config, 'no_conf', False))
    use_flip = False  # Optional: set to True for better accuracy

    # Open camera (match robust behavior from demo script)
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print('V4L2 backend failed, trying default backend...')
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    # UVC cameras often need MJPG to stream reliably in WSL/usbipd setups.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {camera_width}x{camera_height}")

    # Setup visualization
    fig, ax = None, None
    if not args.no_viz:
        fig, ax = setup_visualization()

    # Timing statistics
    timing_yolo = TimingStats("YOLO 2D Detection", window_size=100)
    timing_buffer = TimingStats("Buffer Wait Time", window_size=100)
    timing_posemamba = TimingStats("PoseMamba 3D Inference", window_size=100)
    timing_viz = TimingStats("Visualization Update", window_size=100)
    timing_e2e = TimingStats("End-to-End (capture→display)", window_size=100)

    # Pipeline state
    pose_buffer = deque(maxlen=5)  # Always keep last 5 frames
    n_frames = 5
    target_frame_idx = 2  # Center frame (0-4, so frame 2 is the middle)
    
    frame_count = 0
    prediction_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 10

    print("\nStarting measurements. Press Ctrl+C to stop.\n")

    try:
        while True:
            # ================================================================
            # Capture frame
            # ================================================================
            timer = PipelineTimer()
            timer.mark_frame_capture()
            
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"Error: Failed to read frame {max_consecutive_failures} times in a row")
                    break
                continue
            consecutive_failures = 0
            
            frame_count += 1

            # ================================================================
            # YOLO 2D Detection
            # ================================================================
            timer.mark_yolo_start()
            results = yolo_model.predict(
                frame,
                verbose=False,
                imgsz=args.img_size,
                conf=args.conf,
                device=device,
            )
            timer.mark_yolo_end()

            # Extract 2D poses
            if (results and len(results) > 0 
                and hasattr(results[0], 'keypoints')
                and results[0].keypoints is not None
                and len(results[0].keypoints.xy) > 0):
                
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                confidences = (results[0].keypoints.conf[0].cpu().numpy() 
                              if results[0].keypoints.conf is not None 
                              else np.ones(17))

                pose_2d = np.zeros((17, 3), dtype=np.float32)
                n_kpts = min(17, keypoints.shape[0])
                pose_2d[:n_kpts, :2] = keypoints[:n_kpts]
                pose_2d[:n_kpts, 2] = confidences[:n_kpts]

                # Add to buffer
                pose_buffer.append(pose_2d)
            else:
                # No pose detected, skip this frame
                continue

            # ================================================================
            # Check if buffer is full
            # ================================================================
            timer.mark_buffer_full()  # Mark when buffer is at n_frames
            
            if len(pose_buffer) == n_frames:
                # ================================================================
                # PoseMamba 3D Prediction (on center frame)
                # ================================================================
                prediction_count += 1
                
                poses_2d = np.stack(list(pose_buffer), axis=0)
                poses_2d_normalized = poses_2d.copy()
                poses_2d_normalized[:, :, :2] = normalize_screen_coordinates(
                    poses_2d[:, :, :2],
                    camera_width,
                    camera_height,
                )

                input_2d = torch.from_numpy(poses_2d_normalized).unsqueeze(0).float()
                if no_conf:
                    input_2d = input_2d[:, :, :, :2]
                if device.startswith('cuda'):
                    input_2d = input_2d.cuda()

                timer.mark_posemamba_start()
                with torch.no_grad():
                    if use_flip:
                        input_2d_flip = flip_data(input_2d)
                        pred_3d_main = posemamba_model(input_2d)
                        pred_3d_flip = posemamba_model(input_2d_flip)
                        pred_3d = (pred_3d_main + flip_data(pred_3d_flip)) / 2.0
                    else:
                        pred_3d = posemamba_model(input_2d)
                timer.mark_posemamba_end()

                # Get center frame prediction
                pred_3d_center = pred_3d[0, target_frame_idx].detach().cpu().numpy()

                # ================================================================
                # Visualization
                # ================================================================
                if not args.no_viz:
                    timer.mark_viz_start()
                    update_3d_plot(ax, pred_3d_center, prediction_count)
                    timer.mark_viz_end()

                # ================================================================
                # Record timings
                # ================================================================
                timings = timer.get_timings()
                
                if 'yolo_time' in timings:
                    timing_yolo.add(timings['yolo_time'])
                if 'buffer_wait_time' in timings:
                    timing_buffer.add(timings['buffer_wait_time'])
                if 'posemamba_time' in timings:
                    timing_posemamba.add(timings['posemamba_time'])
                if 'viz_time' in timings and not args.no_viz:
                    timing_viz.add(timings['viz_time'])
                if 'end_to_end_time' in timings:
                    timing_e2e.add(timings['end_to_end_time'])

                # ================================================================
                # Print results periodically
                # ================================================================
                if prediction_count % args.show_results_every == 0:
                    print(f"\n[Prediction {prediction_count}]")
                    timing_yolo.print()
                    timing_buffer.print()
                    timing_posemamba.print()
                    if not args.no_viz:
                        timing_viz.print()
                    timing_e2e.print()
                    print()

            if not args.no_viz and fig is not None:
                if not plt.fignum_exists(fig.number):
                    print("\nVisualization window closed. Exiting...")
                    break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        cap.release()
        if not args.no_viz:
            plt.close('all')

        # ====================================================================
        # Final Summary
        # ====================================================================
        print("\n" + "=" * 80)
        print("FINAL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Total frames processed: {frame_count}")
        print(f"Total predictions: {prediction_count}")
        print()
        timing_yolo.print()
        timing_buffer.print()
        timing_posemamba.print()
        if not args.no_viz:
            timing_viz.print()
        timing_e2e.print()
        
        # Calculate throughput
        if timing_e2e.times:
            avg_e2e = timing_e2e.stats()['mean']
            throughput = 1000.0 / avg_e2e if avg_e2e > 0 else 0
            print(f"\n{'Prediction Throughput':25} | {throughput:7.2f} predictions/sec")
        
        # Latency budget analysis
        print("\n" + "-" * 80)
        print("LATENCY BREAKDOWN (end-to-end average):")
        if timing_yolo.times and timing_buffer.times and timing_posemamba.times:
            yolo_avg = timing_yolo.stats()['mean']
            buffer_avg = timing_buffer.stats()['mean']
            pose_avg = timing_posemamba.stats()['mean']
            viz_avg = timing_viz.stats()['mean'] if timing_viz.times else 0
            
            yolo_pct = (yolo_avg / timing_e2e.stats()['mean'] * 100) if timing_e2e.times else 0
            buffer_pct = (buffer_avg / timing_e2e.stats()['mean'] * 100) if timing_e2e.times else 0
            pose_pct = (pose_avg / timing_e2e.stats()['mean'] * 100) if timing_e2e.times else 0
            viz_pct = (viz_avg / timing_e2e.stats()['mean'] * 100) if timing_e2e.times else 0
            
            print(f"  YOLO:           {yolo_avg:7.2f}ms ({yolo_pct:5.1f}%)")
            print(f"  Buffer wait:    {buffer_avg:7.2f}ms ({buffer_pct:5.1f}%)")
            print(f"  PoseMamba:      {pose_avg:7.2f}ms ({pose_pct:5.1f}%)")
            if viz_avg > 0:
                print(f"  Visualization:  {viz_avg:7.2f}ms ({viz_pct:5.1f}%)")
            print(f"  {'─' * 40}")
            print(f"  Total:          {timing_e2e.stats()['mean']:7.2f}ms (100.0%)")
        
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
