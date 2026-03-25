"""
Real-time 3D pose estimation using YOLO + PoseMamba-S.
Collects 9 frames of 2D poses and predicts 3D poses with live 3D visualization.

Usage:
python 3DestimationPoseMamba9.py
python 3DestimationPoseMamba9.py --camera 0 --conf 0.5
python 3DestimationPoseMamba9.py --show-fps
python 3DestimationPoseMamba9.py --no-viz
"""

import argparse
import importlib.util
import os
import sys
import time

# Quick check for --disable-triton flag before importing lib modules
_quick_parser = argparse.ArgumentParser(add_help=False)
_quick_parser.add_argument('--disable-triton', action='store_true')
_quick_args, _ = _quick_parser.parse_known_args()

if _quick_args.disable_triton:
    print('Disabling Triton imports...')
    # Signal model code to use non-Triton fallbacks.
    os.environ['DISABLE_TRITON'] = '1'

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    from ultralytics import YOLO
except ImportError:
    from ultralytics.models.yolo.model import YOLO

ZDEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ZDEMO_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from lib.utils.utils_data import flip_data


JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand',
    'RShoulder', 'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle',
    'RHip', 'RKnee', 'RAnkle', 'Sacrum', 'Spine', 'Neck'
]

CONNECTIONS_3D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
]

DEFAULT_YOLO_MODEL_PATH = os.path.join(ZDEMO_DIR, 'weights', 'yolo', 'best.pt')
DEFAULT_POSEMAMBA_CONFIG = os.path.join(PROJECT_ROOT, 'configs', 'pose3d', 'PoseMamba_train_3dhp_S_9.yaml')
DEFAULT_POSEMAMBA_CHECKPOINT = os.path.join(ZDEMO_DIR, 'weights', 'PoseMamba', 'ModelS', 'best_epoch_9.bin')


def normalize_screen_coordinates(points, width, height):
    """Normalize 2D keypoints to [-1, 1] range."""
    assert points.shape[-1] == 2
    return points / width * 2 - [1, height / width]


def apply_upright_correction(poses_3d):
    """Rotate poses so the visualization is upright and easier to read."""
    rotation_x_90 = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ], dtype=np.float32)
    rotation_z_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    theta = np.deg2rad(45)
    rotation_z_45 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    return poses_3d @ rotation_x_90.T @ rotation_z_90.T @ rotation_z_45.T


def make_root_relative_3d(poses_3d, root_joint_idx=14):
    """Subtract root joint position from all joints."""
    root_pos = poses_3d[root_joint_idx]
    root_relative_poses = poses_3d - root_pos[np.newaxis, :]
    root_relative_poses[root_joint_idx] = [0.0, 0.0, 0.0]
    return root_relative_poses


def scale_pose_to_max(pose_3d, max_value=900):
    """Scale pose so that the maximum absolute coordinate is max_value."""
    max_coord = np.max(np.abs(pose_3d))
    if max_coord < 1e-6:
        return pose_3d
    scale_factor = max_value / max_coord
    return pose_3d * scale_factor


def check_gpu():
    """Check if GPU is available."""
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        if os.environ.get('DISABLE_TRITON', '0') == '1':
            print('Triton disabled: using GPU without Triton kernels')
    else:
        device = 'cpu'
        print('GPU not available, using CPU')
    return device


def validate_disable_triton_runtime():
    """Validate runtime requirements for non-Triton execution path."""
    has_any_scan_cuda = any(
        importlib.util.find_spec(mod_name) is not None
        for mod_name in ('selective_scan_cuda', 'selective_scan_cuda_core', 'selective_scan_cuda_oflex')
    )
    if has_any_scan_cuda:
        return True

    print('Error: --disable-triton was requested, but no selective-scan CUDA extension is available.')
    print('This PoseMamba build still requires one of: selective_scan_cuda / selective_scan_cuda_core / selective_scan_cuda_oflex.')
    print('On this Windows environment, the repo only contains Linux .so kernels, which cannot be loaded.')
    print('Options:')
    print('  1) Run without --disable-triton (use Triton path).')
    print('  2) Use an environment with compatible selective-scan CUDA extension binaries.')
    return False


def draw_2d_debug(frame, pose_2d, conf_threshold=0.2):
    """Overlay 2D keypoints with index and confidence for debugging."""
    debug_frame = frame.copy()
    h, w = debug_frame.shape[:2]

    for idx in range(min(17, pose_2d.shape[0])):
        x, y, conf = pose_2d[idx]
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h:
            continue

        good = conf >= conf_threshold
        color = (0, 220, 0) if good else (0, 80, 255)
        cv2.circle(debug_frame, (x_i, y_i), 4, color, -1)
        cv2.putText(
            debug_frame,
            f"{idx}:{conf:.2f}",
            (x_i + 5, y_i - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        debug_frame,
        "2D Debug (green>=thr, orange<thr)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return debug_frame


def setup_3d_plot(coord_range=1000):
    """Setup matplotlib 3D plot with reusable plot objects."""
    plt.ion()
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d([-coord_range, coord_range])
    ax.set_ylim3d([-coord_range, coord_range])
    ax.set_zlim3d([-coord_range, coord_range])
    ax.set_xlabel('X (mm, right)', fontsize=12)
    ax.set_ylabel('Y (mm, forward)', fontsize=12)
    ax.set_zlabel('Z (mm, up)', fontsize=12)

    skeleton_lines = []
    for _ in CONNECTIONS_3D:
        line, = ax.plot([], [], [], 'b-', linewidth=3, alpha=0.8)
        skeleton_lines.append(line)

    joint_scatter = ax.scatter([], [], [], c='blue', s=80, alpha=0.9, edgecolors='darkblue', linewidth=2)
    root_scatter = ax.scatter([0], [0], [0], c='green', s=200, marker='*', alpha=1.0, edgecolors='darkgreen', linewidth=3)

    ax.plot([0, 100], [0, 0], [0, 0], 'r-', linewidth=2, alpha=0.7)
    ax.plot([0, 0], [0, 100], [0, 0], 'g-', linewidth=2, alpha=0.7)
    ax.plot([0, 0], [0, 0], [0, 100], 'b-', linewidth=2, alpha=0.7)
    ax.text(100, 0, 0, 'X(R)', fontsize=10, color='red')
    ax.text(0, 100, 0, 'Y(F)', fontsize=10, color='green')
    ax.text(0, 0, 100, 'Z(U)', fontsize=10, color='blue')

    joint_texts = []
    for _ in range(17):
        text = ax.text(0, 0, 0, '', fontsize=11, color='blue', weight='bold')
        joint_texts.append(text)

    ax.view_init(elev=15, azim=45)

    plot_objects = {
        'skeleton_lines': skeleton_lines,
        'joint_scatter': joint_scatter,
        'root_scatter': root_scatter,
        'joint_texts': joint_texts,
    }
    return fig, ax, plot_objects


def update_3d_plot_fast(fig, ax, plot_objects, pose_3d, frame_num):
    """Fast update of 3D plot by only changing data, not recreating objects."""
    ax.set_title(f'Real-time 3D Pose Estimation - Frame {frame_num}', fontsize=14, pad=20)

    for idx, connection in enumerate(CONNECTIONS_3D):
        joint1, joint2 = connection
        if joint1 < len(pose_3d) and joint2 < len(pose_3d):
            x1, y1, z1 = pose_3d[joint1]
            x2, y2, z2 = pose_3d[joint2]
            plot_objects['skeleton_lines'][idx].set_data([x1, x2], [y1, y2])
            plot_objects['skeleton_lines'][idx].set_3d_properties([z1, z2])
        else:
            plot_objects['skeleton_lines'][idx].set_data([], [])
            plot_objects['skeleton_lines'][idx].set_3d_properties([])

    non_root_joints = [idx for idx in range(len(pose_3d)) if idx != 14]
    if non_root_joints:
        joint_positions = pose_3d[non_root_joints]
        plot_objects['joint_scatter']._offsets3d = (
            joint_positions[:, 0],
            joint_positions[:, 1],
            joint_positions[:, 2],
        )

    for joint_idx, (x, y, z) in enumerate(pose_3d):
        if joint_idx != 14:
            plot_objects['joint_texts'][joint_idx].set_position((x + 30, y + 30))
            plot_objects['joint_texts'][joint_idx].set_3d_properties(z + 30)
            plot_objects['joint_texts'][joint_idx].set_text(str(joint_idx))
        else:
            plot_objects['joint_texts'][joint_idx].set_text('')

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def print_3d_keypoints(prediction_num, keypoints_3d, print_output=True):
    """Print 3D keypoint coordinates in the same format as the existing demo."""
    keypoints_3d_corrected = apply_upright_correction(keypoints_3d)
    keypoints_3d_root_rel = make_root_relative_3d(keypoints_3d_corrected, root_joint_idx=14)
    keypoints_3d_scaled = scale_pose_to_max(keypoints_3d_root_rel, max_value=900)

    if print_output:
        print(f"\n{'=' * 80}")
        print(f"3D Pose Prediction #{prediction_num}")
        print(f"{'=' * 80}")
        print(f"{'Joint':<20} {'X (mm, right)':<20} {'Y (mm, forward)':<20} {'Z (mm, up)':<20}")
        print(f"{'-' * 80}")

        for idx in range(17):
            if idx < len(keypoints_3d_scaled):
                x, y, z = keypoints_3d_scaled[idx]
                joint_name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f'Joint_{idx}'
                if idx == 14:
                    print(f"{joint_name:<20} {x:<20.4f} {y:<20.4f} {z:<20.4f}  <- ROOT")
                else:
                    print(f"{joint_name:<20} {x:<20.4f} {y:<20.4f} {z:<20.4f}")
            else:
                joint_name = JOINT_NAMES[idx] if idx < len(JOINT_NAMES) else f'Joint_{idx}'
                print(f"{joint_name:<20} {0.0:<20.4f} {0.0:<20.4f} {0.0:<20.4f}")

        print(f"{'=' * 80}\n")

    return keypoints_3d_scaled


def load_posemamba_model(config_path, checkpoint_path):
    """Load PoseMamba-S model and checkpoint."""
    config = get_config(config_path)
    model_backbone = load_backbone(config)

    # Use DataParallel whenever CUDA is available.
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print(f'Loading PoseMamba-S checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
    state_dict = checkpoint['model_pos']
    model_is_dp = isinstance(model_backbone, nn.DataParallel)
    ckpt_has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())

    # Handle DataParallel prefix mismatch between checkpoint and current model.
    if ckpt_has_module_prefix and not model_is_dp:
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    elif (not ckpt_has_module_prefix) and model_is_dp:
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    model_backbone.load_state_dict(state_dict, strict=True)
    model_backbone.eval()
    return model_backbone, config


def main():
    parser = argparse.ArgumentParser(description='Real-time 3D Pose Estimation with YOLO + PoseMamba-S')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detections (default: 0.5)')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for YOLO inference (default: 640)')
    parser.add_argument('--show-fps', action='store_true', help='Show FPS counter in terminal')
    parser.add_argument('--no-viz', action='store_true', help='Disable 3D visualization (faster performance)')
    parser.add_argument('--no-print', action='store_true', help='Disable printing 3D coordinates (faster performance)')
    parser.add_argument('--flip-tta', action='store_true', help='Enable flip test-time augmentation (can improve accuracy but may add instability)')
    parser.add_argument('--target-frame', choices=['latest', 'middle'], default='middle', help='Which frame in the temporal window to visualize/publish')
    parser.add_argument('--debug-2d', action='store_true', help='Show 2D keypoints with index/confidence overlay')
    parser.add_argument('--debug-stats-every', type=int, default=30, help='Print debug stats every N frames (default: 30)')
    parser.add_argument('--debug-conf-thr', type=float, default=0.2, help='Confidence threshold used in 2D debug overlay')
    parser.add_argument('--fill-missing', action='store_true', help='When YOLO fails, repeat last pose into the buffer to keep it full')
    parser.add_argument('--pose-skip', type=int, default=1, help='Subsample factor for 2D poses (1=use every frame, 2=every 2nd frame, etc.)')
    parser.add_argument('--posemamba-config', type=str, default=DEFAULT_POSEMAMBA_CONFIG, help='Path to PoseMamba-S config file')
    parser.add_argument('--posemamba-checkpoint', type=str, default=DEFAULT_POSEMAMBA_CHECKPOINT, help='Path to PoseMamba-S checkpoint file')
    parser.add_argument('--coord-range', type=int, default=1000, help='3D visualization coordinate range (default: 1000mm)')
    parser.add_argument('--disable-triton', action='store_true', help='Disable Triton CUDA kernel compilation (useful if triton is not installed)')
    args = parser.parse_args()

    print('=' * 60)
    print('Real-time 3D Pose Estimation with YOLO + PoseMamba-S')
    print('=' * 60)
    print(f'YOLO Model: {DEFAULT_YOLO_MODEL_PATH}')
    print(f'PoseMamba Config: {args.posemamba_config}')
    print(f'PoseMamba Checkpoint: {args.posemamba_checkpoint}')
    print(f'Camera: {args.camera}')
    print(f'Confidence threshold: {args.conf}')
    print(f'Image size: {args.img_size}')
    print(f'Flip TTA: {"Enabled" if args.flip_tta else "Disabled"}')
    print(f'Target frame: {args.target_frame}')
    print(f"2D Debug Overlay: {'Enabled' if args.debug_2d else 'Disabled'}")
    print(f"3D Visualization: {'Disabled' if args.no_viz else 'Enabled'}")
    print(f"Print 3D Coordinates: {'Disabled' if args.no_print else 'Enabled'}")
    print('=' * 60)
    if not args.no_viz:
        print('Close 3D visualization window or press Ctrl+C to quit')
    else:
        print('Press Ctrl+C to quit')
    print('=' * 60)

    if not os.path.exists(DEFAULT_YOLO_MODEL_PATH):
        print(f'Error: YOLO model not found at {DEFAULT_YOLO_MODEL_PATH}')
        return
    if not os.path.exists(args.posemamba_config):
        print(f'Error: PoseMamba config not found at {args.posemamba_config}')
        return
    if not os.path.exists(args.posemamba_checkpoint):
        print(f'Error: PoseMamba checkpoint not found at {args.posemamba_checkpoint}')
        return

    if args.disable_triton and not validate_disable_triton_runtime():
        return

    device = check_gpu()

    print('Loading YOLO model...')
    try:
        yolo_model = YOLO(DEFAULT_YOLO_MODEL_PATH)
        yolo_model.to(device)
        print('YOLO model loaded successfully')
    except Exception as exc:
        print(f'Error loading YOLO model: {exc}')
        return

    print('Loading PoseMamba-S model...')
    try:
        posemamba_model, posemamba_config = load_posemamba_model(
            args.posemamba_config,
            args.posemamba_checkpoint,
        )
        print('PoseMamba-S model loaded successfully')
    except Exception as exc:
        print(f'Error loading PoseMamba-S model: {exc}')
        import traceback
        traceback.print_exc()
        return

    n_frames = 9
    use_flip = bool(args.flip_tta)
    no_conf = bool(getattr(posemamba_config, 'no_conf', False))

    print(f'Using temporal window of {n_frames} frames')

    print(f'Opening camera {args.camera}...')
    # Use V4L2 backend on Linux; UVC cameras require MJPG format to start streaming
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print('V4L2 backend failed, trying default backend...')
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Error: Could not open camera {args.camera}')
        return

    # MJPG is required for UVC cameras (e.g. via usbipd) to actually stream frames
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Camera opened successfully ({camera_width}x{camera_height})')

    fig, ax, plot_objects = None, None, None
    viz_initialized = False

    print('Starting real-time detection...')

    pose_buffer = []
    pose_subsample = max(1, args.pose_skip)
    debug_conf_window = []

    yolo_inference_times = []
    yolo_window_size = 30
    current_yolo_fps = 0

    posemamba_inference_times = []
    posemamba_window_size = 30
    current_posemamba_fps = 0

    viz_update_times = []
    viz_window_size = 30
    current_viz_fps = 0

    end_to_end_times = []
    end_to_end_window_size = 30
    current_end_to_end_fps = 0
    last_prediction_time = None

    frame_count = 0
    prediction_count = 0
    buffer_full_once = False

    consecutive_failures = 0
    max_consecutive_failures = 10

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f'\nError: Camera failed {max_consecutive_failures} times in a row, giving up')
                    break
                continue
            consecutive_failures = 0

            frame_count += 1

            yolo_start = time.time()
            results = yolo_model.predict(
                frame,
                verbose=False,
                imgsz=args.img_size,
                conf=args.conf,
                device=device,
            )
            yolo_end = time.time()
            yolo_time = yolo_end - yolo_start

            yolo_inference_times.append(yolo_time)
            if len(yolo_inference_times) > yolo_window_size:
                yolo_inference_times.pop(0)
            avg_yolo_time = np.mean(yolo_inference_times)
            current_yolo_fps = 1.0 / avg_yolo_time if avg_yolo_time > 0 else 0

            if (
                results and len(results) > 0
                and hasattr(results[0], 'keypoints')
                and results[0].keypoints is not None
                and len(results[0].keypoints.xy) > 0
            ):
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                confidences = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

                pose_2d = np.zeros((17, 3), dtype=np.float32)
                n_kpts = min(17, keypoints.shape[0])
                pose_2d[:n_kpts, :2] = keypoints[:n_kpts]
                pose_2d[:n_kpts, 2] = confidences[:n_kpts]

                debug_conf_window.append(float(np.mean(pose_2d[:, 2])))
                if len(debug_conf_window) > 120:
                    debug_conf_window.pop(0)

                if args.debug_2d:
                    debug_frame = draw_2d_debug(frame, pose_2d, conf_threshold=args.debug_conf_thr)
                    cv2.imshow('PoseMamba 2D Debug', debug_frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                        print('\n2D debug window closed by user. Exiting...')
                        break

                if args.debug_stats_every > 0 and (frame_count % args.debug_stats_every == 0):
                    visible = int(np.sum(pose_2d[:, 2] > args.debug_conf_thr))
                    avg_conf = float(np.mean(pose_2d[:, 2]))
                    avg_conf_hist = float(np.mean(debug_conf_window)) if debug_conf_window else 0.0
                    print(
                        f"\n[debug] frame={frame_count} visible(>{args.debug_conf_thr:.2f})={visible}/17 "
                        f"avg_conf={avg_conf:.3f} avg_conf_win={avg_conf_hist:.3f} buffer={len(pose_buffer)}/{n_frames}"
                    )

                if frame_count % pose_subsample == 0:
                    pose_buffer.append(pose_2d)
                    if len(pose_buffer) > n_frames:
                        pose_buffer.pop(0)

                if args.show_fps:
                    base_str = f'Buffer: {len(pose_buffer)}/{n_frames} | YOLO: {current_yolo_fps:.1f} FPS'
                    if len(posemamba_inference_times) > 0:
                        base_str += f' | PoseMamba: {current_posemamba_fps:.1f} FPS'
                    if not args.no_viz and len(viz_update_times) > 0:
                        base_str += f' | 3D Viz: {current_viz_fps:.1f} FPS'
                    if current_end_to_end_fps > 0:
                        base_str += f' | End-to-End: {current_end_to_end_fps:.1f} FPS'
                    print(base_str, end='\r')
            else:
                if args.fill_missing and len(pose_buffer) > 0:
                    pose_buffer.append(pose_buffer[-1])
                    if len(pose_buffer) > n_frames:
                        pose_buffer.pop(0)

                if args.show_fps:
                    base_str = f'No person detected | Frame: {frame_count} | YOLO: {current_yolo_fps:.1f} FPS'
                    if len(posemamba_inference_times) > 0:
                        base_str += f' | PoseMamba: {current_posemamba_fps:.1f} FPS'
                    if not args.no_viz and len(viz_update_times) > 0:
                        base_str += f' | 3D Viz: {current_viz_fps:.1f} FPS'
                    if current_end_to_end_fps > 0:
                        base_str += f' | End-to-End: {current_end_to_end_fps:.1f} FPS'
                    print(base_str, end='\r')

            if len(pose_buffer) == n_frames:
                if not buffer_full_once:
                    buffer_full_once = True
                    if not args.no_viz and not viz_initialized:
                        fig, ax, plot_objects = setup_3d_plot(args.coord_range)
                        viz_initialized = True
                        print('3D visualization window opened')

                prediction_count += 1

                poses_2d = np.stack(pose_buffer, axis=0)
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

                posemamba_start = time.time()
                with torch.no_grad():
                    if use_flip:
                        input_2d_flip = flip_data(input_2d)
                        pred_3d_main = posemamba_model(input_2d)
                        pred_3d_flip = posemamba_model(input_2d_flip)
                        pred_3d = (pred_3d_main + flip_data(pred_3d_flip)) / 2.0
                    else:
                        pred_3d = posemamba_model(input_2d)
                posemamba_end = time.time()
                posemamba_time = posemamba_end - posemamba_start

                posemamba_inference_times.append(posemamba_time)
                if len(posemamba_inference_times) > posemamba_window_size:
                    posemamba_inference_times.pop(0)
                avg_posemamba_time = np.mean(posemamba_inference_times)
                current_posemamba_fps = 1.0 / avg_posemamba_time if avg_posemamba_time > 0 else 0

                if args.target_frame == 'middle':
                    frame_idx = n_frames // 2
                else:
                    frame_idx = n_frames - 1
                pred_3d_middle = pred_3d[0, frame_idx].detach().cpu().numpy()
                pose_3d_viz = print_3d_keypoints(prediction_count, pred_3d_middle, print_output=not args.no_print)

                now = time.time()
                if last_prediction_time is not None:
                    delta = now - last_prediction_time
                    end_to_end_times.append(delta)
                    if len(end_to_end_times) > end_to_end_window_size:
                        end_to_end_times.pop(0)
                    avg_delta = np.mean(end_to_end_times)
                    current_end_to_end_fps = 1.0 / avg_delta if avg_delta > 0 else 0
                last_prediction_time = now

                if not args.no_viz and current_end_to_end_fps > 0:
                    viz_start = time.time()
                    update_3d_plot_fast(fig, ax, plot_objects, pose_3d_viz, prediction_count)
                    viz_end = time.time()
                    viz_time = viz_end - viz_start

                    viz_update_times.append(viz_time)
                    if len(viz_update_times) > viz_window_size:
                        viz_update_times.pop(0)
                    avg_viz_time = np.mean(viz_update_times)
                    current_viz_fps = 1.0 / avg_viz_time if avg_viz_time > 0 else 0

            if not args.no_viz and fig is not None:
                if not plt.fignum_exists(fig.number):
                    print('\n3D visualization window closed. Exiting...')
                    break

    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        cap.release()
        if args.debug_2d:
            cv2.destroyWindow('PoseMamba 2D Debug')
        if not args.no_viz:
            plt.close('all')

        print('\nCamera released and windows closed')
        print(f"\n{'=' * 60}")
        print('Performance Summary:')
        print(f"{'=' * 60}")
        print(f'Total frames processed: {frame_count}')
        print(f'Total 3D predictions: {prediction_count}')
        print(f'Average YOLO FPS: {current_yolo_fps:.1f}')
        if len(posemamba_inference_times) > 0:
            print(f'Average PoseMamba FPS: {current_posemamba_fps:.1f} (throughput after {n_frames} frames)')
        if not args.no_viz and len(viz_update_times) > 0:
            print(f'Average 3D Visualization FPS: {current_viz_fps:.1f}')
        if current_end_to_end_fps > 0:
            print(f'Average End-to-End FPS: {current_end_to_end_fps:.1f}')
        print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
