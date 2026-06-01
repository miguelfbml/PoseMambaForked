"""
Render UCO rehabilitation videos with the raw frame on the left and a PoseMamba
3D visualization on the right.

The script walks folders 0-5, subfolders 09-16, and cameras cam0-cam4 by default.
Each input clip is processed with YOLO pose estimation followed by PoseMamba 3D
prediction, and the output mp4 is saved under:

    <output-dir>/<folder>/<subfolder>/<camera>/<camera>.mp4

This matches the nested layout used by count_uco_yolo_sequences.py.

Example:
python compare_uco_posemamba_sequences.py \
    --yolo-model weights/YOLO/best.pt \
    --posemamba-config configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml \
    --posemamba-checkpoint zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin \
    --output-dir uco_posemamba_sequence_output
"""


import argparse
import gc
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
ZDEMO_DIR = os.path.join(PROJECT_ROOT, 'zdemo')
ZREHAB_DIR = os.path.join(PROJECT_ROOT, 'zReabilitation')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ZREHAB_DIR not in sys.path:
    sys.path.insert(0, ZREHAB_DIR)
if ZDEMO_DIR not in sys.path:
    sys.path.insert(0, ZDEMO_DIR)

from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.utils_data import flip_data  # noqa: E402
from zReabilitation.compare_gt_yolo_2d import estimate_yolo_poses  # noqa: E402
from demo.lib.utils import camera_to_world  # noqa: E402


JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand',
    'RShoulder', 'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle',
    'RHip', 'RKnee', 'RAnkle', 'Sacrum', 'Spine', 'Neck',
]

CONNECTIONS_3D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
]

DEFAULT_YOLO_MODEL_PATH = os.path.join(ZDEMO_DIR, 'weights', 'yolo', 'best.pt')
DEFAULT_POSEMAMBA_CONFIG = os.path.join(
    PROJECT_ROOT,
    'configs',
    'pose3d',
    'testing',
    'notestaug',
    'PoseMamba_train_3dhp_S_5.yaml',
)
DEFAULT_POSEMAMBA_CHECKPOINT = os.path.join(
    ZDEMO_DIR,
    'weights',
    'PoseMamba',
    'ModelS',
    'best_epoch_5.bin',
)
UCO_DATASET_PATH = '/nas-ctm01/datasets/public/UCO Physical Rehabilitation/dataset/clips_mp4'
DEFAULT_CAMERAS = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']
DEFAULT_FOLDERS = list(range(0, 6))
DEFAULT_SUBFOLDERS = list(range(9, 17))


def check_gpu_availability():
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        current_device = torch.cuda.current_device()
        return f'cuda:{current_device}', True
    return 'cpu', False


def normalize_screen_coordinates(points, width, height):
    assert points.shape[-1] == 2
    return points / width * 2 - [1, height / width]


def apply_upright_correction(poses_3d):
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
    root_pos = poses_3d[root_joint_idx]
    root_relative_poses = poses_3d - root_pos[np.newaxis, :]
    root_relative_poses[root_joint_idx] = [0.0, 0.0, 0.0]
    return root_relative_poses


def scale_pose_to_max(pose_3d, max_value=900):
    max_coord = np.max(np.abs(pose_3d))
    if max_coord < 1e-6:
        return pose_3d
    return pose_3d * (max_value / max_coord)


def build_window_indices(center_idx, total_frames, window_size):
    half_window = window_size // 2
    indices = []
    for offset in range(-half_window, half_window + 1):
        index = center_idx + offset
        index = max(0, min(total_frames - 1, index))
        indices.append(index)
    return indices


def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frames = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None, fps, 0, 0
        height, width = frame.shape[:2]
        frames.append(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps, width, height


def load_posemamba_model(config_path, checkpoint_path, device):
    config = get_config(config_path)
    model_backbone = load_backbone(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_pos']
    model_is_dp = isinstance(model_backbone, nn.DataParallel)
    ckpt_has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())

    if ckpt_has_module_prefix and not model_is_dp:
        state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
    elif (not ckpt_has_module_prefix) and model_is_dp:
        state_dict = {f'module.{key}': value for key, value in state_dict.items()}

    model_backbone.load_state_dict(state_dict, strict=True)
    model_backbone.to(device)
    model_backbone.eval()
    return model_backbone, config


def predict_posemamba_window(model, posemamba_config, poses_2d_window, frame_shape, device, use_flip=False):
    normalized_2d = poses_2d_window.copy().astype(np.float32)
    normalized_2d[:, :, :2] = normalize_screen_coordinates(
        normalized_2d[:, :, :2],
        frame_shape[1],
        frame_shape[0],
    )

    input_2d = torch.from_numpy(normalized_2d).unsqueeze(0).float()
    if bool(getattr(posemamba_config, 'no_conf', False)):
        input_2d = input_2d[:, :, :, :2]

    if device.startswith('cuda'):
        input_2d = input_2d.cuda()

    with torch.no_grad():
        if use_flip:
            input_2d_flip = flip_data(input_2d)
            pred_3d_main = model(input_2d)
            pred_3d_flip = model(input_2d_flip)
            pred_3d = (pred_3d_main + flip_data(pred_3d_flip)) / 2.0
        else:
            pred_3d = model(input_2d)

    middle_idx = len(poses_2d_window) // 2
    pred_3d_middle = pred_3d[0, middle_idx].detach().cpu().numpy()
    return pred_3d_middle


def prepare_pose_for_plot(pose_3d):
    if pose_3d is None:
        return None

    post_out = pose_3d.copy().astype(np.float32)
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype=np.float32)
    try:
        post_out = camera_to_world(post_out, R=rot, t=0)
    except Exception:
        pass

    post_out[:, 2] -= np.min(post_out[:, 2])
    max_value = np.max(post_out)
    if max_value < 1e-6:
        return post_out
    return post_out / max_value


def plot_3d_skeleton(ax, pose_3d, title, line_color, point_color, missing_text, show_dots=True):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('X (right)', fontsize=10)
    ax.set_ylabel('Y (forward)', fontsize=10)
    ax.set_zlabel('Z (up)', fontsize=10)
    ax.view_init(elev=15, azim=45)

    if pose_3d is None or np.allclose(pose_3d, 0.0):
        ax.text2D(0.10, 0.50, missing_text, transform=ax.transAxes, fontsize=12, color='red')
        return

    for joint1, joint2 in CONNECTIONS_3D:
        if joint1 < len(pose_3d) and joint2 < len(pose_3d):
            p1 = pose_3d[joint1]
            p2 = pose_3d[joint2]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=line_color,
                linewidth=2.5,
                alpha=0.85,
            )

    if show_dots:
        xs = pose_3d[:, 0]
        ys = pose_3d[:, 1]
        zs = pose_3d[:, 2]
        ax.scatter(xs, ys, zs, c=point_color, s=45, alpha=0.9, edgecolors='black', linewidth=0.4)
        if len(pose_3d) > 14:
            ax.scatter(
                [pose_3d[14, 0]],
                [pose_3d[14, 1]],
                [pose_3d[14, 2]],
                c='green',
                s=120,
                marker='*',
                alpha=1.0,
                edgecolors='darkgreen',
                linewidth=1,
            )

    for joint_idx, (x, y, z) in enumerate(pose_3d):
        joint_name = JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f'Joint_{joint_idx}'
        del joint_name
        ax.text(x + 18, y + 18, z + 18, f'{joint_idx}', fontsize=8, color='black')

    ax.grid(True, alpha=0.25)


def render_pose_panel(pose_3d, width, height, sequence_name, frame_idx):
    pose_plot = prepare_pose_for_plot(pose_3d)

    fig = plt.figure(figsize=(max(width, 1) / 100.0, max(height, 1) / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fcfcfc')

    if pose_plot is not None:
        pose_plot = apply_upright_correction(pose_plot)
        pose_plot = make_root_relative_3d(pose_plot)
        pose_plot = scale_pose_to_max(pose_plot)

        valid = pose_plot[np.isfinite(pose_plot).all(axis=1)]
        if valid.size > 0:
            min_value = np.min(valid, axis=0)
            max_value = np.max(valid, axis=0)
            padding = (max_value - min_value) * 0.15
            min_value -= padding
            max_value += padding
            span = np.maximum(max_value - min_value, 1.0)
            center = pose_plot[14] if pose_plot.shape[0] > 14 else (min_value + max_value) / 2.0
            ax.set_xlim3d([center[0] - span[0] / 2.0, center[0] + span[0] / 2.0])
            ax.set_ylim3d([center[1] - span[1] / 2.0, center[1] + span[1] / 2.0])
            ax.set_zlim3d([center[2] - span[2] / 2.0, center[2] + span[2] / 2.0])

    plot_3d_skeleton(
        ax,
        pose_plot,
        'PoseMamba Prediction',
        line_color='tomato',
        point_color='salmon',
        missing_text='No Prediction',
        show_dots=True,
    )

    fig.suptitle(f'{sequence_name} | Frame {frame_idx}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def compose_frame(frame_bgr, pose_3d, sequence_name, frame_idx):
    height, width = frame_bgr.shape[:2]
    right_panel = render_pose_panel(pose_3d, width, height, sequence_name, frame_idx)
    right_panel_bgr = cv2.cvtColor(right_panel, cv2.COLOR_RGB2BGR)
    if right_panel_bgr.shape[:2] != (height, width):
        right_panel_bgr = cv2.resize(right_panel_bgr, (width, height), interpolation=cv2.INTER_AREA)
    separator = np.full((height, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([frame_bgr, separator, right_panel_bgr], axis=1)


def make_output_video_path(output_dir, folder, subfolder, camera):
    return os.path.join(output_dir, str(folder), f'{subfolder:02d}', camera, f'{camera}.mp4')


def load_uco_video_path(folder, subfolder, camera):
    return os.path.join(UCO_DATASET_PATH, str(folder), f'{subfolder:02d}', f'{camera}.mp4')


def process_sequence_camera(yolo_model, posemamba_model, posemamba_config, folder, subfolder, camera, args, device):
    if not args.save_videos:
        return None

    sequence_name = f'{folder}/{subfolder:02d}'
    video_path = load_uco_video_path(folder, subfolder, camera)

    if not os.path.exists(video_path):
        print(f'⚠ Skipping {sequence_name} {camera}: missing video')
        return None

    frames, fps, width, height = load_video_frames(video_path)
    if not frames:
        print(f'⚠ Skipping {sequence_name} {camera}: cannot load frames')
        return None

    print('\n' + '=' * 80)
    print(f'Processing {sequence_name} | {camera}')
    print(f'Input:  {video_path}')
    print(f'Output: {make_output_video_path(args.output_dir, folder, subfolder, camera)}')
    print(f'Frames: {len(frames)} | FPS: {fps:.2f} | Size: {width}x{height}')
    print('=' * 80)

    yolo_poses_2d, _, performance_metrics = estimate_yolo_poses(
        yolo_model,
        frames,
        args.img_size,
        device,
        batch_size=args.batch_size,
    )

    if len(yolo_poses_2d) != len(frames):
        print(f'⚠ YOLO output length mismatch for {sequence_name} {camera}: {len(yolo_poses_2d)} vs {len(frames)}')
        return None

    output_video_path = make_output_video_path(args.output_dir, folder, subfolder, camera)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width * 2 + 8, height),
    )
    if not writer.isOpened():
        print(f'⚠ Skipping {sequence_name} {camera}: cannot create output video')
        return None

    total_frames = 0
    written_frames = 0

    try:
        for frame_idx, frame in enumerate(frames):
            window_indices = build_window_indices(frame_idx, len(frames), args.window_size)
            pose_window = yolo_poses_2d[window_indices]

            pred_pose_3d = predict_posemamba_window(
                posemamba_model,
                posemamba_config,
                pose_window,
                frame.shape,
                device,
                use_flip=args.flip_tta,
            )

            combined = compose_frame(frame, pred_pose_3d, sequence_name, frame_idx)
            writer.write(combined)
            total_frames += 1
            written_frames += 1

    finally:
        writer.release()

    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()

    summary = {
        'sequence': sequence_name,
        'folder': folder,
        'subfolder': f'{subfolder:02d}',
        'camera': camera,
        'video_path': video_path,
        'output_video_path': output_video_path,
        'total_frames': total_frames,
        'written_frames': written_frames,
        'yolo_mean_inference_time_ms': float(performance_metrics['mean_inference_time'] * 1000.0),
        'yolo_fps': float(performance_metrics['fps']),
    }

    print(
        f"✓ {sequence_name} {camera}: "
        f'frames={written_frames}, '
        f"YOLO FPS={performance_metrics['fps']:.2f}, "
        f"mean inference={performance_metrics['mean_inference_time'] * 1000.0:.2f} ms"
    )
    return summary


def process_all_uco_sequences(yolo_model, posemamba_model, posemamba_config, args, device):
    results = []
    for folder in args.folders:
        print(f'\n--- Subject {folder:02d} start ---')
        for subfolder in args.subfolders:
            print(f'  -> Subfolder {folder:02d}/{subfolder:02d} start')
            for camera in args.cameras:
                print(f'     -> Camera {camera} start')
                try:
                    result = process_sequence_camera(
                        yolo_model,
                        posemamba_model,
                        posemamba_config,
                        folder,
                        subfolder,
                        camera,
                        args,
                        device,
                    )
                except Exception as exc:
                    print(f'     ! Camera {camera} failed with error: {exc}')
                    continue

                if result is not None:
                    results.append(result)
                print(f'     <- Camera {camera} done')

            print(f'  <- Subfolder {folder:02d}/{subfolder:02d} done')

        print(f'--- Subject {folder:02d} done ---')

    return results


def main():
    parser = argparse.ArgumentParser(description='Render UCO PoseMamba video comparisons')
    parser.add_argument('--yolo-model', '--model-path', dest='yolo_model', type=str, default=DEFAULT_YOLO_MODEL_PATH, help='Path to the trained YOLOv11x-Pose model')
    parser.add_argument('--posemamba-config', type=str, default=DEFAULT_POSEMAMBA_CONFIG, help='Path to the PoseMamba config file')
    parser.add_argument('--posemamba-checkpoint', type=str, default=DEFAULT_POSEMAMBA_CHECKPOINT, help='Path to the PoseMamba checkpoint file')
    parser.add_argument('--output-dir', type=str, default='uco_posemamba_sequence_output', help='Directory for saved comparison videos')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO input image size')
    parser.add_argument('--batch-size', type=int, default=16, help='YOLO batch size over frames')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--window-size', type=int, default=5, help='PoseMamba temporal window size (must be odd)')
    parser.add_argument('--flip-tta', action='store_true', help='Enable flip test-time augmentation for PoseMamba')
    parser.add_argument('--folders', type=int, nargs='*', default=DEFAULT_FOLDERS, help='UCO subject folders to process')
    parser.add_argument('--subfolders', type=int, nargs='*', default=DEFAULT_SUBFOLDERS, help='UCO subfolders to process')
    parser.add_argument('--cameras', type=str, nargs='*', default=DEFAULT_CAMERAS, help='Cameras to process (default: cam0 cam1 cam2 cam3 cam4)')
    parser.add_argument('--disable-triton', action='store_true', help='Disable Triton imports for PoseMamba')
    parser.add_argument('--no-save-videos', dest='save_videos', action='store_false', help='Do not save comparison videos')
    parser.set_defaults(save_videos=True)
    args = parser.parse_args()

    if args.window_size < 3 or args.window_size % 2 == 0:
        parser.error('--window-size must be an odd integer greater than or equal to 3')

    if args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')

    if args.disable_triton:
        os.environ['DISABLE_TRITON'] = '1'

    print('=' * 80)
    print('UCO PoseMamba sequence renderer')
    print('=' * 80)
    print(f'Folder range: {args.folders}')
    print(f'Subfolder range: {args.subfolders}')
    print(f'Cameras: {args.cameras}')
    print(f'YOLO model: {args.yolo_model}')
    print(f'PoseMamba config: {args.posemamba_config}')
    print(f'PoseMamba checkpoint: {args.posemamba_checkpoint}')
    print(f'Output dir: {args.output_dir}')
    print(f'Window size: {args.window_size}')
    print(f'Flip TTA: {"Enabled" if args.flip_tta else "Disabled"}')
    print(f'Device: {args.device}')
    print('=' * 80)

    if not os.path.exists(args.yolo_model):
        print(f'❌ YOLO model not found: {args.yolo_model}')
        return
    if not os.path.exists(args.posemamba_config):
        print(f'❌ PoseMamba config not found: {args.posemamba_config}')
        return
    if not os.path.exists(args.posemamba_checkpoint):
        print(f'❌ PoseMamba checkpoint not found: {args.posemamba_checkpoint}')
        return

    if args.device == 'auto':
        device, gpu_available = check_gpu_availability()
    else:
        device = args.device
        gpu_available = device.startswith('cuda') and torch.cuda.is_available()

    print(f'🤖 Loading YOLO model from {args.yolo_model}...')
    try:
        yolo_model = YOLO(args.yolo_model)
        if gpu_available:
            print(f'📦 Moving YOLO model to {device}...')
            yolo_model.to(device)
        print('✓ YOLO model loaded successfully')
    except Exception as exc:
        print(f'❌ Error loading YOLO model: {exc}')
        return

    print(f'🤖 Loading PoseMamba model from {args.posemamba_config}...')
    try:
        posemamba_model, posemamba_config = load_posemamba_model(
            args.posemamba_config,
            args.posemamba_checkpoint,
            device,
        )
        print('✓ PoseMamba model loaded successfully')
    except Exception as exc:
        print(f'❌ Error loading PoseMamba model: {exc}')
        return

    try:
        process_all_uco_sequences(yolo_model, posemamba_model, posemamba_config, args, device)
    finally:
        if gpu_available:
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()