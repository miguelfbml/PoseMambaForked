"""
Render selected UCO rehabilitation frames with 3D ground truth overlaid on the
frame image and PoseMamba 3D prediction on the right.

This is the selected-frame counterpart to comparePose.py. It loads a UCO video
for a given folder/subfolder/camera, runs YOLO + PoseMamba on the full video,
then saves one comparison image per requested frame.

The left panel keeps the raw frame as the background and places a GT 3D inset
on top of it. The right panel shows the PoseMamba 3D prediction using the same
visualization path as the demo/comparePose pipeline.

Example:
python compare_uco_gt_pred_posemamba_selected_frames.py \
    --sequence "0/01" \
    --camera cam0 \
    --frames 0 30 60 90 \
    --output-dir uco_gt_pred_selected_frames
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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ZDEMO_DIR not in sys.path:
    sys.path.insert(0, ZDEMO_DIR)

from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.utils_data import flip_data  # noqa: E402
from zReabilitation.comparePose import (  # noqa: E402
    DEFAULT_COORD_RANGE,
    DEFAULT_POSEMAMBA_CHECKPOINT,
    DEFAULT_POSEMAMBA_CONFIG,
    DEFAULT_YOLO_MODEL_PATH,
    apply_upright_correction,
    build_window_indices,
    check_gpu_availability,
    load_video_frames,
    make_root_relative_3d,
    normalize_screen_coordinates,
    plot_3d_skeleton,
    predict_posemamba_window,
    prepare_pose_for_plot,
    scale_pose_to_max,
)
from zReabilitation.compare_gt_yolo_2d import estimate_yolo_poses  # noqa: E402


JOINT_NAMES = [
    'Head', 'SpineShoulder', 'LShoulder', 'LElbow', 'LHand',
    'RShoulder', 'RElbow', 'RHand', 'LHip', 'LKnee', 'LAnkle',
    'RHip', 'RKnee', 'RAnkle', 'Sacrum', 'Spine', 'Neck',
]

CONNECTIONS_3D = [
    (0, 16), (16, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    (1, 15), (15, 14), (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13),
]

DEFAULT_CAMERAS = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']
DEFAULT_GT_3D_FILE = ''
UCO_DATASET_PATH = '/nas-ctm01/datasets/public/UCO Physical Rehabilitation/dataset/clips_mp4'


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


def load_uco_video_path(folder, subfolder, camera):
    return os.path.join(UCO_DATASET_PATH, str(folder), f'{subfolder:02d}', f'{camera}.mp4')


def resolve_uco_gt_3d_path(folder, subfolder, camera, explicit_path=''):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    base_dir = os.path.join(UCO_DATASET_PATH, str(folder), f'{subfolder:02d}')
    candidates.extend([
        os.path.join(base_dir, f'{camera}_p3d.txt'),
        os.path.join(base_dir, 'p3d.txt'),
        os.path.join(base_dir, f'{camera}.p3d.txt'),
        os.path.join(base_dir, camera, 'p3d.txt'),
        os.path.join(base_dir, camera, f'{camera}_p3d.txt'),
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else None


def load_uco_gt_3d(gt_path, expected_joints=17):
    if not gt_path or not os.path.exists(gt_path):
        return None

    poses = []
    try:
        with open(gt_path, 'r', encoding='utf-8') as handle:
            for line_idx, line in enumerate(handle):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    values = [float(value) for value in line.split()]
                except ValueError:
                    print(f'⚠ Skipping GT line {line_idx}: contains non-numeric values')
                    continue

                if len(values) < 3 or len(values) % 3 != 0:
                    print(f'⚠ Skipping GT line {line_idx}: expected a multiple of 3 values, got {len(values)}')
                    continue

                pose = np.asarray(values, dtype=np.float32).reshape(-1, 3)
                if pose.shape[0] < expected_joints:
                    padded_pose = np.zeros((expected_joints, 3), dtype=np.float32)
                    padded_pose[:pose.shape[0]] = pose
                    pose = padded_pose
                elif pose.shape[0] > expected_joints:
                    pose = pose[:expected_joints]

                poses.append(pose)
    except Exception as exc:
        print(f'❌ Error loading GT 3D file {gt_path}: {exc}')
        return None

    if not poses:
        return None
    return np.asarray(poses, dtype=np.float32)


def render_pose_panel(
    pose_3d,
    width,
    height,
    sequence_name,
    frame_idx,
    panel_title,
    line_color,
    point_color,
    missing_text,
):
    pose_plot = prepare_pose_for_plot(pose_3d)

    fig = plt.figure(figsize=(max(width, 1) / 100.0, max(height, 1) / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fcfcfc')

    ax.set_xlim3d([-DEFAULT_COORD_RANGE, DEFAULT_COORD_RANGE])
    ax.set_ylim3d([-DEFAULT_COORD_RANGE, DEFAULT_COORD_RANGE])
    ax.set_zlim3d([-DEFAULT_COORD_RANGE, DEFAULT_COORD_RANGE])

    plot_3d_skeleton(
        ax,
        pose_plot,
        panel_title,
        line_color=line_color,
        point_color=point_color,
        missing_text=missing_text,
        show_dots=True,
    )

    fig.suptitle(f'{sequence_name} | Frame {frame_idx}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def compose_left_panel_with_gt_overlay(frame_bgr, gt_pose_3d, sequence_name, frame_idx):
    height, width = frame_bgr.shape[:2]
    output = frame_bgr.copy()

    inset_width = max(int(width * 0.42), 240)
    inset_height = max(int(height * 0.42), 180)
    inset_width = min(inset_width, width - 24)
    inset_height = min(inset_height, height - 24)

    inset = render_pose_panel(
        gt_pose_3d,
        inset_width,
        inset_height,
        sequence_name,
        frame_idx,
        panel_title='Ground Truth 3D',
        line_color='royalblue',
        point_color='deepskyblue',
        missing_text='No GT Data',
    )
    inset_bgr = cv2.cvtColor(inset, cv2.COLOR_RGB2BGR)
    inset_bgr = cv2.copyMakeBorder(inset_bgr, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    inset_h, inset_w = inset_bgr.shape[:2]
    margin = 12
    if inset_w + margin * 2 > width or inset_h + margin * 2 > height:
        scale = min((width - 2 * margin) / max(inset_w, 1), (height - 2 * margin) / max(inset_h, 1), 1.0)
        inset_bgr = cv2.resize(
            inset_bgr,
            (max(int(inset_w * scale), 1), max(int(inset_h * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
        inset_h, inset_w = inset_bgr.shape[:2]

    y0 = margin
    x0 = margin
    y1 = min(y0 + inset_h, height)
    x1 = min(x0 + inset_w, width)
    output[y0:y1, x0:x1] = inset_bgr[: y1 - y0, : x1 - x0]

    cv2.putText(
        output,
        'GT 3D overlay',
        (20, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def compose_side_by_side(frame_bgr, gt_pose_3d, pred_pose_3d, sequence_name, frame_idx):
    height, width = frame_bgr.shape[:2]
    left_panel = compose_left_panel_with_gt_overlay(frame_bgr, gt_pose_3d, sequence_name, frame_idx)
    right_panel = render_pose_panel(
        pred_pose_3d,
        width,
        height,
        sequence_name,
        frame_idx,
        panel_title='PoseMamba Prediction',
        line_color='tomato',
        point_color='salmon',
        missing_text='No Prediction',
    )
    right_panel_bgr = cv2.cvtColor(right_panel, cv2.COLOR_RGB2BGR)
    separator = np.full((height, 8, 3), 255, dtype=np.uint8)
    return np.concatenate([left_panel, separator, right_panel_bgr], axis=1)


def save_frame_comparison(frame_bgr, gt_pose_3d, pred_pose_3d, sequence_name, frame_idx, output_dir):
    output = compose_side_by_side(frame_bgr, gt_pose_3d, pred_pose_3d, sequence_name, frame_idx)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}_gt_overlay_posemamba3d.png')
    cv2.imwrite(output_path, output)
    return output_path


def process_selected_frames(sequence_name, frame_indices, args):
    if '/' not in sequence_name:
        print(f'❌ Expected UCO sequence in folder/subfolder format, got: {sequence_name}')
        return None

    folder, subfolder = sequence_name.split('/', 1)
    try:
        folder_idx = int(folder)
        subfolder_idx = int(subfolder)
    except ValueError:
        print(f'❌ Invalid UCO sequence format: {sequence_name}')
        return None

    video_path = load_uco_video_path(folder_idx, subfolder_idx, args.camera)
    if not os.path.exists(video_path):
        print(f'❌ Video file not found: {video_path}')
        return None

    gt_path = resolve_uco_gt_3d_path(folder_idx, subfolder_idx, args.camera, args.gt_3d_file)
    if not gt_path or not os.path.exists(gt_path):
        print(f'❌ Ground-truth 3D file not found. Tried: {gt_path if gt_path else "<no candidate>"}')
        return None

    frames, fps, width, height = load_video_frames(video_path)
    if not frames:
        print(f'❌ Failed to load frames from {video_path}')
        return None

    gt_poses_3d = load_uco_gt_3d(gt_path)
    if gt_poses_3d is None:
        print(f'❌ Failed to load ground-truth poses from {gt_path}')
        return None

    total_frames = min(len(frames), len(gt_poses_3d))
    selected_indices = [frame_idx for frame_idx in frame_indices if 0 <= frame_idx < total_frames]
    if not selected_indices:
        print('❌ No valid frame indices were provided')
        return None

    print('\n' + '=' * 80)
    print(f'Processing {sequence_name} | {args.camera}')
    print(f'Video: {video_path}')
    print(f'GT 3D: {gt_path}')
    print(f'Frames: {len(frames)} | FPS: {fps:.2f} | Size: {width}x{height}')
    print(f'Selected frames: {selected_indices}')
    print('=' * 80)

    yolo_poses_2d, _, performance_metrics = estimate_yolo_poses(
        args.yolo_model_instance,
        frames,
        args.img_size,
        args.device_resolved,
        batch_size=args.batch_size,
    )

    if len(yolo_poses_2d) != len(frames):
        print(f'⚠ YOLO output length mismatch: {len(yolo_poses_2d)} vs {len(frames)}')
        return None

    output_sequence_dir = os.path.join(args.output_dir, str(folder_idx), f'{subfolder_idx:02d}', args.camera)
    os.makedirs(output_sequence_dir, exist_ok=True)

    written = 0
    for frame_idx in selected_indices:
        window_indices = build_window_indices(frame_idx, len(frames), args.window_size)
        pose_window = yolo_poses_2d[window_indices]
        pred_pose_3d = predict_posemamba_window(
            args.posemamba_model,
            args.posemamba_config,
            pose_window,
            frames[frame_idx].shape,
            args.device_resolved,
            use_flip=args.flip_tta,
        )

        gt_pose_3d = gt_poses_3d[frame_idx] if frame_idx < len(gt_poses_3d) else None
        output_path = save_frame_comparison(
            frames[frame_idx],
            gt_pose_3d,
            pred_pose_3d,
            sequence_name,
            frame_idx,
            output_sequence_dir,
        )
        written += 1
        print(f'✓ Saved {output_path}')

    if args.device_resolved.startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()

    print(
        f"✓ {sequence_name} {args.camera}: saved={written}, "
        f"YOLO FPS={performance_metrics['fps']:.2f}, "
        f"mean inference={performance_metrics['mean_inference_time'] * 1000.0:.2f} ms"
    )
    return output_sequence_dir


def main():
    parser = argparse.ArgumentParser(description='Save selected UCO frame comparisons with GT 3D overlay and PoseMamba 3D prediction')
    parser.add_argument('--sequence', type=str, required=True, help='UCO sequence in folder/subfolder format, for example 0/01')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='Frame indices to process')
    parser.add_argument('--camera', type=str, default='cam0', choices=DEFAULT_CAMERAS, help='Camera to use')
    parser.add_argument('--model-path', '--yolo-model', dest='yolo_model', type=str, default=DEFAULT_YOLO_MODEL_PATH, help='Path to the trained YOLO pose model')
    parser.add_argument('--posemamba-config', type=str, default=DEFAULT_POSEMAMBA_CONFIG, help='Path to the PoseMamba config file')
    parser.add_argument('--posemamba-checkpoint', type=str, default=DEFAULT_POSEMAMBA_CHECKPOINT, help='Path to the PoseMamba checkpoint file')
    parser.add_argument('--gt-3d-file', type=str, default=DEFAULT_GT_3D_FILE, help='Explicit path to the UCO p3d.txt file')
    parser.add_argument('--output-dir', type=str, default='uco_gt_pred_selected_frames', help='Directory to save output PNGs')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO input image size')
    parser.add_argument('--batch-size', type=int, default=16, help='YOLO batch size over frames')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--window-size', type=int, default=5, help='PoseMamba temporal window size (must be odd)')
    parser.add_argument('--flip-tta', action='store_true', help='Enable flip test-time augmentation for PoseMamba')
    parser.add_argument('--disable-triton', action='store_true', help='Disable Triton imports for PoseMamba')
    parser.add_argument('--no-save-images', dest='save_images', action='store_false', help='Do not save comparison images')
    parser.set_defaults(save_images=True)
    args = parser.parse_args()

    if args.window_size < 3 or args.window_size % 2 == 0:
        parser.error('--window-size must be an odd integer greater than or equal to 3')
    if args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')
    if not args.save_images:
        print('Saving disabled with --no-save-images; nothing to do.')
        return
    if args.disable_triton:
        os.environ['DISABLE_TRITON'] = '1'

    print('=' * 80)
    print('UCO GT 3D + PoseMamba selected-frame renderer')
    print('=' * 80)
    print(f'Sequence: {args.sequence}')
    print(f'Camera: {args.camera}')
    print(f'Frames: {args.frames}')
    print(f'YOLO model: {args.yolo_model}')
    print(f'PoseMamba config: {args.posemamba_config}')
    print(f'PoseMamba checkpoint: {args.posemamba_checkpoint}')
    print(f'GT 3D file: {args.gt_3d_file or "auto-resolve"}')
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
        device_resolved, gpu_available = check_gpu_availability()
    else:
        device_resolved = args.device
        gpu_available = device_resolved.startswith('cuda') and torch.cuda.is_available()

    print(f'🤖 Loading YOLO model from {args.yolo_model}...')
    try:
        yolo_model = YOLO(args.yolo_model)
        if gpu_available:
            print(f'📦 Moving YOLO model to {device_resolved}...')
            yolo_model.to(device_resolved)
        print('✓ YOLO model loaded successfully')
    except Exception as exc:
        print(f'❌ Error loading YOLO model: {exc}')
        return

    print(f'🤖 Loading PoseMamba model from {args.posemamba_config}...')
    try:
        posemamba_model, posemamba_config = load_posemamba_model(
            args.posemamba_config,
            args.posemamba_checkpoint,
            device_resolved,
        )
        print('✓ PoseMamba model loaded successfully')
    except Exception as exc:
        print(f'❌ Error loading PoseMamba model: {exc}')
        return

    args.yolo_model_instance = yolo_model
    args.posemamba_model = posemamba_model
    args.posemamba_config = posemamba_config
    args.device_resolved = device_resolved

    try:
        process_selected_frames(args.sequence, args.frames, args)
    finally:
        if gpu_available:
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()