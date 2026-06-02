"""
Compute mean 3D angle error between UCO ground-truth poses and PoseMamba
predictions for every folder/subfolder/camera sequence.

This script does not save frames or videos. It walks the UCO dataset layout:
folders 0-26, subfolders 09-16, cameras cam0-cam4 by default, computes the
mean absolute angle difference per sequence, and writes a text summary file.

Angle definitions:
- Ground truth: joints 0-1-2
- PoseMamba prediction:
  - subfolders 09-12 -> joints 5-6-7
  - subfolders 13-16 -> joints 2-3-4

Example:
python compare_uco_gt_pred_posemamba_angle_summary.py \
    --output-file uco_angle_summary.txt \
    --device cuda:0
"""

import argparse
import gc
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

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


DEFAULT_FOLDERS = list(range(0, 27))
DEFAULT_SUBFOLDERS = list(range(9, 17))
DEFAULT_CAMERAS = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']
DEFAULT_REPORT_FILE = 'uco_angle_error_report.txt'
DEFAULT_GT_3D_FILE = ''
UCO_DATASET_PATH = '/nas-ctm01/datasets/public/UCO Physical Rehabilitation/dataset/clips_mp4'


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
    post_out = apply_upright_correction(post_out)
    post_out = make_root_relative_3d(post_out)
    post_out = scale_pose_to_max(post_out)
    return post_out


def estimate_yolo_poses(model, frames, img_size=640, device='cpu', batch_size=None):
    yolo_poses = []
    confidences = []
    inference_times = []

    if batch_size is None:
        batch_size = 32 if device.startswith('cuda') else 16

    for start_idx in range(0, len(frames), batch_size):
        batch_frames = frames[start_idx:start_idx + batch_size]

        for frame in batch_frames:
            try:
                start_time = time.time()
                results = model.predict(frame, verbose=False, imgsz=img_size, conf=0.65, device=device)
                inference_times.append(time.time() - start_time)

                if (
                    results and len(results) > 0 and
                    hasattr(results[0], 'keypoints') and
                    results[0].keypoints is not None and
                    len(results[0].keypoints.xy) > 0
                ):
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    conf = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

                    if keypoints.shape[0] == 17:
                        yolo_poses.append(keypoints)
                        confidences.append(conf)
                    else:
                        padded_kpts = np.zeros((17, 2))
                        padded_conf = np.zeros(17)
                        n_kpts = min(17, keypoints.shape[0])
                        padded_kpts[:n_kpts] = keypoints[:n_kpts]
                        padded_conf[:n_kpts] = conf[:n_kpts] if len(conf) > 0 else 0.5
                        yolo_poses.append(padded_kpts)
                        confidences.append(padded_conf)
                else:
                    yolo_poses.append(np.zeros((17, 2)))
                    confidences.append(np.zeros(17))
            except Exception as exc:
                print(f'Error in YOLO inference: {exc}')
                yolo_poses.append(np.zeros((17, 2)))
                confidences.append(np.zeros(17))
                inference_times.append(0.0)

        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        gc.collect()

    yolo_poses = np.asarray(yolo_poses, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    total_inference_time = float(sum(inference_times))
    mean_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
    fps = 1.0 / mean_inference_time if mean_inference_time > 0 else 0.0

    performance_metrics = {
        'total_inference_time': total_inference_time,
        'mean_inference_time': mean_inference_time,
        'fps': fps,
        'processed_frames': len(frames),
    }

    return yolo_poses, confidences, performance_metrics


def compute_joint_angle_degrees(pose_3d, joint_a, joint_b, joint_c):
    if pose_3d is None:
        return None

    max_joint_idx = max(joint_a, joint_b, joint_c)
    if pose_3d.shape[0] <= max_joint_idx:
        return None

    vec_ab = pose_3d[joint_a] - pose_3d[joint_b]
    vec_cb = pose_3d[joint_c] - pose_3d[joint_b]
    norm_ab = float(np.linalg.norm(vec_ab))
    norm_cb = float(np.linalg.norm(vec_cb))
    if norm_ab < 1e-6 or norm_cb < 1e-6:
        return None

    cos_angle = float(np.dot(vec_ab, vec_cb) / (norm_ab * norm_cb))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def get_prediction_angle_triplet(subfolder_idx):
    if 9 <= subfolder_idx <= 12:
        return (5, 6, 7), '5-6-7'
    if 13 <= subfolder_idx <= 16:
        return (2, 3, 4), '2-3-4'
    return None, None


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


def prepare_uco_gt_for_plot(pose_3d):
    if pose_3d is None:
        return None

    pose_plot = np.asarray(pose_3d, dtype=np.float32).copy()
    pose_plot = pose_plot[:3]
    pose_plot = pose_plot - pose_plot[0:1]
    pose_plot = pose_plot[:3]

    max_coord = np.max(np.abs(pose_plot))
    if max_coord < 1e-6:
        return pose_plot
    return pose_plot * (900.0 / max_coord)


def format_mean(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def format_std(values):
    if not values:
        return None
    return float(np.std(np.asarray(values, dtype=np.float32)))


def process_sequence_camera(yolo_model, posemamba_model, posemamba_config, folder, subfolder, camera, args, device):
    sequence_name = f'{folder}/{subfolder:02d}'
    video_path = load_uco_video_path(folder, subfolder, camera)
    if not os.path.exists(video_path):
        return {
            'sequence': sequence_name,
            'folder': folder,
            'subfolder': f'{subfolder:02d}',
            'camera': camera,
            'status': 'missing_video',
        }

    gt_path = resolve_uco_gt_3d_path(folder, subfolder, camera, args.gt_3d_file)
    if not gt_path or not os.path.exists(gt_path):
        return {
            'sequence': sequence_name,
            'folder': folder,
            'subfolder': f'{subfolder:02d}',
            'camera': camera,
            'status': 'missing_gt',
        }

    frames, fps, width, height = load_video_frames(video_path)
    if not frames:
        return {
            'sequence': sequence_name,
            'folder': folder,
            'subfolder': f'{subfolder:02d}',
            'camera': camera,
            'status': 'failed_video_load',
        }

    gt_poses_3d = load_uco_gt_3d(gt_path)
    if gt_poses_3d is None:
        return {
            'sequence': sequence_name,
            'folder': folder,
            'subfolder': f'{subfolder:02d}',
            'camera': camera,
            'status': 'failed_gt_load',
        }

    yolo_poses_2d, _, performance_metrics = estimate_yolo_poses(
        yolo_model,
        frames,
        args.img_size,
        device,
        batch_size=args.batch_size,
    )

    if len(yolo_poses_2d) != len(frames):
        return {
            'sequence': sequence_name,
            'folder': folder,
            'subfolder': f'{subfolder:02d}',
            'camera': camera,
            'status': 'yolo_length_mismatch',
        }

    pred_triplet, pred_triplet_label = get_prediction_angle_triplet(subfolder)
    if pred_triplet is None:
        return {
            'sequence': sequence_name,
            'folder': folder,
            'subfolder': f'{subfolder:02d}',
            'camera': camera,
            'status': 'unsupported_subfolder',
        }

    gt_angles = []
    pred_angles = []
    abs_diffs = []
    valid_frames = 0

    total_frames = min(len(frames), len(gt_poses_3d))
    for frame_idx in range(total_frames):
        window_indices = build_window_indices(frame_idx, len(frames), args.window_size)
        pose_window = yolo_poses_2d[window_indices]
        pred_pose_3d = predict_posemamba_window(
            posemamba_model,
            posemamba_config,
            pose_window,
            frames[frame_idx].shape,
            device,
            use_flip=args.flip_tta,
        )

        gt_pose_3d = gt_poses_3d[frame_idx]
        gt_pose_plot = prepare_uco_gt_for_plot(gt_pose_3d)
        pred_pose_plot = prepare_pose_for_plot(pred_pose_3d)

        gt_angle = compute_joint_angle_degrees(gt_pose_plot, 0, 1, 2)
        pred_angle = compute_joint_angle_degrees(pred_pose_plot, *pred_triplet)

        if gt_angle is None or pred_angle is None:
            continue

        valid_frames += 1
        gt_angles.append(gt_angle)
        pred_angles.append(pred_angle)
        abs_diffs.append(abs(gt_angle - pred_angle))

    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    gc.collect()

    return {
        'sequence': sequence_name,
        'folder': folder,
        'subfolder': f'{subfolder:02d}',
        'camera': camera,
        'status': 'ok',
        'video_path': video_path,
        'gt_path': gt_path,
        'fps': float(fps),
        'width': int(width),
        'height': int(height),
        'total_frames': int(total_frames),
        'valid_frames': int(valid_frames),
        'gt_triplet': '0-1-2',
        'pred_triplet': pred_triplet_label,
        'mean_gt_angle': format_mean(gt_angles),
        'mean_pred_angle': format_mean(pred_angles),
        'mean_abs_error': format_mean(abs_diffs),
        'std_abs_error': format_std(abs_diffs),
        'min_abs_error': float(np.min(abs_diffs)) if abs_diffs else None,
        'max_abs_error': float(np.max(abs_diffs)) if abs_diffs else None,
        'yolo_fps': float(performance_metrics['fps']),
        'yolo_mean_inference_time_ms': float(performance_metrics['mean_inference_time'] * 1000.0),
    }


def write_report(report_path, results, args):
    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)

    successful = [result for result in results if result.get('status') == 'ok']
    by_folder = defaultdict(list)
    by_subfolder = defaultdict(list)
    by_camera = defaultdict(list)

    for result in successful:
        by_folder[result['folder']].append(result)
        by_subfolder[(result['folder'], result['subfolder'])].append(result)
        by_camera[result['camera']].append(result)

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('UCO GT vs PoseMamba angle-error summary\n')
        handle.write('=' * 80 + '\n')
        handle.write(f'Folders: {args.folders}\n')
        handle.write(f'Subfolders: {args.subfolders}\n')
        handle.write(f'Cameras: {args.cameras}\n')
        handle.write(f'Window size: {args.window_size}\n')
        handle.write(f'Flip TTA: {"Enabled" if args.flip_tta else "Disabled"}\n')
        handle.write('\n')

        handle.write('Per-sequence results\n')
        handle.write('-' * 80 + '\n')
        for result in results:
            if result['status'] != 'ok':
                handle.write(
                    f"{result['sequence']} | {result['camera']} | status={result['status']}\n"
                )
                continue

            handle.write(
                f"folder={result['folder']} subfolder={result['subfolder']} camera={result['camera']} "
                f"frames={result['valid_frames']}/{result['total_frames']} "
                f"gt_triplet={result['gt_triplet']} pred_triplet={result['pred_triplet']} "
                f"mean_gt_angle={result['mean_gt_angle']:.3f} deg "
                f"mean_pred_angle={result['mean_pred_angle']:.3f} deg "
                f"mean_abs_error={result['mean_abs_error']:.3f} deg "
                f"std_abs_error={result['std_abs_error']:.3f} deg "
                f"min_abs_error={result['min_abs_error']:.3f} deg "
                f"max_abs_error={result['max_abs_error']:.3f} deg\n"
            )

        handle.write('\n')
        handle.write('Aggregated summaries\n')
        handle.write('-' * 80 + '\n')

        handle.write('By folder\n')
        for folder in sorted(by_folder.keys()):
            folder_results = by_folder[folder]
            handle.write(
                f"folder={folder} sequences={len(folder_results)} "
                f"mean_abs_error={format_mean([r['mean_abs_error'] for r in folder_results]):.3f} deg "
                f"mean_gt_angle={format_mean([r['mean_gt_angle'] for r in folder_results]):.3f} deg "
                f"mean_pred_angle={format_mean([r['mean_pred_angle'] for r in folder_results]):.3f} deg\n"
            )

        handle.write('\nBy subfolder\n')
        for folder, subfolder in sorted(by_subfolder.keys()):
            seq_results = by_subfolder[(folder, subfolder)]
            handle.write(
                f"folder={folder} subfolder={subfolder} sequences={len(seq_results)} "
                f"mean_abs_error={format_mean([r['mean_abs_error'] for r in seq_results]):.3f} deg "
                f"mean_gt_angle={format_mean([r['mean_gt_angle'] for r in seq_results]):.3f} deg "
                f"mean_pred_angle={format_mean([r['mean_pred_angle'] for r in seq_results]):.3f} deg\n"
            )

        handle.write('\nBy camera\n')
        for camera in sorted(by_camera.keys()):
            camera_results = by_camera[camera]
            handle.write(
                f"camera={camera} sequences={len(camera_results)} "
                f"mean_abs_error={format_mean([r['mean_abs_error'] for r in camera_results]):.3f} deg "
                f"mean_gt_angle={format_mean([r['mean_gt_angle'] for r in camera_results]):.3f} deg "
                f"mean_pred_angle={format_mean([r['mean_pred_angle'] for r in camera_results]):.3f} deg\n"
            )

        handle.write('\nOverall\n')
        handle.write(
            f"sequences={len(successful)} "
            f"mean_abs_error={format_mean([r['mean_abs_error'] for r in successful]):.3f} deg "
            f"mean_gt_angle={format_mean([r['mean_gt_angle'] for r in successful]):.3f} deg "
            f"mean_pred_angle={format_mean([r['mean_pred_angle'] for r in successful]):.3f} deg\n"
        )


def main():
    parser = argparse.ArgumentParser(description='Compute UCO GT vs PoseMamba 3D angle error summaries')
    parser.add_argument('--yolo-model', '--model-path', dest='yolo_model', type=str, default=DEFAULT_YOLO_MODEL_PATH, help='Path to the trained YOLO pose model')
    parser.add_argument('--posemamba-config', type=str, default=DEFAULT_POSEMAMBA_CONFIG, help='Path to the PoseMamba config file')
    parser.add_argument('--posemamba-checkpoint', type=str, default=DEFAULT_POSEMAMBA_CHECKPOINT, help='Path to the PoseMamba checkpoint file')
    parser.add_argument('--gt-3d-file', type=str, default=DEFAULT_GT_3D_FILE, help='Optional explicit GT 3D text file; otherwise auto-resolve per sequence')
    parser.add_argument('--report-file', type=str, default=DEFAULT_REPORT_FILE, help='Text file to write the summary report')
    parser.add_argument('--folders', type=int, nargs='*', default=DEFAULT_FOLDERS, help='UCO folders to process (default: 0..26)')
    parser.add_argument('--subfolders', type=int, nargs='*', default=DEFAULT_SUBFOLDERS, help='UCO subfolders to process (default: 9..16)')
    parser.add_argument('--cameras', type=str, nargs='*', default=DEFAULT_CAMERAS, help='UCO cameras to process')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO input image size')
    parser.add_argument('--batch-size', type=int, default=16, help='YOLO batch size over frames')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--window-size', type=int, default=5, help='PoseMamba temporal window size (must be odd)')
    parser.add_argument('--flip-tta', action='store_true', help='Enable flip test-time augmentation for PoseMamba')
    parser.add_argument('--disable-triton', action='store_true', help='Disable Triton imports for PoseMamba')
    args = parser.parse_args()

    if args.window_size < 3 or args.window_size % 2 == 0:
        parser.error('--window-size must be an odd integer greater than or equal to 3')
    if args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')

    if args.disable_triton:
        os.environ['DISABLE_TRITON'] = '1'

    print('=' * 80)
    print('UCO GT vs PoseMamba angle error summary')
    print('=' * 80)
    print(f'Folders: {args.folders}')
    print(f'Subfolders: {args.subfolders}')
    print(f'Cameras: {args.cameras}')
    print(f'YOLO model: {args.yolo_model}')
    print(f'PoseMamba config: {args.posemamba_config}')
    print(f'PoseMamba checkpoint: {args.posemamba_checkpoint}')
    print(f'Report file: {args.report_file}')
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

    results = []
    try:
        for folder in args.folders:
            print(f'--- Folder {folder:02d} ---')
            for subfolder in args.subfolders:
                print(f'  -> Subfolder {folder:02d}/{subfolder:02d}')
                for camera in args.cameras:
                    print(f'     -> Camera {camera}')
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
                        result = {
                            'sequence': f'{folder}/{subfolder:02d}',
                            'folder': folder,
                            'subfolder': f'{subfolder:02d}',
                            'camera': camera,
                            'status': f'error: {exc}',
                        }
                    results.append(result)

                    if result.get('status') == 'ok':
                        print(
                            f"     ✓ mean_abs_error={result['mean_abs_error']:.3f} deg "
                            f"(gt={result['mean_gt_angle']:.3f}, pred={result['mean_pred_angle']:.3f})"
                        )
                    else:
                        print(f"     ! status={result['status']}")

        write_report(args.report_file, results, args)
        print(f'✓ Report written to {args.report_file}')
    finally:
        if gpu_available:
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()