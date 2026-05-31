"""
Compare ground-truth 3D poses, the source image, and PoseMamba 3D predictions
for selected frames.

This script mirrors the selected-frame save flow from compare_gt_yolo_selected_frames.py,
but the saved figure has three panels:
- left: ground-truth 3D skeleton
- middle: RGB frame image
- right: predicted 3D skeleton from YOLOv11x-Pose + PoseMamba

Example:
python compare_gt_pred_posemamba_selected_frames.py --sequence TS1 --frames 0 10 20 \
    --yolo-model zdemo/weights/yolo/best.pt \
    --posemamba-config configs/pose3d/testing/notestaug/PoseMamba_train_3dhp_S_5.yaml \
    --posemamba-checkpoint zdemo/weights/PoseMamba/ModelS/best_epoch_5.bin
"""

import argparse
import gc
import glob
import importlib.util
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
PROJECT_ROOT = CURRENT_DIR
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
DEFAULT_GT_DATA = os.path.join(PROJECT_ROOT, 'data', 'motion3d', 'data_test_3dhp.npz')
DEFAULT_IMAGE_ROOT = '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set'


def resolve_existing_path(candidates):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else None


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


def load_image_folder(sequence_name, image_root):
    image_folder = os.path.join(image_root, sequence_name, 'imageSequence')
    if not os.path.exists(image_folder):
        return None

    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_files.extend(glob.glob(os.path.join(image_folder, '*.png')))
    image_files.sort()
    return image_files if image_files else None


def load_gt_sequence_data(gt_data_path, sequence_name):
    if not os.path.exists(gt_data_path):
        return None, None, None

    data = np.load(gt_data_path, allow_pickle=True)['data'].item()
    if sequence_name not in data:
        return None, None, None

    seq_data = data[sequence_name]
    poses_2d = seq_data['data_2d']
    poses_3d = seq_data['data_3d']
    return poses_2d, poses_3d, sequence_name


def load_frames_by_indices(image_files, frame_indices):
    frames = []
    valid_indices = []

    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(image_files):
            continue

        frame = cv2.imread(image_files[frame_idx])
        if frame is None:
            continue

        frames.append(frame)
        valid_indices.append(frame_idx)

    return frames, valid_indices


def build_window_indices(center_idx, total_frames, window_size):
    half_window = window_size // 2
    indices = []
    for offset in range(-half_window, half_window + 1):
        index = center_idx + offset
        index = max(0, min(total_frames - 1, index))
        indices.append(index)
    return indices


def infer_posemamba_3d(model, posemamba_config, frames, device, img_size, batch_size=None, use_flip=False):
    yolo_poses_2d, _, performance_metrics = estimate_yolo_poses(
        model['yolo'],
        frames,
        img_size,
        device,
        batch_size=batch_size,
    )

    poses_2d = np.asarray(yolo_poses_2d, dtype=np.float32)
    poses_2d_normalized = poses_2d.copy()
    poses_2d_normalized[:, :, :2] = normalize_screen_coordinates(
        poses_2d[:, :, :2],
        frames[0].shape[1],
        frames[0].shape[0],
    )

    input_2d = torch.from_numpy(poses_2d_normalized).unsqueeze(0).float()
    if bool(getattr(posemamba_config, 'no_conf', False)):
        input_2d = input_2d[:, :, :, :2]

    if device.startswith('cuda'):
        input_2d = input_2d.cuda()

    with torch.no_grad():
        if use_flip:
            input_2d_flip = flip_data(input_2d)
            pred_3d_main = model['posemamba'](input_2d)
            pred_3d_flip = model['posemamba'](input_2d_flip)
            pred_3d = (pred_3d_main + flip_data(pred_3d_flip)) / 2.0
        else:
            pred_3d = model['posemamba'](input_2d)

    middle_idx = len(frames) // 2
    pred_3d_middle = pred_3d[0, middle_idx].detach().cpu().numpy()
    return pred_3d_middle, yolo_poses_2d, performance_metrics


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


def prepare_pose_for_plot(pose_3d):
    # Align coordinate post-processing with vis.py demo pipeline:
    # - convert camera->world using the same rotation
    # - shift Z so min(Z)=0
    # - normalize by the global max coordinate (unit-scale)
    if pose_3d is None:
        return None
    post_out = pose_3d.copy().astype(np.float32)

    # use the same rotation quaternion/vector as in vis.py
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088], dtype=np.float32)
    try:
        post_out = camera_to_world(post_out, R=rot, t=0)
    except Exception:
        # fallback: if camera_to_world isn't compatible, continue with pose as-is
        pass

    # shift so lowest Z is zero (same as vis.py)
    post_out[:, 2] -= np.min(post_out[:, 2])

    max_value = np.max(post_out)
    if max_value < 1e-6:
        return post_out
    post_out = post_out / max_value
    return post_out


def plot_3d_skeleton(ax, pose_3d, title, line_color, point_color, missing_text):
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
        ax.text(x + 18, y + 18, z + 18, f'{joint_idx}', fontsize=8, color='black')

    ax.grid(True, alpha=0.25)


def save_frame_comparison(image, gt_pose, pred_pose, sequence_name, frame_idx, output_dir):
    gt_plot = prepare_pose_for_plot(gt_pose)
    pred_plot = prepare_pose_for_plot(pred_pose)

    valid_gt = gt_plot[~np.isnan(gt_plot) & ~np.isinf(gt_plot)]
    valid_pred = pred_plot[~np.isnan(pred_plot) & ~np.isinf(pred_plot)]
    if valid_gt.size == 0:
        all_poses = valid_pred
    else:
        all_poses = np.vstack([valid_gt.reshape(-1, 3), valid_pred.reshape(-1, 3)])
    min_value = np.min(all_poses, axis=0)
    max_value = np.max(all_poses, axis=0)
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding

    fig = plt.figure(figsize=(18, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1])
    ax_gt = fig.add_subplot(grid[0, 0], projection='3d')
    ax_img = fig.add_subplot(grid[0, 1])
    ax_pred = fig.add_subplot(grid[0, 2], projection='3d')

    fig.suptitle(
        f'Ground Truth vs PoseMamba 3D Prediction - {sequence_name} | Frame {frame_idx}',
        fontsize=14,
    )

    # Compute a shared extent but center each subplot on joint index 14
    span = max_value - min_value
    # ensure a sensible extent if pose is degenerate
    span = np.where(span <= 1e-6, 1.0, span)
    padding = (max_value - min_value) * 0.1
    min_value -= padding
    max_value += padding
    span = max_value - min_value
    half_extent = span / 2.0

    def _get_center(plot):
        if plot is None or (np.isnan(plot).all() or np.isinf(plot).all()):
            return (min_value + max_value) / 2.0
        if plot.shape[0] > 14:
            return plot[14]
        return (min_value + max_value) / 2.0

    center_gt = _get_center(gt_plot)
    center_pred = _get_center(pred_plot)

    # Set limits centered on joint-14 for each subplot, using the same half-extent
    ax_gt.set_xlim3d([center_gt[0] - half_extent[0], center_gt[0] + half_extent[0]])
    ax_gt.set_ylim3d([center_gt[1] - half_extent[1], center_gt[1] + half_extent[1]])
    ax_gt.set_zlim3d([center_gt[2] - half_extent[2], center_gt[2] + half_extent[2]])

    ax_pred.set_xlim3d([center_pred[0] - half_extent[0], center_pred[0] + half_extent[0]])
    ax_pred.set_ylim3d([center_pred[1] - half_extent[1], center_pred[1] + half_extent[1]])
    ax_pred.set_zlim3d([center_pred[2] - half_extent[2], center_pred[2] + half_extent[2]])

    plot_3d_skeleton(
        ax_gt,
        gt_plot,
        'Ground Truth 3D',
        line_color='royalblue',
        point_color='deepskyblue',
        missing_text='No GT Data',
    )
    plot_3d_skeleton(
        ax_pred,
        pred_plot,
        'PoseMamba Prediction',
        line_color='tomato',
        point_color='salmon',
        missing_text='No Prediction',
    )

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax_img.imshow(rgb_image)
    ax_img.set_title('Frame Image', fontsize=12)
    ax_img.axis('off')
    ax_img.text(
        0.02,
        0.96,
        f'Frame {frame_idx}',
        transform=ax_img.transAxes,
        fontsize=11,
        color='white',
        bbox=dict(facecolor='black', alpha=0.55, pad=4),
        va='top',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}_gt_image_posemamba3d.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def process_selected_frames(sequence_name, frame_indices, args):
    gt_data_path = resolve_existing_path(
        [
            args.gt_data,
            os.path.join(PROJECT_ROOT, 'data', 'motion3d', 'data_test_3dhp.npz'),
            os.path.join(PROJECT_ROOT, '..', 'motion3d', 'data_test_3dhp.npz'),
            os.path.join(PROJECT_ROOT, 'motion3d', 'data_test_3dhp.npz'),
        ]
    )
    gt_poses_2d, gt_poses_3d, seq_name = load_gt_sequence_data(gt_data_path, sequence_name)
    if gt_poses_3d is None:
        print(f'Could not load GT data for {sequence_name} from {gt_data_path}')
        return None

    image_files = load_image_folder(sequence_name, args.image_root)
    if image_files is None:
        print(f'Could not find image folder for sequence {sequence_name}')
        return None

    total_frames = min(len(gt_poses_3d), len(image_files))
    selected_frames = [frame_idx for frame_idx in frame_indices if 0 <= frame_idx < total_frames]
    if not selected_frames:
        print('No valid frame indices were provided')
        return None

    device, gpu_available = check_gpu_availability() if args.device == 'auto' else (args.device, args.device.startswith('cuda') and torch.cuda.is_available())
    if args.yolo_model is None:
        yolo_model_path = DEFAULT_YOLO_MODEL_PATH
    else:
        yolo_model_path = args.yolo_model

    if not os.path.exists(yolo_model_path):
        print(f'YOLO model not found: {yolo_model_path}')
        return None
    if not os.path.exists(args.posemamba_config):
        print(f'PoseMamba config not found: {args.posemamba_config}')
        return None
    if not os.path.exists(args.posemamba_checkpoint):
        print(f'PoseMamba checkpoint not found: {args.posemamba_checkpoint}')
        return None

    print(f'Loading YOLO model from: {yolo_model_path}')
    yolo_model = YOLO(yolo_model_path)
    if gpu_available:
        yolo_model.to(device)

    print(f'Loading PoseMamba model from: {args.posemamba_config}')
    posemamba_model, posemamba_config = load_posemamba_model(
        args.posemamba_config,
        args.posemamba_checkpoint,
        device,
    )

    model_bundle = {
        'yolo': yolo_model,
        'posemamba': posemamba_model,
    }

    output_sequence_dir = os.path.join(args.output_dir, sequence_name)
    os.makedirs(output_sequence_dir, exist_ok=True)

    saved_files = []
    for frame_idx in selected_frames:
        window_indices = build_window_indices(frame_idx, total_frames, args.window_size)
        window_frames, valid_indices = load_frames_by_indices(image_files, window_indices)
        if len(window_frames) != args.window_size:
            print(f'Skipping frame {frame_idx}: unable to assemble a full window')
            continue

        gt_pose = gt_poses_3d[frame_idx]
        pred_pose, _, performance_metrics = infer_posemamba_3d(
            model_bundle,
            posemamba_config,
            window_frames,
            device,
            args.img_size,
            batch_size=args.batch_size,
            use_flip=args.flip_tta,
        )

        saved_path = save_frame_comparison(
            window_frames[len(window_frames) // 2],
            gt_pose,
            pred_pose,
            sequence_name,
            frame_idx,
            output_sequence_dir,
        )
        saved_files.append(saved_path)
        print(f'Saved frame {frame_idx} -> {saved_path}')
        print(
            f"  YOLO FPS: {performance_metrics['fps']:.2f} | "
            f"Mean inference: {performance_metrics['mean_inference_time'] * 1000:.2f} ms"
        )

    if gpu_available:
        torch.cuda.empty_cache()
    gc.collect()

    print(f'Output folder: {os.path.abspath(output_sequence_dir)}')
    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Save GT / image / PoseMamba 3D comparisons for selected frames')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence name, for example TS1')
    parser.add_argument('--frames', type=int, nargs='+', required=True, help='0-based frame indices to process')
    parser.add_argument('--output-dir', type=str, default='comparison_posemamba_selected_frames', help='Directory for saved comparison images')
    parser.add_argument('--image-root', type=str, default=DEFAULT_IMAGE_ROOT, help='Root folder that contains <sequence>/imageSequence')
    parser.add_argument('--gt-data', type=str, default=DEFAULT_GT_DATA, help='Path to the GT data_test_3dhp.npz file')
    parser.add_argument(
        '--yolo-model',
        '--model-path',
        dest='yolo_model',
        type=str,
        default=DEFAULT_YOLO_MODEL_PATH,
        help='Path to the trained YOLOv11x-Pose model',
    )
    parser.add_argument('--posemamba-config', type=str, default=DEFAULT_POSEMAMBA_CONFIG, help='Path to the PoseMamba config file')
    parser.add_argument('--posemamba-checkpoint', type=str, default=DEFAULT_POSEMAMBA_CHECKPOINT, help='Path to the PoseMamba checkpoint file')
    parser.add_argument('--img-size', type=int, default=640, help='YOLO input image size')
    parser.add_argument('--batch-size', type=int, default=None, help='YOLO batch size override')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, cuda:0, etc.')
    parser.add_argument('--window-size', type=int, default=5, help='PoseMamba temporal window size (must be odd)')
    parser.add_argument('--flip-tta', action='store_true', help='Enable flip test-time augmentation for PoseMamba')
    parser.add_argument('--disable-triton', action='store_true', help='Disable Triton imports for PoseMamba')
    args = parser.parse_args()

    if args.window_size < 3 or args.window_size % 2 == 0:
        parser.error('--window-size must be an odd integer greater than or equal to 3')

    if args.batch_size is not None and args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')

    if args.disable_triton:
        os.environ['DISABLE_TRITON'] = '1'

    print('=' * 80)
    print('GT / Image / PoseMamba selected-frame comparison')
    print('=' * 80)
    print(f'Sequence: {args.sequence}')
    print(f'Frames: {args.frames}')
    print(f'Image root: {args.image_root}')
    print(f'GT data: {args.gt_data}')
    print(f'YOLO model: {args.yolo_model}')
    print(f'PoseMamba config: {args.posemamba_config}')
    print(f'PoseMamba checkpoint: {args.posemamba_checkpoint}')
    print(f'Window size: {args.window_size}')
    print(f'Flip TTA: {"Enabled" if args.flip_tta else "Disabled"}')
    print(f'Device: {args.device}')
    print('=' * 80)

    process_selected_frames(args.sequence, args.frames, args)


if __name__ == '__main__':
    main()
