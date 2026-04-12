"""
Training script for PoseMamba on the MPI-INF-3DHP dataset.

Based on train.py (Human3.6M) and MotionAGFormer/train_3dhp.py.
Uses the MPI3DHP dataset loader from MotionAGFormer and the PoseMamba model
from lib/, keeping the same loss structure as the H36M training script.

Data files expected in args.data_root:
  - data_train_3dhp.npz
  - data_test_3dhp.npz
"""

import os
import sys
import numpy as np
import argparse
import errno
import datetime
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import yaml
import random

# Add MotionAGFormer to sys.path so its data/utils packages are importable.
# This is required for MPI3DHP and its internal imports (utils.data, etc.).
_motionagformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MotionAGFormer')
if _motionagformer_path not in sys.path:
    sys.path.insert(0, _motionagformer_path)

from lib.utils.tools import get_config
from lib.utils.learning import load_backbone, AverageMeter
from lib.model.loss import (
    loss_mpjpe, n_mpjpe, loss_velocity,
    loss_limb_var, loss_limb_gt,
    loss_angle, loss_angle_velocity,
    weighted_mpjpe,
)
import logger
from logger import colorlogger

# MPI-INF-3DHP dataset loader (lives inside MotionAGFormer).
# Use explicit package import to avoid conflicts with the top-level `data/`
# directory in this repository.
from MotionAGFormer.data.reader.motion_dataset import MPI3DHP

# ---------------------------------------------------------------------------
# MPI-INF-3DHP joint symmetry for horizontal-flip augmentation.
# Indices follow the 17-joint 3DHP skeleton used by MPI3DHP.
#   left_joints  = [8=r_hip, 9=r_knee, 10=r_ankle, 2=r_shoulder, 3=r_elbow, 4=r_wrist]
#   right_joints = [11=l_hip,12=l_knee,13=l_ankle, 5=l_shoulder, 6=l_elbow, 7=l_wrist]
# (same convention as MPI3DHP.left_joints / right_joints)
# ---------------------------------------------------------------------------
JOINTS_LEFT_3DHP  = [8, 9, 10, 2, 3, 4]
JOINTS_RIGHT_3DHP = [11, 12, 13, 5, 6, 7]

# Global logger – initialised inside train_with_config before any function that
# uses it is called.
log = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/pose3d/PoseMamba_train_3dhp_B.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint_3dhp', type=str,
                        metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str,
                        metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str,
                        metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str,
                        metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str,
                        metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_beijing_timestamp():
    local_offset = time.localtime().tm_gmtoff
    beijing_offset = int(8 * 60 * 60)
    offset = local_offset - beijing_offset
    timestamp = int(datetime.datetime.now().timestamp())
    return timestamp - offset


def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    log.info(f'Saving checkpoint to {chk_path}')
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss': min_loss,
    }, chk_path)


def flip_data_3dhp(data):
    """
    Horizontal flip for MPI-INF-3DHP tensors.

    Args:
        data: torch.Tensor of shape (..., J, C) where C[0] is the x coordinate.

    Returns:
        Flipped tensor with x negated and symmetric joints swapped.
    """
    flipped = data.clone()
    flipped[..., 0] *= -1                                          # negate x
    left_data  = flipped[..., JOINTS_LEFT_3DHP,  :].clone()
    right_data = flipped[..., JOINTS_RIGHT_3DHP, :].clone()
    flipped[..., JOINTS_LEFT_3DHP,  :] = right_data               # swap left ↔ right
    flipped[..., JOINTS_RIGHT_3DHP, :] = left_data
    return flipped


def compute_torso_diameter_3dhp(gt_root_rel_mm):
    """
    Compute torso diameter per frame from root-relative 3D joints.

        Uses the distance between shoulder and hip midpoints:
            shoulder_mid = (right_shoulder + left_shoulder) / 2
            hip_mid      = (right_hip + left_hip) / 2

    Args:
        gt_root_rel_mm: torch.Tensor of shape (T, J, 3) in millimetres.

    Returns:
        torch.Tensor of shape (T,) with torso diameter in millimetres.
    """
    right_shoulder = gt_root_rel_mm[:, 2, :]
    left_shoulder = gt_root_rel_mm[:, 5, :]
    right_hip = gt_root_rel_mm[:, 8, :]
    left_hip = gt_root_rel_mm[:, 11, :]

    shoulder_mid = (right_shoulder + left_shoulder) / 2.0
    hip_mid = (right_hip + left_hip) / 2.0
    return torch.norm(shoulder_mid - hip_mid, dim=-1)


def compute_pck_metrics_3dhp(joint_errors_mm, torso_diameters_mm):
    """
    Compute PCK metrics from joint errors on valid 3DHP frames.

    Args:
        joint_errors_mm: torch.Tensor of shape (T_valid, J) with per-joint errors in mm.
        torso_diameters_mm: torch.Tensor of shape (T_valid,) with torso diameters in mm.

    Returns:
        dict[str, float]: PCK metrics in [0, 1].
    """
    if joint_errors_mm.numel() == 0:
        return {
            'PCK@10%_torso': 0.0,
            'PCK@20%_torso': 0.0,
            'PCK@30%_torso': 0.0,
            'PCK@100%_torso': 0.0,
            'PCK@10%_150mm': 0.0,
            'PCK@20%_150mm': 0.0,
            'PCK@30%_150mm': 0.0,
            'PCK@100%_150mm': 0.0,
        }

    torso_thresholds = {
        'PCK@10%_torso': 0.10,
        'PCK@20%_torso': 0.20,
        'PCK@30%_torso': 0.30,
        'PCK@100%_torso': 1.00,
    }

    fixed_thresholds = {
        'PCK@10%_150mm': 15.0,
        'PCK@20%_150mm': 30.0,
        'PCK@30%_150mm': 45.0,
        'PCK@100%_150mm': 150.0,
    }

    metrics = {}
    for key, ratio in torso_thresholds.items():
        thr = torso_diameters_mm.unsqueeze(-1) * ratio
        metrics[key] = (joint_errors_mm <= thr).float().mean().item()

    for key, thr_mm in fixed_thresholds.items():
        metrics[key] = (joint_errors_mm <= thr_mm).float().mean().item()

    return metrics


def compute_auc_3dhp(joint_errors_mm, max_threshold_mm=150.0, step_mm=5.0):
    """
    Compute AUC of the PCK curve in [0, max_threshold_mm].

    Args:
        joint_errors_mm: torch.Tensor of shape (N, J) or (N,) in mm.
        max_threshold_mm: maximum threshold in mm for integration.
        step_mm: threshold step in mm.

    Returns:
        float: normalised AUC in [0, 1].
    """
    if joint_errors_mm.numel() == 0:
        return 0.0

    thresholds = torch.arange(
        0.0,
        max_threshold_mm + step_mm,
        step_mm,
        device=joint_errors_mm.device,
        dtype=joint_errors_mm.dtype,
    )
    pck_curve = torch.stack([(joint_errors_mm <= thr).float().mean() for thr in thresholds])
    auc = torch.trapz(pck_curve, thresholds) / max_threshold_mm
    return auc.item()


def timed_model_forward(model_pos, pose_2d):
    """Run one model forward pass and return output plus elapsed time in ms."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model_pos(pose_2d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return output, elapsed_ms


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args, model_pos, test_loader):
    """
    Evaluate PoseMamba on the MPI-INF-3DHP test set.

    Converts normalised predictions back to millimetres and computes MPJPE
    only on valid frames (as indicated by the dataset's validity mask).

    Normalisation used by MPI3DHP (root-relative, joint 14 = origin):
        pos_norm = (pos_mm - root_mm) / (width / 2)
    So  pos_mm  = pos_norm * (width / 2)

    Width is 2048 for TS1–TS4, TS7–..., and 1920 for TS5–TS6.

    Returns:
        mpjpe_mm (float): mean per-joint position error in millimetres.
    """
    log.info('INFO: Testing on MPI-INF-3DHP')
    model_pos.eval()

    total_error = 0.0
    total_valid_frames = 0
    all_valid_joint_errors = []
    all_valid_torso_diameters = []
    inference_call_times_ms = []
    inference_sample_times_ms = []
    inference_frame_times_ms = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            pose_2d, pose_3d_normalized, pose_3d, valid_frames, seq_names = batch
            N = pose_2d.shape[0]

            pose_2d = pose_2d.cuda()
            if args.no_conf:
                pose_2d = pose_2d[:, :, :, :2]   # (N, T, J, 2)

            if args.flip:
                pose_2d_flip = flip_data_3dhp(pose_2d)
                pred_1, t1_ms = timed_model_forward(model_pos, pose_2d)
                pred_flip, t2_ms = timed_model_forward(model_pos, pose_2d_flip)
                total_ms = t1_ms + t2_ms
                inference_call_times_ms.extend([t1_ms, t2_ms])
                # Two forward calls are used to build one prediction batch when flip-test is on.
                inference_sample_times_ms.append(total_ms / max(N, 1))
                T = pose_2d.shape[1]
                inference_frame_times_ms.append(total_ms / max(N * T, 1))
                # Undo the flip on the predictions
                pred_flip[..., 0] *= -1
                lf = pred_flip[..., JOINTS_LEFT_3DHP,  :].clone()
                rf = pred_flip[..., JOINTS_RIGHT_3DHP, :].clone()
                pred_flip[..., JOINTS_LEFT_3DHP,  :] = rf
                pred_flip[..., JOINTS_RIGHT_3DHP, :] = lf
                predicted_3d_pos = (pred_1 + pred_flip) / 2.0
            else:
                predicted_3d_pos, t_ms = timed_model_forward(model_pos, pose_2d)   # (N, T, J, 3)
                inference_call_times_ms.append(t_ms)
                inference_sample_times_ms.append(t_ms / max(N, 1))
                T = pose_2d.shape[1]
                inference_frame_times_ms.append(t_ms / max(N * T, 1))

            # Make root-relative at joint 14 (pelvis in 3DHP)
            predicted_3d_pos = predicted_3d_pos - predicted_3d_pos[:, :, 14:15, :]

            # Move everything to CPU for per-sample processing
            predicted_3d_pos = predicted_3d_pos.cpu()
            # GT: root-relative, millimetre scale
            gt_root_rel = pose_3d - pose_3d[:, :, 14:15, :]  # (N, T, J, 3) in mm

            for i in range(N):
                seq_name = seq_names[i]
                width    = 1920.0 if seq_name in ("TS5", "TS6") else 2048.0
                pred_mm  = predicted_3d_pos[i] * (width / 2.0)   # (T, J, 3)

                valid_mask  = valid_frames[i].bool()              # (T,)
                pred_valid  = pred_mm[valid_mask]                 # (T_valid, J, 3)
                gt_valid    = gt_root_rel[i][valid_mask]          # (T_valid, J, 3)

                if pred_valid.shape[0] == 0:
                    continue

                # Per-joint errors used for PCK/AUC.
                joint_errors = torch.norm(pred_valid - gt_valid, dim=-1)  # (T_valid, J)
                torso_diameters = compute_torso_diameter_3dhp(gt_valid)    # (T_valid,)
                all_valid_joint_errors.append(joint_errors)
                all_valid_torso_diameters.append(torso_diameters)

                # Per-frame MPJPE averaged over joints, then summed over frames
                per_frame = joint_errors.mean(dim=-1)  # (T_valid,)
                total_error        += per_frame.sum().item()
                total_valid_frames += pred_valid.shape[0]

    if total_valid_frames == 0:
        log.warning('No valid frames found during evaluation!')
        return float('inf')

    mpjpe_mm = total_error / total_valid_frames
    joint_errors_all = torch.cat(all_valid_joint_errors, dim=0)
    torso_diameters_all = torch.cat(all_valid_torso_diameters, dim=0)
    pck_metrics = compute_pck_metrics_3dhp(joint_errors_all, torso_diameters_all)
    auc = compute_auc_3dhp(joint_errors_all, max_threshold_mm=150.0, step_mm=5.0)

    call_times = np.array(inference_call_times_ms, dtype=np.float64)
    sample_times = np.array(inference_sample_times_ms, dtype=np.float64)
    frame_times = np.array(inference_frame_times_ms, dtype=np.float64)
    sample_mean_ms = sample_times.mean() if sample_times.size > 0 else 0.0
    sample_p50_ms = np.percentile(sample_times, 50) if sample_times.size > 0 else 0.0
    sample_p95_ms = np.percentile(sample_times, 95) if sample_times.size > 0 else 0.0
    sample_fps_mean = (1000.0 / sample_mean_ms) if sample_mean_ms > 0 else 0.0
    sample_fps_p50 = (1000.0 / sample_p50_ms) if sample_p50_ms > 0 else 0.0
    sample_fps_p95 = (1000.0 / sample_p95_ms) if sample_p95_ms > 0 else 0.0
    frame_mean_ms = frame_times.mean() if frame_times.size > 0 else 0.0
    frame_fps_mean = (1000.0 / frame_mean_ms) if frame_mean_ms > 0 else 0.0
    window_center_delay_est_ms = ((args.n_frames - 1) / 2.0) * sample_mean_ms

    log.info(f'Protocol #1 Error (MPJPE): {mpjpe_mm:.2f} mm')
    if call_times.size > 0:
        log.info(
            'Inference time / forward call (ms): '
            f'mean={call_times.mean():.3f}, p50={np.percentile(call_times, 50):.3f}, '
            f'p95={np.percentile(call_times, 95):.3f}'
        )
    if sample_times.size > 0:
        log.info(
            'Inference time / sample (ms): '
            f'mean={sample_times.mean():.3f}, p50={np.percentile(sample_times, 50):.3f}, '
            f'p95={np.percentile(sample_times, 95):.3f}'
        )
    if frame_times.size > 0:
        log.info(
            'Inference time / frame (ms): '
            f'mean={frame_times.mean():.4f}, p50={np.percentile(frame_times, 50):.4f}, '
            f'p95={np.percentile(frame_times, 95):.4f}'
        )
    log.info(
        f'Inference FPS / sample: mean={sample_fps_mean:.2f}, '
        f'p50={sample_fps_p50:.2f}, p95={sample_fps_p95:.2f}'
    )
    log.info(f'Inference FPS / frame: mean={frame_fps_mean:.2f}')
    log.info(f'Estimated window-center delay (ms): {window_center_delay_est_ms:.3f}')
    log.info(f'PCK@10%_torso: {pck_metrics["PCK@10%_torso"] * 100:.2f}%')
    log.info(f'PCK@20%_torso: {pck_metrics["PCK@20%_torso"] * 100:.2f}%')
    log.info(f'PCK@30%_torso: {pck_metrics["PCK@30%_torso"] * 100:.2f}%')
    log.info(f'PCK@100%_torso: {pck_metrics["PCK@100%_torso"] * 100:.2f}%')
    log.info(f'PCK@10%_150mm: {pck_metrics["PCK@10%_150mm"] * 100:.2f}%')
    log.info(f'PCK@20%_150mm: {pck_metrics["PCK@20%_150mm"] * 100:.2f}%')
    log.info(f'PCK@30%_150mm: {pck_metrics["PCK@30%_150mm"] * 100:.2f}%')
    log.info(f'PCK@100%_150mm: {pck_metrics["PCK@100%_150mm"] * 100:.2f}%')
    log.info(f'AUC@0-150mm: {auc:.4f}')
    return mpjpe_mm


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(args, model_pos, train_loader, losses, optimizer):
    """One epoch of supervised training on MPI-INF-3DHP."""
    model_pos.train()
    for batch_input, batch_gt in tqdm(train_loader):
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt    = batch_gt.cuda()

        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]   # (N, T, J, 2)
            # batch_gt is already root-relative at joint 14 (from MPI3DHP)

        predicted_3d_pos = model_pos(batch_input)   # (N, T, J, 3)

        optimizer.zero_grad()

        loss_3d_pos      = loss_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_scale    = n_mpjpe(predicted_3d_pos, batch_gt)
        loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
        loss_lv          = loss_limb_var(predicted_3d_pos)
        loss_lg          = loss_limb_gt(predicted_3d_pos, batch_gt)
        loss_a           = loss_angle(predicted_3d_pos, batch_gt)
        loss_av          = loss_angle_velocity(predicted_3d_pos, batch_gt)

        # Uniform joint weights (3DHP joints differ from H36M ordering)
        w_mpjpe  = torch.ones(17).cuda()
        loss_3d_w = weighted_mpjpe(predicted_3d_pos, batch_gt, w_mpjpe)

        # Temporal consistency loss
        dif_seq       = predicted_3d_pos[:, 1:, :, :] - predicted_3d_pos[:, :-1, :, :]
        weights_joints = torch.ones_like(dif_seq).cuda()
        weights_joints = torch.mul(
            weights_joints.permute(0, 1, 3, 2), w_mpjpe
        ).permute(0, 1, 3, 2)
        loss_diff = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))

        loss_total = (
            args.lambda_3d        * loss_3d_pos      +
            args.lambda_scale     * loss_3d_scale     +
            args.lambda_3d_velocity * loss_3d_velocity +
            args.lambda_lv        * loss_lv           +
            args.lambda_lg        * loss_lg           +
            args.lambda_a         * loss_a            +
            args.lambda_av        * loss_av           +
            args.lambda_3dw       * loss_3d_w         +
            args.lambda_diff      * loss_diff
        )

        losses['3d_pos'].update(loss_3d_pos.item(),      batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(),  batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(),              batch_size)
        losses['lg'].update(loss_lg.item(),              batch_size)
        losses['angle'].update(loss_a.item(),            batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(),        batch_size)

        loss_total.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_with_config(args, opts):
    # Append timestamp to checkpoint dir so each run is unique
    opts.checkpoint = (
        opts.checkpoint + '_' +
        datetime.datetime.fromtimestamp(
            get_beijing_timestamp()
        ).strftime('%Y_%m_%d_T_%H_%M_%S')
    )

    global log
    log = colorlogger(opts.checkpoint, log_name='log.txt')
    log.info(args)

    # Persist config alongside checkpoints
    with open(os.path.join(opts.checkpoint, 'config.yaml'), 'w') as f:
        yaml.dump(dict(args), f, sort_keys=False)

    log.info(f"Number of GPUs found: {torch.cuda.device_count()}")

    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, 'logs'))

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    log.info('Loading MPI-INF-3DHP dataset...')

    common_loader_params = dict(
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    train_dataset = MPI3DHP(args, train=True)
    test_dataset  = MPI3DHP(args, train=False)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  **common_loader_params)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size,
                               shuffle=False, **common_loader_params)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    min_loss       = float('inf')
    model_backbone = load_backbone(args)
    model_params   = sum(p.numel() for p in model_backbone.parameters())
    log.info(f'INFO: Trainable parameter count: {model_params}')

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
    model_pos = model_backbone

    # ------------------------------------------------------------------
    # Load weights (fine-tune from H36M checkpoint, resume, or evaluate)
    # ------------------------------------------------------------------
    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint {chk_filename}')
            checkpoint = torch.load(chk_filename,
                                    map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=True)
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            log.info(f'Loading pretrained checkpoint {chk_filename}')
            checkpoint = torch.load(chk_filename,
                                    map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=True)
    else:
        # Auto-resume from the latest checkpoint in the same directory
        chk_filename = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            log.info(f'Loading checkpoint {chk_filename}')
            checkpoint = torch.load(chk_filename,
                                    map_location=lambda storage, loc: storage)
            model_pos.load_state_dict(checkpoint['model_pos'], strict=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    if not opts.evaluate:
        lr         = args.learning_rate
        optimizer  = optim.AdamW(
            filter(lambda p: p.requires_grad, model_pos.parameters()),
            lr=lr, weight_decay=args.weight_decay,
        )
        lr_decay = args.lr_decay
        st       = 0

        log.info(f'INFO: Training on {len(train_loader)} batches per epoch')

        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                log.info('WARNING: checkpoint has no optimizer state. Reinitialising.')
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']

        for epoch in range(st, args.epochs):
            log.info(f'Training epoch {epoch}.')
            start_time = time.time()

            losses = {
                '3d_pos':          AverageMeter(),
                '3d_scale':        AverageMeter(),
                '3d_velocity':     AverageMeter(),
                'lv':              AverageMeter(),
                'lg':              AverageMeter(),
                'angle':           AverageMeter(),
                'angle_velocity':  AverageMeter(),
                'total':           AverageMeter(),
            }

            train_epoch(args, model_pos, train_loader, losses, optimizer)
            elapsed = (time.time() - start_time) / 60.0

            if args.no_eval:
                log.info('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1, elapsed, lr, losses['3d_pos'].avg))
                e1 = None
            else:
                e1 = evaluate(args, model_pos, test_loader)
                log.info('[%d] time %.2f lr %f 3d_train %f e1(mm) %f' % (
                    epoch + 1, elapsed, lr, losses['3d_pos'].avg, e1))

                train_writer.add_scalar('Error P1 (mm)',     e1,                        epoch + 1)
                train_writer.add_scalar('loss_3d_pos',       losses['3d_pos'].avg,      epoch + 1)
                train_writer.add_scalar('loss_3d_scale',     losses['3d_scale'].avg,    epoch + 1)
                train_writer.add_scalar('loss_3d_velocity',  losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv',           losses['lv'].avg,          epoch + 1)
                train_writer.add_scalar('loss_lg',           losses['lg'].avg,          epoch + 1)
                train_writer.add_scalar('loss_a',            losses['angle'].avg,       epoch + 1)
                train_writer.add_scalar('loss_av',           losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total',        losses['total'].avg,       epoch + 1)

            # Exponential learning-rate decay
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best   = os.path.join(opts.checkpoint, 'best_epoch.bin')
            chk_path        = os.path.join(opts.checkpoint, f'epoch_{epoch}.bin')

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 is not None and e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)

    # ------------------------------------------------------------------
    # Evaluation-only mode
    # ------------------------------------------------------------------
    if opts.evaluate:
        evaluate(args, model_pos, test_loader)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
