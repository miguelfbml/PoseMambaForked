import argparse
import os
import pkg_resources
import time
import pynvml
import threading
from statistics import mean
import numpy as np
import scipy.io as scio
import torch
import wandb
from torch import optim
from tqdm import tqdm
from ptflops import get_model_complexity_info

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import loss_mpjpe_conf
from utils.data import denormalize
from data.reader.motion_dataset import MPI3DHP, Fusion
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader
from utils.learning import AverageMeter, decay_lr_exponentially, load_model_TCPFormer
from utils.tools import count_param_numbers
from utils.utils_3dhp import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class GPUUtilizationMonitor:
    def __init__(self, device_idx=0):
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self.utilization_rates = []
        self.memory_usage = []
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while self.running:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
            self.utilization_rates.append(util.gpu)
            self.memory_usage.append(memory_info.used / memory_info.total * 100)
            time.sleep(0.1)  # Poll every 0.1 seconds for finer granularity

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        pynvml.nvmlShutdown()

    def get_stats(self):
        avg_util = mean(self.utilization_rates) if self.utilization_rates else 0
        avg_mem = mean(self.memory_usage) if self.memory_usage else 0
        return avg_util, avg_mem

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mpi/TCPFormer_mpi_81.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint_mpi',
                        help='new checkpoint directory')
    parser.add_argument('-sd', '--seed', default=1, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts

def train_one_epoch(args, model, train_loader, optimizer, losses):
    model.train()
    gpu_monitor = GPUUtilizationMonitor(device_idx=0)
    gpu_monitor.start()
    batch_times = []
    n_frames = args.n_frames
    
    for x, y, confidences in tqdm(train_loader, desc="Training"):  # <-- Add confidences to the dataloader output
        batch_start = time.perf_counter()
        torch.cuda.synchronize()  # Ensure GPU operations are complete
        
        batch_size = x.shape[0]
        if torch.cuda.is_available():
            x, y, confidences = x.cuda(), y.cuda(), confidences.cuda()  # <-- Move confidences to GPU

        pred = model(x)  # (N, T, 17, 3)

        optimizer.zero_grad()

        # Use confidence-weighted loss
        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_pose_conf = loss_mpjpe_conf(pred, y, confidences)  # Confidence-weighted MPJPE
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av

        #losses['3d_pose_conf'].update(loss_3d_pose_conf.item(), batch_size)
        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)

        loss_total.backward()
        optimizer.step()

        torch.cuda.synchronize()
        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)

    gpu_monitor.stop()
    avg_gpu_util, avg_gpu_mem = gpu_monitor.get_stats()
    mean_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    mean_frame_time = mean_batch_time / n_frames if n_frames > 0 else 0
    
    print(f"Training Mean Batch Time: {mean_batch_time:.4f} seconds")
    print(f"Training Mean Frame Time: {mean_frame_time:.6f} seconds")
    print(f"Training GPU Utilization: {avg_gpu_util:.2f}%")
    print(f"Training GPU Memory Usage: {avg_gpu_mem:.2f}%")
    
    return mean_frame_time, avg_gpu_util, avg_gpu_mem

def input_augmentation(input_2D, model, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape
    input_2D_flip = input_2D[:, 1]
    input_2D_non_flip = input_2D[:, 0]

    output_3D_flip = model(input_2D_flip)
    output_3D_flip[..., 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_non_flip = model(input_2D_non_flip)
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip
    return input_2D, output_3D

def evaluate(model, test_loader, n_frames, test_augmentation=True):
    model.eval()
    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}
    error_sum_test = AccumLoss()
    pck_results = {
        'PCK@10%_torso': 0.0, 'PCK@20%_torso': 0.0, 'PCK@30%_torso': 0.0, 'PCK@100%_torso': 0.0,
        'PCK@10%_150mm': 0.0, 'PCK@20%_150mm': 0.0, 'PCK@30%_150mm': 0.0, 'PCK@100%_150mm': 0.0
    }
    auc_sum = 0.0
    valid_samples = 0
    gpu_monitor = GPUUtilizationMonitor(device_idx=0)
    gpu_monitor.start()
    batch_times = []
    inference_call_times_ms = []
    inference_sample_times_ms = []
    inference_frame_times_ms = []
    
    for data in tqdm(test_loader, desc="Evaluating"):
        batch_start = time.perf_counter()
        torch.cuda.synchronize()
        
        batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_variable('test', [input_2D, gt_3D, batch_cam, scale, bb_box])
        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, 17, 3)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, 17, 3).type(torch.cuda.FloatTensor)

        has_test_views = (input_2D.dim() == 5)
        if has_test_views:
            T = input_2D.shape[2]
            input_2D_non_flip = input_2D[:, 0]
        elif input_2D.dim() == 4:
            T = input_2D.shape[1]
            input_2D_non_flip = input_2D
        else:
            raise ValueError(f'Unexpected input_2D shape in evaluate: {tuple(input_2D.shape)}')

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output_3D_non_flip = model(input_2D_non_flip)
        torch.cuda.synchronize()
        t_non_flip_ms = (time.perf_counter() - t0) * 1000.0

        if test_augmentation and has_test_views:
            input_2D_flip = input_2D[:, 1]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            output_3D_flip = model(input_2D_flip)
            torch.cuda.synchronize()
            t_flip_ms = (time.perf_counter() - t0) * 1000.0

            output_3D_flip[..., 0] *= -1
            output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
            output_3D = (output_3D_non_flip + output_3D_flip) / 2
            total_fwd_ms = t_flip_ms + t_non_flip_ms
            inference_call_times_ms.extend([t_flip_ms, t_non_flip_ms])
        else:
            output_3D = output_3D_non_flip
            total_fwd_ms = t_non_flip_ms
            inference_call_times_ms.append(t_non_flip_ms)

        input_2D = input_2D_non_flip
        inference_sample_times_ms.append(total_fwd_ms / max(N, 1))
        inference_frame_times_ms.append(total_fwd_ms / max(N * T, 1))

        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1), 17, 3)
        pad = (n_frames - 1) // 2
        pred_out = output_3D[:, pad].unsqueeze(1)

        pred_out[..., 14, :] = 0
        pred_out = denormalize(pred_out, seq)

        pred_out = pred_out - pred_out[..., 14:15, :]  # Root-relative prediction
        inference_out = pred_out + out_target[..., 14:15, :]  # Final inference (not root-relative for PCK/AUC)
        out_target = out_target - out_target[..., 14:15, :]  # Root-relative ground truth

        torch.cuda.synchronize()
        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)

        # Calculate MPJPE
        joint_error_test = mpjpe_cal(pred_out, out_target).item()
        error_sum_test.update(joint_error_test * N, N)

        # Calculate torso diameters
        torso_diameters = calculate_torso_diameter(gt_3D)

        # Compute PCK for torso-based and 150 mm thresholds
        pred_frame = pred_out[:, 0]  # Shape: (N, 17, 3)
        gt_frame = out_target[:, 0]  # Shape: (N, 17, 3)
        batch_pck = compute_pck(pred_frame, gt_frame, torso_diameters, fixed_threshold=150.0)
        
        for key in pck_results:
            if key in batch_pck:
                pck_results[key] += batch_pck[key] * N

        # Compute AUC
        auc = compute_auc(pred_frame, gt_frame)
        auc_sum += auc * N

        valid_samples += N

        # Store inference data
        for seq_cnt in range(len(seq)):
            seq_name = seq[seq_cnt]
            if seq_name in data_inference:
                data_inference[seq_name] = np.concatenate(
                    (data_inference[seq_name], inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
            else:
                data_inference[seq_name] = inference_out[seq_cnt].permute(2, 1, 0).cpu().numpy()

        # torch.cuda.synchronize()
        # batch_time = time.perf_counter() - batch_start
        # batch_times.append(batch_time)

    for seq_name in data_inference.keys():
        data_inference[seq_name] = data_inference[seq_name][:, :, None, :]

    # Average metrics
    mpjpe_avg = error_sum_test.avg
    for key in pck_results:
        pck_results[key] /= valid_samples
    auc_avg = auc_sum / valid_samples

    gpu_monitor.stop()
    avg_gpu_util, avg_gpu_mem = gpu_monitor.get_stats()
    mean_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    mean_frame_time = mean_batch_time / n_frames if n_frames > 0 else 0

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
    window_center_delay_est_ms = ((n_frames - 1) / 2.0) * sample_mean_ms

    # Print results
    print(f'Evaluation Mean Batch Time: {mean_batch_time:.4f} seconds')
    print(f'Evaluation Mean Frame Time: {mean_frame_time:.6f} seconds')

    call_stats = (
        f'mean={call_times.mean():.3f}, p50={np.percentile(call_times, 50):.3f}, p95={np.percentile(call_times, 95):.3f}'
        if call_times.size > 0 else 'mean=n/a, p50=n/a, p95=n/a'
    )
    sample_stats = (
        f'mean={sample_times.mean():.3f}, p50={np.percentile(sample_times, 50):.3f}, p95={np.percentile(sample_times, 95):.3f}'
        if sample_times.size > 0 else 'mean=n/a, p50=n/a, p95=n/a'
    )
    frame_stats = (
        f'mean={frame_times.mean():.4f}, p50={np.percentile(frame_times, 50):.4f}, p95={np.percentile(frame_times, 95):.4f}'
        if frame_times.size > 0 else 'mean=n/a, p50=n/a, p95=n/a'
    )

    print(f'Inference time / forward call (ms): {call_stats}')
    print(f'Inference time / sample (ms): {sample_stats}')
    print(f'Inference time / frame (ms): {frame_stats}')
    print(f'Inference FPS / sample: mean={sample_fps_mean:.2f}, p50={sample_fps_p50:.2f}, p95={sample_fps_p95:.2f}')
    print(f'Inference FPS / frame: mean={frame_fps_mean:.2f}')
    print(f'Estimated window-center delay (ms): {window_center_delay_est_ms:.3f}')
    print(f'Evaluation GPU Utilization: {avg_gpu_util:.2f}%')
    print(f'Evaluation GPU Memory Usage: {avg_gpu_mem:.2f}%')
    print(f'Protocol #1 Error (MPJPE): {mpjpe_avg:.2f} mm')
    print(f'PCK@10%_torso: {pck_results["PCK@10%_torso"]*100:.2f}%')
    print(f'PCK@20%_torso: {pck_results["PCK@20%_torso"]*100:.2f}%')
    print(f'PCK@30%_torso: {pck_results["PCK@30%_torso"]*100:.2f}%')
    print(f'PCK@100%_torso: {pck_results["PCK@100%_torso"]*100:.2f}%')
    print(f'PCK@10%_150mm: {pck_results["PCK@10%_150mm"]*100:.2f}%')
    print(f'PCK@20%_150mm: {pck_results["PCK@20%_150mm"]*100:.2f}%')
    print(f'PCK@30%_150mm: {pck_results["PCK@30%_150mm"]*100:.2f}%')
    print(f'PCK@100%_150mm: {pck_results["PCK@100%_150mm"]*100:.2f}%')
    print(f'AUC: {auc_avg:.4f}')

    eval_perf = {
        'sample_time_mean_ms': sample_mean_ms,
        'sample_time_p50_ms': sample_p50_ms,
        'sample_time_p95_ms': sample_p95_ms,
        'sample_fps_mean': sample_fps_mean,
        'sample_fps_p50': sample_fps_p50,
        'sample_fps_p95': sample_fps_p95,
        'frame_time_mean_ms': frame_mean_ms,
        'frame_fps_mean': frame_fps_mean,
        'window_center_delay_est_ms': window_center_delay_est_ms,
    }

    return mpjpe_avg, data_inference, mean_frame_time, avg_gpu_util, avg_gpu_mem, eval_perf

def compute_flops(model, input_shape, device):
    model.eval()
    flops, params = get_model_complexity_info(
        model, input_shape[1:], as_strings=False, print_per_layer_stat=False
    )
    return flops, params

def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_mpjpe, wandb_id):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)

def save_data_inference(path, data_inference, latest):
    if latest:
        mat_path = os.path.join(path, 'inference_data.mat')
    else:
        mat_path = os.path.join(path, 'inference_data_best.mat')
    scio.savemat(mat_path, data_inference)

def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)

    train_dataset = MPI3DHP(args, train=True)
    test_dataset = Fusion(args, train=False)

    common_loader_params = {
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.test_batch_size, **common_loader_params)
    model = load_model_TCPFormer(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=[0])
        model = model.cuda()

    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")

    # Compute FLOPs
    input_shape = (args.test_batch_size, args.n_frames, 17, 3)  # [B, T=27, J=17, C=3]
    flops, params = compute_flops(model.module if isinstance(model, torch.nn.DataParallel) else model, input_shape, device)
    training_flops = flops * 3  # Approximate: forward + 2x backward
    print(f"Model FLOPs (forward pass): {flops / 1e9:.2f} GFLOPs")
    print(f"Training FLOPs per sample (approx): {training_flops / 1e9:.2f} GFLOPs")

    lr = args.learning_rate
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr,
                           amsgrad=True)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_mpjpe = float('inf')
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)

            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                          project='MemoryInducedTransformer',
                          resume="must",
                          settings=wandb.Settings(start_method='fork'))
        else:
            if opts.use_wandb:
                print(f"Run ID: {wandb_id}")
                wandb.init(id=wandb_id,
                          name=opts.wandb_name,
                          project='MemoryInducedTransformer',
                          settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})

    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    test_augmentation = getattr(args, 'test_augmentation', True)

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            with torch.no_grad():
                mpjpe, data_inference, eval_frame_time, eval_gpu_util, eval_gpu_mem, eval_perf = evaluate(
                    model, test_loader, args.n_frames, test_augmentation=test_augmentation
                )
                save_data_inference(opts.new_checkpoint, data_inference, latest=True)
                print(f"Inference data saved to: {os.path.join(opts.new_checkpoint, 'inference_data.mat')}")
                if opts.use_wandb:
                    wandb.log({
                        'eval/mean_frame_time': eval_frame_time,
                        'eval/gpu_utilization': eval_gpu_util,
                        'eval/gpu_memory_usage': eval_gpu_mem,
                        'eval/flops_per_sample': flops / 1e9,
                        'eval/inference_sample_time_mean_ms': eval_perf['sample_time_mean_ms'],
                        'eval/inference_sample_time_p95_ms': eval_perf['sample_time_p95_ms'],
                        'eval/inference_sample_fps_mean': eval_perf['sample_fps_mean'],
                        'eval/inference_sample_fps_p95': eval_perf['sample_fps_p95'],
                        'eval/window_center_delay_est_ms': eval_perf['window_center_delay_est_ms'],
                    })
                exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = ['3d_pose', '3d_pose_conf', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_frame_time, train_gpu_util, train_gpu_mem = train_one_epoch(args, model, train_loader, optimizer, losses)
        
        with torch.no_grad():
            mpjpe, data_inference, eval_frame_time, eval_gpu_util, eval_gpu_mem, eval_perf = evaluate(
                model, test_loader, args.n_frames, test_augmentation=test_augmentation
            )

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
            save_data_inference(opts.new_checkpoint, data_inference, latest=False)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_mpjpe, wandb_id)
        save_data_inference(opts.new_checkpoint, data_inference, latest=True)

        if opts.use_wandb:
            wandb_log_dict = {
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_pose_conf': losses['3d_pose_conf'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'eval/mpjpe': mpjpe,
                'eval/min_mpjpe': min_mpjpe,
                'train/mean_frame_time': train_frame_time,
                'train/gpu_utilization': train_gpu_util,
                'train/gpu_memory_usage': train_gpu_mem,
                'eval/mean_frame_time': eval_frame_time,
                'eval/gpu_utilization': eval_gpu_util,
                'eval/gpu_memory_usage': eval_gpu_mem,
                'eval/inference_sample_time_mean_ms': eval_perf['sample_time_mean_ms'],
                'eval/inference_sample_time_p95_ms': eval_perf['sample_time_p95_ms'],
                'eval/inference_sample_fps_mean': eval_perf['sample_fps_mean'],
                'eval/inference_sample_fps_p95': eval_perf['sample_fps_p95'],
                'eval/window_center_delay_est_ms': eval_perf['window_center_delay_est_ms'],
                'train/flops_per_sample': training_flops / 1e9,
                'eval/flops_per_sample': flops / 1e9
            }
            wandb.log(wandb_log_dict, step=epoch + 1)

        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

    if opts.use_wandb:
        artifact = wandb.Artifact(f'model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)

def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = get_config(opts.config)
    #args.n_frames = 27  # Ensure T=27 as specified
    train(args, opts)

if __name__ == '__main__':
    main()