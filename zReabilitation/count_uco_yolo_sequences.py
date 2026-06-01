"""
Process UCO videos with YOLO pose estimation and count detected vs undetected frames.

The script walks folders 0-26 and subfolders 09-16 by default, processes all cameras
unless a subset is provided, optionally saves the raw mp4 for every processed video,
and prints/saves summary counts per sequence and per camera.

Example:
python count_uco_yolo_sequences.py \
    --model-path weights/YOLO/best.pt \
    --output-dir yolo_sequence_outputs
"""

import argparse
import gc
import json
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from compare_gt_yolo_2d import check_gpu_availability  # noqa: E402


UCO_DATASET_PATH = '/nas-ctm01/datasets/public/UCO Physical Rehabilitation/dataset/clips_mp4'
DEFAULT_CAMERAS = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']
FOLDER_RANGE = range(27)
SUBFOLDER_RANGE = range(9, 17)


def load_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None, fps, 0, 0
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return cap, fps, width, height


def format_ratio(numerator, denominator):
    if denominator <= 0:
        return '0/0'
    return f'{numerator}/{denominator}'


def predict_batch(model, frames, img_size, device, confidence):
    results = model.predict(
        frames,
        verbose=False,
        imgsz=img_size,
        conf=confidence,
        device=device,
    )

    batch_predictions = []
    for result in results:
        pose_count = 0
        first_pose = None

        if (
            hasattr(result, 'keypoints')
            and result.keypoints is not None
            and result.keypoints.xy is not None
        ):
            pose_tensor = result.keypoints.xy
            pose_count = int(len(pose_tensor))
            if pose_count > 0:
                first_pose = pose_tensor[0].cpu().numpy()

        batch_predictions.append((pose_count, first_pose))

    return batch_predictions


def make_output_video_path(output_dir, folder, subfolder, camera):
    return os.path.join(output_dir, str(folder), f'{subfolder:02d}', camera, f'{camera}.mp4')


def process_sequence_camera(model, folder, subfolder, camera, args, device):
    sequence_name = f'{folder}/{subfolder:02d}'
    video_path = os.path.join(UCO_DATASET_PATH, str(folder), f'{subfolder:02d}', f'{camera}.mp4')

    if not os.path.exists(video_path):
        print(f'⚠ Skipping {sequence_name} {camera}: missing video')
        return None

    cap, fps, width, height = load_video_capture(video_path)
    if cap is None:
        print(f'⚠ Skipping {sequence_name} {camera}: cannot open video')
        return None

    output_video_path = None
    writer = None
    if args.save_videos:
        output_video_path = make_output_video_path(args.output_dir, folder, subfolder, camera)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            print(f'⚠ Skipping {sequence_name} {camera}: cannot create output video')
            return None

    total_frames = 0
    detected_frames = 0
    not_detected_frames = 0
    total_pose_instances = 0
    cuda_error = False
    cuda_error_message = None
    batch_size = args.batch_size

    print('\n' + '=' * 80)
    print(f'Processing {sequence_name} | {camera}')
    print(f'Input:  {video_path}')
    if output_video_path is not None:
        print(f'Output: {output_video_path}')
    else:
        print('Output: not saving video')
    print(f'FPS: {fps:.2f} | Size: {width}x{height}')
    print('=' * 80)

    try:
        while True:
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break

            try:
                batch_predictions = predict_batch(model, batch_frames, args.img_size, device, args.confidence)
            except RuntimeError as exc:
                cuda_error = True
                cuda_error_message = str(exc)
                print(f'⚠ CUDA/runtime error in {sequence_name} {camera} after {total_frames} frames: {exc}')
                break

            for frame, (pose_count, first_pose) in zip(batch_frames, batch_predictions):
                total_frames += 1
                total_pose_instances += pose_count

                if pose_count > 0:
                    detected_frames += 1
                    status_text = 'DETECTED'
                else:
                    not_detected_frames += 1
                    status_text = 'NOT DETECTED'

                if writer is not None:
                    writer.write(frame)

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    summary = {
        'sequence': sequence_name,
        'folder': folder,
        'subfolder': f'{subfolder:02d}',
        'camera': camera,
        'video_path': video_path,
        'output_video_path': output_video_path,
        'total_frames': total_frames,
        'detected_frames': detected_frames,
        'not_detected_frames': not_detected_frames,
        'total_pose_instances': total_pose_instances,
        'detection_rate': float(detected_frames / total_frames) if total_frames > 0 else 0.0,
        'cuda_error': cuda_error,
        'cuda_error_message': cuda_error_message,
    }

    if cuda_error:
        print(
            f"⚠ {sequence_name} {camera}: "
            f"detected={detected_frames}, not_detected={not_detected_frames}, "
            f"pose_instances={total_pose_instances}, stopped_due_to_cuda_error=True"
        )
    else:
        print(
            f"✓ {sequence_name} {camera}: "
            f"detected={detected_frames}, not_detected={not_detected_frames}, "
            f"pose_instances={total_pose_instances}"
        )
    return summary


def print_summary(sequence_results, sequence_totals, camera_totals):
    print('\n' + '=' * 80)
    print('UCO YOLO detection summary')
    print('=' * 80)

    print(f"{'Subfolder':<12} {'Camera':<8} {'Frames':<8} {'Detected':<10} {'Missing':<10} {'Poses':<10} {'Poses/Frames':<13}")
    print('-' * 80)
    for result in sequence_results:
        print(
            f"{result['sequence']:<12} {result['camera']:<8} {result['total_frames']:<8} "
            f"{result['detected_frames']:<10} {result['not_detected_frames']:<10} "
            f"{result['total_pose_instances']:<10} {format_ratio(result['total_pose_instances'], result['total_frames']):<13}"
        )

    print('\nPer-subfolder totals across cameras')
    print(f"{'Subfolder':<12} {'Frames':<10} {'Detected':<10} {'Missing':<10} {'Poses':<10} {'Poses/Frames':<13} {'Rate':<10}")
    print('-' * 70)
    for sequence_name, totals in sequence_totals.items():
        print(
            f"{sequence_name:<12} {totals['total_frames']:<10} {totals['detected_frames']:<10} "
            f"{totals['not_detected_frames']:<10} {totals['total_pose_instances']:<10} "
            f"{format_ratio(totals['total_pose_instances'], totals['total_frames']):<13} {totals['detection_rate'] * 100:>7.2f}%"
        )

    print('\nPer-camera totals')
    print(f"{'Camera':<8} {'Frames':<10} {'Detected':<10} {'Missing':<10} {'Poses':<10} {'Poses/Frames':<13} {'Rate':<10}")
    print('-' * 70)
    for camera, totals in camera_totals.items():
        print(
            f"{camera:<8} {totals['total_frames']:<10} {totals['detected_frames']:<10} "
            f"{totals['not_detected_frames']:<10} {totals['total_pose_instances']:<10} "
            f"{format_ratio(totals['total_pose_instances'], totals['total_frames']):<13} {totals['detection_rate'] * 100:>7.2f}%"
        )

    overall_frames = sum(totals['total_frames'] for totals in camera_totals.values())
    overall_detected = sum(totals['detected_frames'] for totals in camera_totals.values())
    overall_missing = sum(totals['not_detected_frames'] for totals in camera_totals.values())
    overall_poses = sum(totals['total_pose_instances'] for totals in camera_totals.values())
    overall_rate = (overall_detected / overall_frames) if overall_frames > 0 else 0.0

    print('\nOverall totals across all cameras')
    print(f"Frames: {overall_frames}")
    print(f"Detected frames: {overall_detected}")
    print(f"Not detected frames: {overall_missing}")
    print(f"Poses: {overall_poses}")
    print(f"Poses/Frames: {format_ratio(overall_poses, overall_frames)}")
    print(f"Detection rate: {overall_rate * 100:.2f}%")
    print('=' * 80)


def save_summary_json(sequence_results, sequence_totals, camera_totals, args):
    summary_path = os.path.join(args.output_dir, 'detection_summary.json')
    os.makedirs(args.output_dir, exist_ok=True)

    payload = {
        'dataset': 'UCO Physical Rehabilitation',
        'folder_range': [0, 26],
        'subfolder_range': [9, 16],
        'cameras': args.cameras,
        'model_path': args.model_path,
        'img_size': args.img_size,
        'confidence': args.confidence,
        'sequences': sequence_results,
        'sequence_totals': sequence_totals,
        'camera_totals': camera_totals,
    }

    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)

    print(f'✓ Saved summary to {summary_path}')


def process_all_uco_sequences(model, args, device):
    sequence_results = []
    sequence_totals = {}
    camera_totals = {
        camera: {
            'total_frames': 0,
            'detected_frames': 0,
            'not_detected_frames': 0,
            'total_pose_instances': 0,
            'detection_rate': 0.0,
        }
        for camera in args.cameras
    }

    for folder in FOLDER_RANGE:
        print(f'\n--- Subject {folder:02d} start ---')
        subject_processed = 0
        for subfolder in SUBFOLDER_RANGE:
            print(f'  -> Subfolder {folder:02d}/{subfolder:02d} start')
            for camera in args.cameras:
                print(f'     -> Camera {camera} start')
                try:
                    result = process_sequence_camera(model, folder, subfolder, camera, args, device)
                except Exception as exc:
                    print(f'     ! Camera {camera} failed with error: {exc}')
                    continue

                if result is None:
                    print(f'     <- Camera {camera} skipped or failed')
                    continue

                if result.get('cuda_error'):
                    failure_location = f"{result['sequence']} | {camera}"
                    failure_message = result.get('cuda_error_message') or 'CUDA runtime error'
                    print(
                        f'\nFATAL: stopping job after first CUDA error at {failure_location}. '
                        f'Last error: {failure_message}'
                    )
                    raise RuntimeError(
                        f'CUDA failure at {failure_location}: {failure_message}'
                    )

                sequence_results.append(result)
                subject_processed += 1

                sequence_name = result['sequence']
                if sequence_name not in sequence_totals:
                    sequence_totals[sequence_name] = {
                        'total_frames': 0,
                        'detected_frames': 0,
                        'not_detected_frames': 0,
                        'total_pose_instances': 0,
                        'detection_rate': 0.0,
                    }

                sequence_aggregate = sequence_totals[sequence_name]
                sequence_aggregate['total_frames'] += result['total_frames']
                sequence_aggregate['detected_frames'] += result['detected_frames']
                sequence_aggregate['not_detected_frames'] += result['not_detected_frames']
                sequence_aggregate['total_pose_instances'] += result['total_pose_instances']

                if sequence_aggregate['total_frames'] > 0:
                    sequence_aggregate['detection_rate'] = (
                        sequence_aggregate['detected_frames'] / sequence_aggregate['total_frames']
                    )

                totals = camera_totals[camera]
                totals['total_frames'] += result['total_frames']
                totals['detected_frames'] += result['detected_frames']
                totals['not_detected_frames'] += result['not_detected_frames']
                totals['total_pose_instances'] += result['total_pose_instances']

                if totals['total_frames'] > 0:
                    totals['detection_rate'] = totals['detected_frames'] / totals['total_frames']

                gc.collect()

                print(f'     <- Camera {camera} done')

            print(f'  <- Subfolder {folder:02d}/{subfolder:02d} done')

        print(f'--- Subject {folder:02d} done ({subject_processed} sequence-camera runs) ---')

    print_summary(sequence_results, sequence_totals, camera_totals)
    save_summary_json(sequence_results, sequence_totals, camera_totals, args)


def main():
    parser = argparse.ArgumentParser(description='Process all UCO videos with YOLO and count detections')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained YOLO pose model (.pt file)')
    parser.add_argument('--output-dir', type=str, default='uco_yolo_sequence_output', help='Directory to save annotated videos and summaries')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size for YOLO inference')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for YOLO inference over frames')
    parser.add_argument('--confidence', type=float, default=0.65, help='YOLO confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--cameras', type=str, nargs='*', default=DEFAULT_CAMERAS, help='Cameras to process (default: cam0 cam1 cam2 cam3 cam4)')
    parser.add_argument('--save-videos', dest='save_videos', action='store_true', help='Save annotated mp4 videos')
    parser.add_argument('--no-save-videos', dest='save_videos', action='store_false', help='Do not save annotated mp4 videos')
    parser.set_defaults(save_videos=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error('--batch-size must be a positive integer')

    if args.confidence < 0.0 or args.confidence > 1.0:
        parser.error('--confidence must be between 0 and 1')

    print('🎯 UCO YOLO sequence processor')
    print('=' * 80)
    print('Folder range: 0-26')
    print('Subfolder range: 09-16')
    print(f'Cameras: {args.cameras}')
    print(f'Model: {args.model_path}')
    print(f'Output dir: {args.output_dir}')
    print(f'Save videos: {args.save_videos}')
    print(f'Input size: {args.img_size}')
    print(f'Batch size: {args.batch_size}')
    print(f'Confidence: {args.confidence}')
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
        process_all_uco_sequences(model, args, device)
    finally:
        gc.collect()


if __name__ == '__main__':
    main()