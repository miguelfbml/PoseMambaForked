# UCO Dataset Usage Guide

## Overview

The `compare_gt_yolo_selected_frames.py` script and `compare_selected_frames.sh` have been updated to support the **UCO Physical Rehabilitation** dataset while maintaining backward compatibility with the MPI 3DHP dataset.

## Dataset Structure

The UCO dataset is organized as follows:
```
/nas-ctm01/datasets/public/UCO Physical Rehabilitation/dataset/clips_mp4/
├── 0/                    # Folder 0
│   ├── 01/              # Subfolder 01
│   │   ├── cam0.mp4     # Video files (cam0-cam4)
│   │   ├── cam0_p2d.txt # Ground truth 2D poses
│   │   ├── cam1.mp4
│   │   ├── cam1_p2d.txt
│   │   └── ...
│   ├── 02/
│   └── ...
├── 1/
│   ├── 01/
│   └── ...
└── ... (up to folder 26, each with subfolders 01-16)
```

## Usage Examples

### Basic UCO Dataset Usage

Process selected frames from folder 0, subfolder 01, camera 0:

```bash
python3 compare_gt_yolo_selected_frames.py \
    --sequence "0/01" \
    --camera "cam0" \
    --frames 0 30 60 90 120 \
    --model-path "runs/pose/yolo_model/weights/best.pt"
```

### With Custom Output Directory

```bash
python3 compare_gt_yolo_selected_frames.py \
    --sequence "0/01" \
    --camera "cam0" \
    --frames 0 30 60 90 120 \
    --model-path "runs/pose/yolo_model/weights/best.pt" \
    --output-dir "uco_comparisons"
```

### Different Camera Angles

Process the same folder/subfolder with different cameras:

```bash
# Camera 0 (frontal)
python3 compare_gt_yolo_selected_frames.py --sequence "0/01" --camera "cam0" --frames 0 30 60 --model-path "model.pt"

# Camera 2 (side)
python3 compare_gt_yolo_selected_frames.py --sequence "0/01" --camera "cam2" --frames 0 30 60 --model-path "model.pt"

# Camera 4 (back)
python3 compare_gt_yolo_selected_frames.py --sequence "0/01" --camera "cam4" --frames 0 30 60 --model-path "model.pt"
```

### Multiple Folders/Subfolders

```bash
# Folder 1, subfolder 05, camera 1
python3 compare_gt_yolo_selected_frames.py --sequence "1/05" --camera "cam1" --frames 0 30 60 --model-path "model.pt"

# Folder 10, subfolder 16, camera 3
python3 compare_gt_yolo_selected_frames.py --sequence "10/16" --camera "cam3" --frames 0 30 60 --model-path "model.pt"
```

## Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--sequence` | Yes | Format: `folder/subfolder` for UCO, or `TS1-TS6` for MPI 3DHP | `0/01` or `TS1` |
| `--frames` | Yes | Space-separated frame indices (0-based) | `0 30 60 90` |
| `--model-path` | Yes | Path to YOLO .pt model | `runs/pose/model/weights/best.pt` |
| `--camera` | No | Camera to use (UCO only): cam0-cam4 | `cam0` (default) |
| `--output-dir` | No | Output directory for comparison images | `comparison_selected_frames` (default) |
| `--img-size` | No | YOLO input image size | `640` (default) |
| `--batch-size` | No | Inference batch size | Auto (32 CUDA, 16 CPU) |
| `--device` | No | Device: auto, cpu, cuda, cuda:0 | `auto` (default) |

## Batch Processing with Bash Script

Edit `compare_selected_frames.sh` to add your UCO dataset sequences:

```bash
python3 compare_gt_yolo_selected_frames.py \
    --sequence "0/01" \
    --camera "cam0" \
    --frames 0 30 60 90 120 150 180 210 240 270 \
    --model-path "runs/pose/yolo_model/weights/best.pt" \
    --output-dir "comparison_selected_frames_uco" \
    --img-size "640" \
    --batch-size "32" \
    --device "cuda:0"
```

Then submit with:
```bash
sbatch compare_selected_frames.sh
```

## Output

For each frame processed, the script generates:
- Side-by-side comparison images with GT vs YOLO predictions
- Frame-level MPJPE (Mean Per-Joint Position Error)
- Performance metrics (inference time, FPS)
- Output saved in: `{output-dir}/{sequence}/frame_{frame_idx:06d}_gt_vs_yolo.png`

Example output structure:
```
comparison_selected_frames_uco/
├── 0/01/
│   ├── frame_000000_gt_vs_yolo.png
│   ├── frame_000030_gt_vs_yolo.png
│   └── ...
└── 0/02/
    └── ...
```

## Important Notes

1. **Folder/Subfolder Format**: Always use `folder/subfolder` (e.g., `0/01`, not `0_01`)
2. **Camera Selection**: Default is `cam0`. Use `--camera` to select others (cam1-cam4)
3. **Ground Truth Files**: Assumes `cam{0-4}_p2d.txt` files exist in the dataset
4. **Frame Indices**: 0-based indexing. Check video duration to determine valid frame ranges
5. **Backward Compatibility**: MPI 3DHP sequences (TS1-TS6) still work with the old format

## Dataset Auto-Detection

The script automatically detects the dataset format:
- If sequence contains `/` → UCO dataset mode (loads from MP4, reads _p2d.txt)
- Otherwise → MPI 3DHP mode (legacy behavior)

## Troubleshooting

### Ground truth file not found
```
❌ Ground truth file not found: /nas-ctm01/datasets/.../0/01/cam0_p2d.txt
```
**Solution**: Check that the folder/subfolder path exists and contains the `cam*_p2d.txt` files.

### Video file not found
```
❌ Video file not found: /nas-ctm01/datasets/.../0/01/cam0.mp4
```
**Solution**: Verify the video file exists at the specified path with correct camera number.

### Frames out of range
```
⚠ Frame 300 out of range (0-299)
```
**Solution**: The video has fewer frames than requested. Use valid frame indices for the specific video.
