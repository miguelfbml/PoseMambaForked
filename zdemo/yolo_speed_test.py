"""
Simple YOLO inference speed benchmark.

Examples:
  python yolo_speed_test.py
  python yolo_speed_test.py --viz
  python yolo_speed_test.py --camera 0 --imgsz 640 --conf 0.25 --frames 300
  python yolo_speed_test.py --model weights/yolo/best.pt --warmup 30
"""

import argparse
import os
import statistics
import time

import cv2
import torch
from ultralytics import YOLO


def default_model_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "weights", "yolo", "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO inference speed test")
    parser.add_argument("--model", type=str, default=default_model_path(), help="Path to YOLO model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames (not counted)")
    parser.add_argument("--frames", type=int, default=200, help="Measured frames")
    parser.add_argument("--device", type=str, default="", help='Device (e.g. "cpu", "0"); empty = auto')
    parser.add_argument("--viz", action="store_true", help="Show live visualization")
    return parser.parse_args()


def percentile(values, q):
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_vals) - 1)
    weight = pos - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    device = args.device if args.device else ("0" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("YOLO Inference Speed Test")
    print("=" * 60)
    print(f"Model  : {args.model}")
    print(f"Camera : {args.camera}")
    print(f"Device : {device}")
    print(f"Warmup : {args.warmup}")
    print(f"Frames : {args.frames}")
    print(f"Viz    : {args.viz}")
    print("=" * 60)

    model = YOLO(args.model)
    # Try V4L2 backend first (needed for WSL camera access)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Fallback to default backend
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    # Warmup pass for more stable timing.
    warmed = 0
    while warmed < args.warmup:
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
        warmed += 1

    times_ms = []
    measured = 0

    try:
        while measured < args.frames:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed, stopping early.")
                break

            t0 = time.perf_counter()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=device, verbose=False)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt_ms)
            measured += 1

            if args.viz:
                vis = results[0].plot() if results else frame
                cv2.putText(
                    vis,
                    f"{dt_ms:.2f} ms | {1000.0 / dt_ms:.2f} FPS",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("YOLO speed test", vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if measured % 20 == 0:
                avg = statistics.mean(times_ms)
                print(f"[{measured}/{args.frames}] avg: {avg:.2f} ms ({1000.0 / avg:.2f} FPS)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not times_ms:
        print("No frames measured.")
        return

    mean_ms = statistics.mean(times_ms)
    med_ms = statistics.median(times_ms)
    p90_ms = percentile(times_ms, 0.90)
    p95_ms = percentile(times_ms, 0.95)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Measured frames: {len(times_ms)}")
    print(f"Mean latency : {mean_ms:.2f} ms")
    print(f"Median       : {med_ms:.2f} ms")
    print(f"P90          : {p90_ms:.2f} ms")
    print(f"P95          : {p95_ms:.2f} ms")
    print(f"Min / Max    : {min_ms:.2f} / {max_ms:.2f} ms")
    print(f"Mean FPS     : {1000.0 / mean_ms:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
