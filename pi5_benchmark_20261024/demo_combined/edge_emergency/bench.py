#!/usr/bin/env python
"""Edge-pipeline benchmark harness.

Times each stage of a {detector, pose, recognizer} pipeline over a real video
(or synthetic frames) and reports per-stage latency (mean/p50/p95) plus the
effective end-to-end FPS the live demo would actually achieve given the
detection/recognition skip cadence.

Written to run UNCHANGED on a laptop x86 CPU now and on a Raspberry Pi 5 ARM
CPU later — so the same numbers are directly comparable across devices.

Pose backends (swap the detect+pose front-end, keep the COCO-17 skeleton so the
trained STGCN++ medical model still works downstream):
    rtmpose  : YOLOX-tiny detector  -> RTMPose top-down (the current baseline)
    movenet  : MoveNet SinglePose (TFLite, single-person, NO separate detector)

Examples
--------
# Baseline pipeline, Pi-like settings, 200 frames of the sample clip:
python demo_combined/edge_emergency/bench.py \
    --video demo/ntu_sample.avi --frames 200 \
    --pose-backend rtmpose --short-side 256 --threads 4

# MoveNet single-stage front-end (after fetch_movenet.py downloads the model):
python demo_combined/edge_emergency/bench.py \
    --video demo/ntu_sample.avi --frames 200 \
    --pose-backend movenet --movenet-model demo/onnx_models/movenet_lightning_int8.tflite

# A/B several configs and print one comparison table:
python demo_combined/edge_emergency/bench.py --suite --video demo/ntu_sample.avi --frames 150
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# --- Reuse the proven pure-ONNX pipeline classes from demo/demo_onnx.py -------
_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMO_DIR = _ROOT / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import demo_onnx  # noqa: E402  (YOLOXDetector, RTMPoseEstimator, STGCNRecognizer)

DEFAULT_MODEL_DIR = _ROOT / "demo" / "onnx_models"
DEFAULT_LABEL_MAP = _ROOT / "tools" / "data" / "label_map" / "medical_15.txt"


# ---------------------------------------------------------------------------
#  COCO-17 keypoint order (shared contract between every pose backend and
#  the STGCN++ recognizer). MoveNet emits this exact ordering.
# ---------------------------------------------------------------------------
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
#  MoveNet SinglePose pose backend (TFLite via ai-edge-litert)
# ---------------------------------------------------------------------------
class MoveNetEstimator:
    """MoveNet SinglePose (Lightning/Thunder) TFLite wrapper.

    Presents the SAME call signature as demo_onnx.RTMPoseEstimator:
        __call__(img, bboxes) -> (keypoints (N,17,2) px, scores (N,17))

    It ignores incoming bboxes (MoveNet does its own single-person localisation
    on the full frame), and converts MoveNet's native (y, x, score) in [0,1] to
    pixel-space (x, y) so downstream code is identical to the RTMPose path.

    Input size is read from the model: Lightning=192, Thunder=256 (square).
    """

    def __init__(self, model_path, device="cpu", threads=0):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as e:
            raise ImportError(
                "MoveNet backend needs ai-edge-litert: "
                "uv pip install --python .venv/Scripts/python.exe ai-edge-litert"
            ) from e
        n_threads = threads if threads and threads > 0 else None
        self.interp = Interpreter(model_path=str(model_path), num_threads=n_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_size = int(self.inp["shape"][1])  # square H==W
        # int8/uint8 models want uint8 input; float models want float32
        self.in_dtype = self.inp["dtype"]
        print(f"  MoveNet: {Path(model_path).name} in={self.in_size}px "
              f"dtype={np.dtype(self.in_dtype).name} threads={n_threads or 'auto'}")

    def _preprocess(self, img):
        h, w = img.shape[:2]
        # letterbox to square, keep aspect (MoveNet expects square input)
        scale = self.in_size / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.in_size, self.in_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = canvas[None]  # (1,H,W,3) — MoveNet is NHWC
        if self.in_dtype == np.float32:
            blob = blob.astype(np.float32)  # MoveNet float models take [0,255] floats
        return blob, scale, nw, nh

    def __call__(self, img, bboxes=None):
        h, w = img.shape[:2]
        blob, scale, nw, nh = self._preprocess(img)
        self.interp.set_tensor(self.inp["index"], blob)
        self.interp.invoke()
        kpts = self.interp.get_tensor(self.out["index"])  # (1,1,17,3) = (y,x,score) norm
        kpts = kpts[0, 0]  # (17,3)
        ys = kpts[:, 0] * self.in_size / scale
        xs = kpts[:, 1] * self.in_size / scale
        scores = kpts[:, 2].astype(np.float32)
        xy = np.stack([xs, ys], axis=-1).astype(np.float32)  # (17,2) pixel (x,y)
        return xy[None], scores[None]  # (1,17,2), (1,17)


# ---------------------------------------------------------------------------
#  Timing helpers
# ---------------------------------------------------------------------------
def _stats(ms_list):
    if not ms_list:
        return dict(n=0, mean=0.0, p50=0.0, p95=0.0)
    a = np.asarray(ms_list, dtype=np.float64)
    return dict(
        n=len(a),
        mean=float(a.mean()),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
    )


def load_frames(video, n, short_side):
    """Read up to n frames, resized so the short side == short_side (0=no resize)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    frames = []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            break
        if short_side > 0:
            h, w = f.shape[:2]
            s = short_side / min(h, w)
            f = cv2.resize(f, (int(round(w * s)), int(round(h * s))))
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"No frames read from {video}")
    return frames


# ---------------------------------------------------------------------------
#  One benchmark run
# ---------------------------------------------------------------------------
def run_one(cfg, frames):
    """Run a single pipeline config over the given frames. Returns a result dict."""
    model_dir = Path(cfg["model_dir"])
    threads = cfg["threads"]
    backend = cfg["pose_backend"]

    # --- Build front-end (detector + pose) ---
    detector = None
    if backend == "rtmpose":
        detector = demo_onnx.YOLOXDetector(
            str(model_dir / cfg["det_model"]), device="cpu",
            score_thr=cfg["det_score_thr"], threads=threads,
        )
        pose = demo_onnx.RTMPoseEstimator(
            str(model_dir / cfg["pose_model"]), device="cpu", threads=threads,
        )
    elif backend == "movenet":
        pose = MoveNetEstimator(cfg["movenet_model"], threads=threads)
    else:
        raise SystemExit(f"Unknown pose backend: {backend}")

    # --- Recognizer (shared COCO-17 STGCN++) ---
    recog = demo_onnx.STGCNRecognizer(
        str(model_dir / cfg["recog_model"]), str(cfg["label_map"]),
        device="cpu", clip_len=cfg["clip_len"], num_person=cfg["num_person"],
        threads=threads,
    )

    h, w = frames[0].shape[:2]
    det_every, recog_every = cfg["det_every"], cfg["recog_every"]
    win = cfg["window_frames"]

    from collections import deque
    kbuf, sbuf = deque(maxlen=win), deque(maxlen=win)

    # --- Warmup ---
    dummy = frames[0]
    for _ in range(3):
        if detector is not None:
            b = detector(dummy)
            pose(dummy, b if b else [[0, 0, w, h]])
        else:
            pose(dummy)

    det_ms, pose_ms, recog_ms, frame_ms = [], [], [], []
    cached_boxes, last_result = [], None
    t_wall0 = time.perf_counter()

    for i, frame in enumerate(frames, 1):
        t_f = time.perf_counter()

        # detection (skip cadence; movenet has no detector)
        t0 = time.perf_counter()
        if detector is not None:
            if i % det_every == 1 or det_every == 1:
                cached_boxes = detector(frame)
            boxes = cached_boxes
            det_ms.append((time.perf_counter() - t0) * 1000)
        else:
            boxes = None

        # pose
        t0 = time.perf_counter()
        if detector is not None and len(boxes) == 0:
            kpts, scores = np.zeros((0, 17, 2)), np.zeros((0, 17))
        else:
            kpts, scores = pose(frame, boxes) if detector is not None else pose(frame)
        pose_ms.append((time.perf_counter() - t0) * 1000)

        kbuf.append(kpts)
        sbuf.append(scores)

        # recognition (skip cadence)
        t0 = time.perf_counter()
        if len(kbuf) >= cfg["min_frames"] and (i % recog_every == 0 or recog_every == 1):
            last_result = recog(list(kbuf), list(sbuf), img_shape=(h, w))
            recog_ms.append((time.perf_counter() - t0) * 1000)

        frame_ms.append((time.perf_counter() - t_f) * 1000)

    wall = time.perf_counter() - t_wall0
    eff_fps = len(frames) / wall

    return {
        "cfg": cfg,
        "proc_res": f"{w}x{h}",
        "det": _stats(det_ms),
        "pose": _stats(pose_ms),
        "recog": _stats(recog_ms),
        "frame": _stats(frame_ms),
        "eff_fps": eff_fps,
        "n_frames": len(frames),
        "last_top1": (last_result[0][1], last_result[0][2]) if last_result else None,
    }


def print_result(r):
    c = r["cfg"]
    tag = c.get("tag", c["pose_backend"])
    print(f"\n{'='*74}")
    print(f"  CONFIG: {tag}  |  proc {r['proc_res']}  |  threads={c['threads']}  "
          f"|  det-every={c['det_every']} recog-every={c['recog_every']} "
          f"persons={c['num_person']}")
    print(f"  {'-'*70}")
    print(f"  {'stage':<10}{'mean ms':>10}{'p50':>9}{'p95':>9}{'≈FPS':>9}   calls")
    for k in ("det", "pose", "recog", "frame"):
        s = r[k]
        if s["n"] == 0:
            print(f"  {k:<10}{'—':>10}{'—':>9}{'—':>9}{'—':>9}   0")
            continue
        fps = 1000.0 / s["mean"] if s["mean"] > 0 else 0
        print(f"  {k:<10}{s['mean']:>10.1f}{s['p50']:>9.1f}{s['p95']:>9.1f}{fps:>9.1f}   {s['n']}")
    print(f"  {'-'*70}")
    print(f"  EFFECTIVE PIPELINE FPS (end-to-end, with skip cadence): {r['eff_fps']:.1f}")
    if r["last_top1"]:
        print(f"  last prediction: {r['last_top1'][0]} ({r['last_top1'][1]:.0%})")


def base_cfg(args):
    return dict(
        model_dir=args.model_dir, label_map=args.label_map,
        det_model=args.det_model, pose_model=args.pose_model,
        recog_model=args.recog_model, movenet_model=args.movenet_model,
        pose_backend=args.pose_backend, threads=args.threads,
        short_side=args.short_side, det_score_thr=args.det_score_thr,
        det_every=args.det_every, recog_every=args.recog_every,
        window_frames=args.window_frames, clip_len=args.clip_len,
        min_frames=args.min_frames, num_person=args.num_person,
    )


def main():
    p = argparse.ArgumentParser(description="Edge pipeline benchmark harness")
    p.add_argument("--video", default=str(_ROOT / "demo" / "ntu_sample.avi"))
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--label-map", default=str(DEFAULT_LABEL_MAP))
    p.add_argument("--det-model", default="yolox_tiny.onnx")
    p.add_argument("--pose-model", default="rtmpose_m.onnx")
    p.add_argument("--recog-model", default="stgcnpp_medical15_cent2d.onnx")
    p.add_argument("--movenet-model", default="")
    p.add_argument("--pose-backend", choices=["rtmpose", "movenet"], default="rtmpose")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--short-side", type=int, default=256)
    p.add_argument("--det-score-thr", type=float, default=0.5)
    p.add_argument("--det-every", type=int, default=1)
    p.add_argument("--recog-every", type=int, default=1)
    p.add_argument("--window-frames", type=int, default=30)
    p.add_argument("--clip-len", type=int, default=100)
    p.add_argument("--min-frames", type=int, default=8)
    p.add_argument("--num-person", type=int, default=1)
    p.add_argument("--suite", action="store_true",
                   help="Run a preset A/B sweep of configs and print a comparison.")
    args = p.parse_args()

    frames = load_frames(args.video, args.frames, args.short_side)
    print(f"Loaded {len(frames)} frames from {Path(args.video).name} "
          f"(short-side={args.short_side} → {frames[0].shape[1]}x{frames[0].shape[0]})")

    if not args.suite:
        cfg = base_cfg(args)
        cfg["tag"] = args.pose_backend
        print_result(run_one(cfg, frames))
        return

    # --- Preset sweep: vary skip cadence on the baseline (device-independent shape) ---
    results = []
    sweeps = [
        ("rtmpose det1/recog1", dict(det_every=1, recog_every=1)),
        ("rtmpose det2/recog4", dict(det_every=2, recog_every=4)),
        ("rtmpose det3/recog6", dict(det_every=3, recog_every=6)),
    ]
    for tag, over in sweeps:
        cfg = base_cfg(args)
        cfg.update(over)
        cfg["tag"] = tag
        try:
            results.append(run_one(cfg, frames))
        except Exception as e:
            print(f"  [skip] {tag}: {e}")
    for r in results:
        print_result(r)

    print(f"\n{'='*74}\n  SUITE SUMMARY (effective end-to-end FPS)\n  {'-'*70}")
    for r in results:
        print(f"  {r['cfg']['tag']:<28} {r['eff_fps']:>6.1f} FPS   "
              f"det {r['det']['mean']:>5.1f}  pose {r['pose']['mean']:>5.1f}  "
              f"recog {r['recog']['mean']:>5.1f} ms")


if __name__ == "__main__":
    main()
