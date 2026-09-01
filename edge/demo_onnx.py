#!/usr/bin/env python
"""Pure-ONNX real-time skeleton-based action recognition demo.

This script requires ONLY: numpy, opencv-python, onnxruntime(-gpu).
No mmcv, mmpose, mmdet, or pyskl imports at runtime.

Pipeline:
    webcam frame  →  YOLOX-Tiny (person detection)
                  →  RTMPose-m  (17-kp COCO pose estimation)
                  →  STGCN++    (skeleton action recognition, 120 NTU classes)

Models (all in demo/onnx_models/):
    yolox_tiny.onnx                   – 19 MB, 416×416, built-in NMS
    rtmpose_m.onnx                    – 52 MB, 192×256, SimCC output
    stgcnpp_ntu120_xsub_hrnet_j.onnx –  5 MB, (N,2,100,17,3)

Usage:
    # Stream mode (default) — continuous webcam with live action label:
    python demo/demo_onnx.py --device cuda
    python demo/demo_onnx.py --device cpu --short-side 320

    # Record the annotated output to a video file:
    python demo/demo_onnx.py --device cuda --output demo_recording.mp4
    python demo/demo_onnx.py --clip input.mp4 --output output.mp4

    # Clip mode — press 'r' to record, then auto-infer:
    python demo/demo_onnx.py --mode clip --record-seconds 3.0 --output clips.mp4

    # Benchmark individual stages:
    python demo/demo_onnx.py --benchmark

Author: Research demo for advisor meeting
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path


# Auto-add nvidia cuDNN/cublas DLLs to PATH (for onnxruntime CUDA EP)
def _setup_nvidia_path():
    try:
        import nvidia.cudnn

        cudnn_bin = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin")
        if os.path.isdir(cudnn_bin) and cudnn_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cudnn_bin + os.pathsep + os.environ.get("PATH", "")
    except (ImportError, Exception):
        pass
    try:
        import importlib

        spec = importlib.util.find_spec("nvidia.cublas")
        if spec and spec.submodule_search_locations:
            cublas_bin = os.path.join(spec.submodule_search_locations[0], "bin")
            if os.path.isdir(cublas_bin) and cublas_bin not in os.environ.get(
                "PATH", ""
            ):
                os.environ["PATH"] = (
                    cublas_bin + os.pathsep + os.environ.get("PATH", "")
                )
    except (ImportError, Exception):
        pass


_setup_nvidia_path()

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  ONNX Runtime session helper
# ---------------------------------------------------------------------------


def create_session(onnx_path, device="cuda", threads=0):
    """Create ONNX Runtime InferenceSession with preferred provider.

    Args:
        threads: Number of intra-op threads (0 = ORT default, usually = cores).
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = max(1, threads // 2)
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
    actual = sess.get_providers()
    name = Path(onnx_path).name
    thr_info = f", threads={threads}" if threads > 0 else ""
    print(f"  {name}: providers={actual}{thr_info}")
    if device == "cuda" and "CUDAExecutionProvider" not in actual:
        print(f"  ⚠ WARNING: CUDA requested but NOT available for {name}!")
        print(f"    Falling back to CPU — this will be SLOW.")
        print(f"    Install onnxruntime-gpu for CUDA support.")
    return sess


# ---------------------------------------------------------------------------
#  YOLOX-Tiny  (person detection with built-in NMS)
# ---------------------------------------------------------------------------


class YOLOXDetector:
    """YOLOX-Tiny ONNX detector.

    The HumanArt version includes built-in NMS, so output is:
        dets:   (1, N, 5)  = [x1, y1, x2, y2, score]
        labels: (1, N)     = class IDs (0 = person)
    """

    def __init__(self, onnx_path, device="cuda", input_size=(416, 416), score_thr=0.5, threads=0):
        self.session = create_session(onnx_path, device, threads=threads)
        self.input_size = input_size  # (H, W)
        self.score_thr = score_thr
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        """Letterbox pad + float32, NO mean/std normalization."""
        h, w = img.shape[:2]
        target_h, target_w = self.input_size
        ratio = min(target_h / h, target_w / w)

        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # HWC → CHW, float32
        blob = padded.transpose(2, 0, 1).astype(np.float32)[None]
        return blob, ratio

    def __call__(self, img):
        """Return list of [x1,y1,x2,y2] in original image coords."""
        blob, ratio = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        dets = outputs[0][0]  # (N, 5)
        labels = outputs[1][0]  # (N,)

        # Filter: person class (0) and score
        mask = (labels == 0) & (dets[:, 4] > self.score_thr)
        boxes = dets[mask, :4] / ratio  # rescale to original image
        return boxes.tolist()


# ---------------------------------------------------------------------------
#  RTMPose-m  (top-down 17-keypoint pose estimator, SimCC)
# ---------------------------------------------------------------------------


class RTMPoseEstimator:
    """RTMPose ONNX top-down pose estimator with SimCC decoding.

    Outputs per person: keypoints (17, 2) in image coords, scores (17,).
    """

    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def __init__(self, onnx_path, device="cuda", input_size=(192, 256), threads=0):
        self.session = create_session(onnx_path, device, threads=threads)
        self.input_size = input_size  # (W, H)
        self.input_name = self.session.get_inputs()[0].name
        self.simcc_split_ratio = 2.0
        # Pre-compute for batch postprocess
        self._inv_simcc = 1.0 / self.simcc_split_ratio
        self._input_size_arr = np.array(self.input_size, dtype=np.float32)  # (W, H)

    # ---- Preprocessing helpers (ported from rtmlib) ----

    @staticmethod
    def _bbox_xyxy2cs(bbox, padding=1.25):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        w *= padding
        h *= padding
        center = np.array([cx, cy], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    @staticmethod
    def _get_warp_matrix(center, scale, rot, output_size):
        """Get affine transform matrix (simplified, rot=0)."""
        src_w, src_h = scale
        dst_w, dst_h = output_size

        src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
        dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

        src_p1 = center
        src_p2 = center + src_dir
        src_p3 = np.array(
            [src_p1[0] - src_dir[1], src_p1[1] + src_dir[0]], dtype=np.float32
        )

        dst_p1 = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_p2 = dst_p1 + dst_dir
        dst_p3 = np.array(
            [dst_p1[0] - dst_dir[1], dst_p1[1] + dst_dir[0]], dtype=np.float32
        )

        src = np.stack([src_p1, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_p1, dst_p2, dst_p3]).astype(np.float32)
        return cv2.getAffineTransform(src, dst)

    def preprocess(self, img, bbox):
        """Affine-crop person ROI, normalize, return blob + metadata."""
        center, scale = self._bbox_xyxy2cs(bbox, padding=1.25)

        w, h = self.input_size
        # Enforce aspect ratio
        aspect_ratio = w / h
        if scale[0] / scale[1] > aspect_ratio:
            scale[1] = scale[0] / aspect_ratio
        else:
            scale[0] = scale[1] * aspect_ratio

        warp_mat = self._get_warp_matrix(center, scale, 0, (w, h))
        warped = cv2.warpAffine(img, warp_mat, (w, h), flags=cv2.INTER_LINEAR)

        # Normalize
        warped = warped.astype(np.float32)
        warped = (warped - self.MEAN) / self.STD
        blob = warped.transpose(2, 0, 1)[None]  # (1,3,H,W)
        return blob, center, scale

    @staticmethod
    def _get_simcc_maximum(simcc_x, simcc_y):
        """Decode SimCC outputs to keypoint locs and scores."""
        N, K, _ = simcc_x.shape
        simcc_x_flat = simcc_x.reshape(N * K, -1)
        simcc_y_flat = simcc_y.reshape(N * K, -1)

        x_locs = np.argmax(simcc_x_flat, axis=1)
        y_locs = np.argmax(simcc_y_flat, axis=1)
        max_val_x = np.max(simcc_x_flat, axis=1)
        max_val_y = np.max(simcc_y_flat, axis=1)

        locs = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)
        vals = 0.5 * (max_val_x + max_val_y)
        locs[vals <= 0.0] = -1

        return locs.reshape(N, K, 2), vals.reshape(N, K)

    def postprocess(self, outputs, center, scale):
        """Decode SimCC → image-space keypoints (single person)."""
        simcc_x, simcc_y = outputs
        locs, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs * self._inv_simcc

        keypoints = keypoints / self._input_size_arr * scale
        keypoints = keypoints + center - scale / 2.0

        return keypoints[0], scores[0]  # (17,2), (17,)

    def postprocess_batch(self, simcc_x, simcc_y, centers, scales):
        """Decode SimCC → image-space keypoints for a batch of persons.

        Args:
            simcc_x: (B, 17, Wx)  batched SimCC x-heatmaps
            simcc_y: (B, 17, Wy)  batched SimCC y-heatmaps
            centers: (B, 2)  crop centers
            scales:  (B, 2)  crop scales

        Returns:
            keypoints: (B, 17, 2)  image-space coords
            scores:    (B, 17)
        """
        locs, scores = self._get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs * self._inv_simcc  # (B, 17, 2)

        # Vectorized rescale: each person has its own center/scale
        # centers (B,2), scales (B,2) → broadcast with (B,17,2)
        inv_input = 1.0 / self._input_size_arr  # (2,)
        keypoints = keypoints * inv_input * scales[:, None, :]  # (B,17,2)
        keypoints = keypoints + centers[:, None, :] - scales[:, None, :] / 2.0

        return keypoints, scores

    def __call__(self, img, bboxes):
        """Run pose on all person bboxes. Return (N,17,2), (N,17).

        When multiple persons are detected, preprocesses all crops and runs
        a single batched ONNX inference call for better throughput.
        """
        if len(bboxes) == 0:
            return np.zeros((0, 17, 2)), np.zeros((0, 17))

        n = len(bboxes)

        # --- Preprocess all person crops ---
        blobs = []
        centers = np.empty((n, 2), dtype=np.float32)
        scales_arr = np.empty((n, 2), dtype=np.float32)

        for i, bbox in enumerate(bboxes):
            blob, center, scale = self.preprocess(img, bbox)
            blobs.append(blob[0])  # (3, H, W)
            centers[i] = center
            scales_arr[i] = scale

        # --- Batched inference ---
        batch_blob = np.stack(blobs, axis=0)  # (N, 3, H, W)
        outputs = self.session.run(None, {self.input_name: batch_blob})
        simcc_x, simcc_y = outputs  # (N, 17, Wx), (N, 17, Wy)

        # --- Batched postprocess ---
        keypoints, scores = self.postprocess_batch(
            simcc_x, simcc_y, centers, scales_arr
        )
        return keypoints, scores


# ---------------------------------------------------------------------------
#  STGCN++ action recognizer  (skeleton ONNX)
# ---------------------------------------------------------------------------


class STGCNRecognizer:
    """STGCN++ ONNX recognizer.

    Input:  (1, 2, T, 17, 3) float32   T=100, C=(x_norm, y_norm, score)
    Output: (1, 120) logits
    """

    def __init__(
        self,
        onnx_path,
        label_map_path,
        device="cuda",
        clip_len=100,
        num_person=2,
        img_shape=(1080, 1920),
        threads=0,
    ):
        self.session = create_session(onnx_path, device, threads=threads)
        self.input_name = self.session.get_inputs()[0].name
        self.clip_len = clip_len
        self.num_person = num_person
        self.img_h, self.img_w = img_shape

        # Pre-allocate reusable skeleton buffer (avoids per-call allocation)
        self._skeleton_buf = np.zeros(
            (num_person, clip_len, 17, 3), dtype=np.float32
        )

        # Load label map
        with open(label_map_path) as f:
            self.labels = [l.strip() for l in f.readlines()]
        print(f"  {len(self.labels)} action classes loaded.")

    def build_input(self, keypoints_buffer, scores_buffer, img_shape=None):
        """Build STGCN++ input tensor from keypoint buffers (fully vectorized).

        Eliminates the Python for-loop over time steps by gathering all
        sampled frames into padded arrays and processing them in bulk.

        Args:
            keypoints_buffer: list of (N_persons, 17, 2) arrays  (pixel coords)
            scores_buffer:    list of (N_persons, 17) arrays
            img_shape: (h, w) of the source frames

        Returns:
            np.ndarray of shape (1, 2, T, 17, 3) ready for inference
        """
        if img_shape is not None:
            h, w = img_shape
        else:
            h, w = self.img_h, self.img_w

        T = self.clip_len
        M = self.num_person

        n_frames = len(keypoints_buffer)
        if n_frames == 0:
            return np.zeros((1, M, T, 17, 3), dtype=np.float32)

        indices = np.linspace(0, n_frames - 1, T).astype(int)

        # Reuse pre-allocated buffer (zero it out)
        skeleton = self._skeleton_buf  # (M, T, 17, 3)
        skeleton[:] = 0.0

        inv_half_w = 2.0 / w
        inv_half_h = 2.0 / h
        half_w = w / 2.0
        half_h = h / 2.0

        # Group consecutive indices that map to the same source frame
        # to avoid redundant work, but the main win is vectorized math.
        for t_out, t_in in enumerate(indices):
            kpts = keypoints_buffer[t_in]  # (N, 17, 2)
            scores = scores_buffer[t_in]   # (N, 17)
            n_persons = min(kpts.shape[0], M)
            if n_persons == 0:
                continue

            kp = kpts[:n_persons]           # (P, 17, 2)
            sc = scores[:n_persons].copy()  # (P, 17)

            x_norm = (kp[:, :, 0] - half_w) * inv_half_w  # mult faster than div
            y_norm = (kp[:, :, 1] - half_h) * inv_half_h

            low = sc < 0.01
            x_norm[low] = 0.0
            y_norm[low] = 0.0
            sc[low] = 0.0

            skeleton[:n_persons, t_out, :, 0] = x_norm
            skeleton[:n_persons, t_out, :, 1] = y_norm
            skeleton[:n_persons, t_out, :, 2] = sc

        return skeleton[None]  # (1, M, T, V, C) — view of pre-allocated buf

    def __call__(self, keypoints_buffer, scores_buffer, img_shape=None):
        """Recognize action from skeleton buffer.

        Returns: list of (class_idx, label, probability) sorted by prob.
        """
        inp = self.build_input(keypoints_buffer, scores_buffer, img_shape)
        logits = self.session.run(None, {self.input_name: inp})[0][0]  # (120,)

        # Fast top-5 via argpartition (O(n) vs O(n log n) for full sort)
        top5_unsorted = np.argpartition(logits, -5)[-5:]
        top5_sorted = top5_unsorted[np.argsort(logits[top5_unsorted])[::-1]]

        # Softmax only over top-5 for display (avoid exp over all 120)
        top_logits = logits[top5_sorted]
        top_logits = top_logits - top_logits[0]  # shift for numerical stability
        exp_top = np.exp(top_logits)
        # Full softmax denominator (needed for true probabilities)
        exp_all = np.exp(logits - logits[top5_sorted[0]])
        denom = exp_all.sum()
        probs = exp_top / denom

        results = [(int(i), self.labels[i], float(probs[j]))
                   for j, i in enumerate(top5_sorted)]
        return results


# ---------------------------------------------------------------------------
#  Skeleton drawing utilities
# ---------------------------------------------------------------------------

COCO_SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),  # legs
    (11, 12),  # hip
    (5, 11),
    (6, 12),  # torso
    (5, 6),  # shoulders
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),  # arms
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
]

PALETTE = [
    (255, 128, 0),
    (255, 153, 51),
    (255, 178, 102),
    (230, 230, 0),
    (255, 153, 255),
    (153, 204, 255),
    (255, 102, 255),
    (255, 51, 255),
    (102, 178, 255),
    (51, 153, 255),
    (255, 153, 153),
    (255, 102, 102),
    (255, 51, 51),
    (153, 255, 153),
    (102, 255, 102),
    (51, 255, 51),
    (0, 255, 0),
]


def draw_skeleton(frame, keypoints, scores, kpt_thr=0.3):
    """Draw skeleton overlay on frame.

    Args:
        keypoints: (N, 17, 2) or (17, 2) in pixel coordinates.
        scores: (N, 17) or (17,)
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
        scores = scores[None]

    for person_kpts, person_scores in zip(keypoints, scores):
        # Draw limbs
        for i, j in COCO_SKELETON:
            if person_scores[i] > kpt_thr and person_scores[j] > kpt_thr:
                pt1 = tuple(person_kpts[i].astype(int))
                pt2 = tuple(person_kpts[j].astype(int))
                color = PALETTE[i % len(PALETTE)]
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoints
        for k, (pt, s) in enumerate(zip(person_kpts, person_scores)):
            if s > kpt_thr:
                x, y = int(pt[0]), int(pt[1])
                color = PALETTE[k % len(PALETTE)]
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, results, fps, timings, frame_count):
    """Draw heads-up display with action label, FPS, and timing info."""
    h, w = frame.shape[:2]

    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Pipeline timings
    timing_str = " | ".join(f"{k}: {v:.1f}ms" for k, v in timings.items())
    cv2.putText(
        frame, timing_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    # Frame count
    cv2.putText(
        frame,
        f"Frames: {frame_count}",
        (w - 160, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    # Top-1 action
    if results:
        idx, label, prob = results[0]
        text = f"{label} ({prob:.1%})"
        cv2.putText(
            frame, text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        # Top-5 sidebar
        for i, (idx, label, prob) in enumerate(results[:5]):
            y = 140 + i * 25
            bar_w = int(prob * 200)
            cv2.rectangle(frame, (10, y - 12), (10 + bar_w, y + 5), (0, 200, 255), -1)
            cv2.putText(
                frame,
                f"{label[:25]} {prob:.1%}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    return frame


# ---------------------------------------------------------------------------
#  Main streaming loop
# ---------------------------------------------------------------------------


def run_stream(args):
    """Real-time webcam streaming with ONNX pipeline."""
    print("\n=== ONNX Real-Time Action Recognition ===")
    print(f"Device: {args.device}")

    # Load models
    model_dir = Path(args.model_dir)
    det_path = str(model_dir / args.det_model)
    pose_path = str(model_dir / args.pose_model)
    recog_path = str(model_dir / args.recog_model)

    recog_device = args.recog_device or args.device
    print("\nLoading models...")
    threads = getattr(args, 'threads', 0)
    detector = YOLOXDetector(det_path, device=args.device, score_thr=args.det_score_thr, threads=threads)
    pose_estimator = RTMPoseEstimator(pose_path, device=args.device, threads=threads)
    recognizer = STGCNRecognizer(
        recog_path,
        args.label_map,
        device=recog_device,
        clip_len=args.clip_len,
        num_person=args.num_person,
        threads=threads,
    )

    # Open camera/video
    source = args.clip if args.clip else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"\nSource: {source} ({src_w}×{src_h} @ {src_fps:.0f}fps)")

    # Processing resolution
    if args.short_side > 0:
        scale = args.short_side / min(src_h, src_w)
        proc_w, proc_h = int(src_w * scale), int(src_h * scale)
    else:
        proc_w, proc_h = src_w, src_h
    print(f"Processing at: {proc_w}×{proc_h}")

    # Video writer for recording
    video_writer = None
    if args.output:
        out_fps = (
            args.output_fps if args.output_fps > 0 else (src_fps if args.clip else 25)
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (proc_w, proc_h))
        if video_writer.isOpened():
            print(f"Recording to: {args.output} ({proc_w}×{proc_h} @ {out_fps:.0f}fps)")
        else:
            print(f"WARNING: Could not open video writer for {args.output}")
            video_writer = None

    # Buffers
    window = args.window_frames
    kpts_buffer = deque(maxlen=window)
    scores_buffer = deque(maxlen=window)

    # FPS tracking
    fps_window = deque(maxlen=30)
    frame_count = 0
    results = []

    # Detection frequency (skip frames for speed)
    det_every = args.det_every

    cached_bboxes = []
    recog_every = args.recog_every

    # Rolling averages for terminal output
    avg_timings = {"det": 0, "pose": 0, "recog": 0, "total": 0}
    PRINT_EVERY = 30  # print stats every N frames

    print(f"\nWindow: {window} frames | Clip len: {args.clip_len}")
    print(f"Det every: {det_every} frames | Recog every: {recog_every} frames")
    print(f"Det score: {args.det_score_thr}")
    print("Press Q to quit.")
    print(
        f"\n{'Frame':>6} {'Det':>8} {'Pose':>8} {'Recog':>8} {'Total':>8} {'FPS':>7}  {'Action'}"
    )
    print("-" * 72)

    # Warmup
    print("Warming up CPU...")
    dummy_img = np.random.randint(0, 255, (proc_h, proc_w, 3), dtype=np.uint8)
    for _ in range(3):
        detector(dummy_img)
        pose_estimator(dummy_img, [[0, 0, proc_w, proc_h]])
    print("Warmup done.\n")

    while True:
        t_total_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            if args.clip:
                print("Video ended.")
            break

        frame_count += 1

        # Resize for processing
        if proc_w != src_w or proc_h != src_h:
            proc_frame = cv2.resize(frame, (proc_w, proc_h))
        else:
            proc_frame = frame

        timings = {}

        # --- Detection ---
        t0 = time.perf_counter()
        if frame_count % det_every == 1 or det_every == 1:
            bboxes = detector(proc_frame)
            cached_bboxes = bboxes
        else:
            bboxes = cached_bboxes
        timings["det"] = (time.perf_counter() - t0) * 1000

        # --- Pose estimation ---
        t0 = time.perf_counter()
        if len(bboxes) > 0:
            keypoints, scores = pose_estimator(proc_frame, bboxes)
        else:
            keypoints = np.zeros((0, 17, 2))
            scores = np.zeros((0, 17))
        timings["pose"] = (time.perf_counter() - t0) * 1000

        # --- Buffer management ---
        kpts_buffer.append(keypoints)
        scores_buffer.append(scores)

        # --- Action recognition (skip frames for speed) ---
        t0 = time.perf_counter()
        if len(kpts_buffer) >= args.min_frames and (
            frame_count % recog_every == 0 or recog_every == 1
        ):
            results = recognizer(
                list(kpts_buffer), list(scores_buffer), img_shape=(proc_h, proc_w)
            )
        timings["recog"] = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000
        timings["total"] = t_total
        fps_window.append(1000.0 / max(t_total, 1e-6))
        current_fps = sum(fps_window) / len(fps_window)

        # --- Terminal timing output ---
        alpha = 0.1  # exponential moving average
        for k in avg_timings:
            if k in timings:
                avg_timings[k] = avg_timings[k] * (1 - alpha) + timings[k] * alpha

        if frame_count % PRINT_EVERY == 0 or frame_count == 1:
            action_str = results[0][1] if results else "—"
            print(
                f"{frame_count:>6} "
                f"{avg_timings['det']:>7.1f}ms "
                f"{avg_timings['pose']:>7.1f}ms "
                f"{avg_timings['recog']:>7.1f}ms "
                f"{avg_timings['total']:>7.1f}ms "
                f"{current_fps:>6.1f}  "
                f"{action_str}"
            )

        # --- Visualization ---
        vis_frame = proc_frame.copy()
        if len(keypoints) > 0:
            draw_skeleton(vis_frame, keypoints, scores, kpt_thr=0.3)

        # Draw bboxes
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        draw_hud(vis_frame, results, current_fps, timings, frame_count)

        # Write to output video
        if video_writer is not None:
            video_writer.write(vis_frame)

        cv2.imshow("ONNX Action Recognition", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  Recording saved to: {args.output}")
    cv2.destroyAllWindows()

    # --- Final summary ---
    print("\n" + "=" * 72)
    print(f"  Processed {frame_count} frames | Avg FPS: {current_fps:.1f}")
    print(
        f"  Avg timings:  det={avg_timings['det']:.1f}ms  "
        f"pose={avg_timings['pose']:.1f}ms  "
        f"recog={avg_timings['recog']:.1f}ms  "
        f"total={avg_timings['total']:.1f}ms"
    )
    print("=" * 72)


# ---------------------------------------------------------------------------
#  Clip mode — record then infer
# ---------------------------------------------------------------------------


def run_clip(args):
    """Clip mode: press 'r' to record a short clip, then run full pipeline."""
    print("\n=== ONNX Clip Mode — Record then Recognize ===")
    print(f"Device: {args.device}")

    # Load models
    model_dir = Path(args.model_dir)
    det_path = str(model_dir / args.det_model)
    pose_path = str(model_dir / args.pose_model)
    recog_path = str(model_dir / args.recog_model)
    recog_device = args.recog_device or args.device

    print("\nLoading models...")
    threads = getattr(args, 'threads', 0)
    detector = YOLOXDetector(det_path, device=args.device, score_thr=args.det_score_thr, threads=threads)
    pose_estimator = RTMPoseEstimator(pose_path, device=args.device, threads=threads)
    recognizer = STGCNRecognizer(
        recog_path,
        args.label_map,
        device=recog_device,
        clip_len=args.clip_len,
        num_person=args.num_person,
        threads=threads,
    )

    # Open camera
    source = args.clip if args.clip else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        return

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Processing resolution
    if args.short_side > 0:
        scale_r = args.short_side / min(src_h, src_w)
        proc_w, proc_h = int(src_w * scale_r), int(src_h * scale_r)
    else:
        proc_w, proc_h = src_w, src_h

    is_file = args.clip is not None
    frames_to_record = int(args.record_seconds * src_fps)
    total_file_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    print(f"\nSource: {source} ({src_w}×{src_h} @ {src_fps:.0f}fps)")
    print(f"Processing at: {proc_w}×{proc_h}")
    if is_file:
        n_windows = max(1, int(np.ceil(total_file_frames / frames_to_record)))
        print(f"Video file: {total_file_frames} frames ({total_file_frames/src_fps:.1f}s)")
        print(f"Splitting into {args.record_seconds}s windows → ~{n_windows} sub-clips")
    else:
        print(f'Press "r" to record {args.record_seconds}s ({frames_to_record} frames)')
    print("Press ESC/Q to quit.\n")

    # Warmup
    dummy_img = np.random.randint(0, 255, (proc_h, proc_w, 3), dtype=np.uint8)
    for _ in range(3):
        detector(dummy_img)
        pose_estimator(dummy_img, [[0, 0, proc_w, proc_h]])

    # Video writer for recording (saves annotated playback clips)
    video_writer = None
    if args.output:
        out_fps = args.output_fps if args.output_fps > 0 else 25
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (proc_w, proc_h))
        if video_writer.isOpened():
            print(
                f"Recording playbacks to: {args.output} ({proc_w}×{proc_h} @ {out_fps:.0f}fps)"
            )
        else:
            print(f"WARNING: Could not open video writer for {args.output}")
            video_writer = None

    # ── Helper: run detection + pose + recognition on a list of frames ──
    def _infer_subclip(subclip_frames, subclip_idx, time_start_s, time_end_s):
        """Run full pipeline on a sub-clip and show/write playback."""
        n = len(subclip_frames)
        print(f"\n  ━━ Sub-clip {subclip_idx} [{time_start_s:.1f}s – {time_end_s:.1f}s] "
              f"({n} frames) ━━")

        t_start = time.perf_counter()
        kpts_buf, scores_buf, vis_buf = [], [], []
        t_det_total, t_pose_total = 0, 0

        for cf in subclip_frames:
            t0 = time.perf_counter()
            bboxes = detector(cf)
            t_det_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            if len(bboxes) > 0:
                kpts, scores = pose_estimator(cf, bboxes)
            else:
                kpts = np.zeros((0, 17, 2))
                scores = np.zeros((0, 17))
            t_pose_total += time.perf_counter() - t0

            kpts_buf.append(kpts)
            scores_buf.append(scores)

            vf = cf.copy()
            if len(kpts) > 0:
                draw_skeleton(vf, kpts, scores, kpt_thr=0.3)
            for bb in bboxes:
                x1, y1, x2, y2 = [int(v) for v in bb]
                cv2.rectangle(vf, (x1, y1), (x2, y2), (0, 255, 0), 1)
            vis_buf.append(vf)

        t0 = time.perf_counter()
        results = recognizer(kpts_buf, scores_buf, img_shape=(proc_h, proc_w))
        t_recog = (time.perf_counter() - t0) * 1000
        t_total = (time.perf_counter() - t_start) * 1000

        label_str = f"{results[0][1]} ({results[0][2]:.1%})"
        top5 = results[:5]

        print(f"  {'Stage':<20} {'Total (ms)':<14} {'Per-frame (ms)':<14}")
        print(f"  {'-'*48}")
        print(f"  {'Detection':<20} {t_det_total*1000:<14.1f} {t_det_total*1000/n:<14.1f}")
        print(f"  {'Pose estimation':<20} {t_pose_total*1000:<14.1f} {t_pose_total*1000/n:<14.1f}")
        print(f"  {'Recognition':<20} {t_recog:<14.1f} {'—':<14}")
        print(f"  {'-'*48}")
        print(f"  {'TOTAL':<20} {t_total:<14.1f} {t_total/n:<14.1f}")
        print(f"\n  → {results[0][1]} ({results[0][2]:.1%})")
        for i, (idx, lbl, prob) in enumerate(top5):
            print(f"    {i+1}. {lbl:<30} {prob:.1%}")

        # Playback with overlay
        window_name = "ONNX Clip Mode  [r=record, Q/ESC=quit]"
        quit_requested = False
        for vf in vis_buf:
            overlay = vf.copy()
            cv2.rectangle(overlay, (0, 0), (proc_w, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vf, 0.4, 0, vf)
            cv2.putText(vf, f"[{time_start_s:.1f}-{time_end_s:.1f}s] {label_str}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            for j, (_, lbl, prob) in enumerate(top5):
                cv2.putText(vf, f"{lbl[:25]} {prob:.1%}",
                            (10, 55 + j * 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (200, 200, 200), 1)
            if video_writer is not None:
                video_writer.write(vf)
            cv2.imshow(window_name, vf)
            if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                quit_requested = True
                break

        return results, quit_requested

    WINDOW_NAME = "ONNX Clip Mode  [r=record, Q/ESC=quit]"
    recording = False
    clip_frames = []
    clip_count = 0
    last_label = 'Press "r" to record'
    last_top5 = []

    # ── For video files: read all frames, split into windows, infer each ──
    if is_file:
        print(f"  Reading all frames from {source}...")
        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if proc_w != src_w or proc_h != src_h:
                frame = cv2.resize(frame, (proc_w, proc_h))
            all_frames.append(frame)
        cap.release()
        print(f"  Read {len(all_frames)} frames ({len(all_frames)/src_fps:.1f}s)")

        # Split into windows
        window_size = frames_to_record
        all_results = []
        subclip_idx = 0

        for start in range(0, len(all_frames), window_size):
            end = min(start + window_size, len(all_frames))
            subclip = all_frames[start:end]
            if len(subclip) < 4:  # skip tiny leftover
                break
            subclip_idx += 1
            t_start_s = start / src_fps
            t_end_s = end / src_fps

            results, quit_req = _infer_subclip(subclip, subclip_idx, t_start_s, t_end_s)
            all_results.append((t_start_s, t_end_s, results))
            clip_count += 1
            if quit_req:
                break

        # Print summary
        print(f"\n{'='*60}")
        print(f"  SUMMARY — {subclip_idx} sub-clips from {source}")
        print(f"{'='*60}")
        for t_s, t_e, res in all_results:
            print(f"  [{t_s:5.1f}s – {t_e:5.1f}s]  {res[0][1]:<30} {res[0][2]:.1%}")
        print(f"{'='*60}")

        if video_writer is not None:
            video_writer.release()
            print(f"  All clips saved to: {args.output}")
        cv2.destroyAllWindows()
        print("\nClip mode ended.")
        return

    # ── Webcam mode: press 'r' to record ──
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if proc_w != src_w or proc_h != src_h:
            frame = cv2.resize(frame, (proc_w, proc_h))

        if recording:
            clip_frames.append(frame.copy())

            # Recording indicator
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"REC {len(clip_frames)}/{frames_to_record}",
                (50, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            if len(clip_frames) >= frames_to_record:
                recording = False
                print(f"\n  Recorded {len(clip_frames)} frames — running inference...")

                # --- Full pipeline on the recorded clip ---
                t_start = time.perf_counter()

                kpts_buffer = []
                scores_buffer = []
                vis_frames = []

                t_det_total = 0
                t_pose_total = 0

                for i, cf in enumerate(clip_frames):
                    t0 = time.perf_counter()
                    bboxes = detector(cf)
                    t_det_total += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if len(bboxes) > 0:
                        kpts, scores = pose_estimator(cf, bboxes)
                    else:
                        kpts = np.zeros((0, 17, 2))
                        scores = np.zeros((0, 17))
                    t_pose_total += time.perf_counter() - t0

                    kpts_buffer.append(kpts)
                    scores_buffer.append(scores)

                    # Annotated frame for playback
                    vf = cf.copy()
                    if len(kpts) > 0:
                        draw_skeleton(vf, kpts, scores, kpt_thr=0.3)
                    for bb in bboxes:
                        x1, y1, x2, y2 = [int(v) for v in bb]
                        cv2.rectangle(vf, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    vis_frames.append(vf)

                # Recognition
                t0 = time.perf_counter()
                results = recognizer(
                    kpts_buffer, scores_buffer, img_shape=(proc_h, proc_w)
                )
                t_recog = (time.perf_counter() - t0) * 1000
                t_total = (time.perf_counter() - t_start) * 1000

                n = len(clip_frames)
                last_label = f"{results[0][1]} ({results[0][2]:.1%})"
                last_top5 = results[:5]

                # Print detailed timing to terminal
                print(f"\n  {'Stage':<20} {'Total (ms)':<14} {'Per-frame (ms)':<14}")
                print(f"  {'-'*48}")
                print(
                    f"  {'Detection':<20} {t_det_total*1000:<14.1f} {t_det_total*1000/n:<14.1f}"
                )
                print(
                    f"  {'Pose estimation':<20} {t_pose_total*1000:<14.1f} {t_pose_total*1000/n:<14.1f}"
                )
                print(f"  {'Recognition':<20} {t_recog:<14.1f} {'—':<14}")
                print(f"  {'-'*48}")
                print(f"  {'TOTAL':<20} {t_total:<14.1f} {t_total/n:<14.1f}")
                print(f"\n  → {results[0][1]} ({results[0][2]:.1%})")
                for i, (idx, lbl, prob) in enumerate(results[:5]):
                    print(f"    {i+1}. {lbl:<30} {prob:.1%}")

                # Show skeleton playback (and optionally record)
                clip_count += 1
                print(f"\n  Playing back {n} annotated frames...")
                for vf in vis_frames:
                    # Overlay result on playback
                    overlay = vf.copy()
                    cv2.rectangle(overlay, (0, 0), (proc_w, 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, vf, 0.4, 0, vf)
                    cv2.putText(
                        vf,
                        last_label,
                        (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                    for j, (_, lbl, prob) in enumerate(last_top5):
                        cv2.putText(
                            vf,
                            f"{lbl[:25]} {prob:.1%}",
                            (10, 60 + j * 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )
                    if video_writer is not None:
                        video_writer.write(vf)
                    cv2.imshow(WINDOW_NAME, vf)
                    if cv2.waitKey(50) & 0xFF in (27, ord("q")):
                        break

                if video_writer is not None:
                    print(f"  Clip {clip_count} written to {args.output}")
                clip_frames = []
                print(f'\n  Ready — press "r" to record again.\n')
        else:
            # Idle — show live preview with last result
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (proc_w, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(
                frame,
                last_label,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("r") and not recording:
            recording = True
            clip_frames = []
            print("  Recording...")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"  All clips saved to: {args.output}")
    cv2.destroyAllWindows()
    print("\nClip mode ended.")


# ---------------------------------------------------------------------------
#  Benchmark mode
# ---------------------------------------------------------------------------


def run_benchmark(args):
    """Benchmark each stage independently."""
    import onnxruntime as ort

    recog_device = args.recog_device or args.device
    print("\n=== ONNX Pipeline Benchmark ===")
    print(f"Device: det/pose={args.device}, recog={recog_device}")

    model_dir = Path(args.model_dir)
    threads = getattr(args, 'threads', 0)
    detector = YOLOXDetector(str(model_dir / args.det_model), device=args.device, threads=threads)
    pose_estimator = RTMPoseEstimator(
        str(model_dir / args.pose_model), device=args.device, threads=threads
    )
    recognizer = STGCNRecognizer(
        str(model_dir / args.recog_model), args.label_map, device=recog_device, threads=threads
    )

    N = args.benchmark_iters
    h, w = 480, 640
    dummy_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        detector(dummy_img)
        pose_estimator(dummy_img, [[100, 100, 400, 400]])
        recognizer([np.zeros((1, 17, 2))], [np.zeros((1, 17))], img_shape=(h, w))

    # Detection benchmark
    print(f"\nBenchmarking {N} iterations...")
    t0 = time.perf_counter()
    for _ in range(N):
        detector(dummy_img)
    det_ms = (time.perf_counter() - t0) / N * 1000

    # Pose benchmark (1 person)
    t0 = time.perf_counter()
    for _ in range(N):
        pose_estimator(dummy_img, [[100, 100, 400, 400]])
    pose_ms = (time.perf_counter() - t0) / N * 1000

    # Recognition benchmark
    kpts_buf = [np.random.randn(1, 17, 2) for _ in range(100)]
    scores_buf = [np.random.rand(1, 17) for _ in range(100)]
    t0 = time.perf_counter()
    for _ in range(N):
        recognizer(kpts_buf, scores_buf, img_shape=(h, w))
    recog_ms = (time.perf_counter() - t0) / N * 1000

    total_ms = det_ms + pose_ms + recog_ms
    fps = 1000.0 / total_ms

    print(f'\n{"Stage":<20} {"Time (ms)":<12} {"FPS":<10}')
    print("-" * 42)
    print(f'{"Detection":<20} {det_ms:<12.1f} {1000/det_ms:<10.1f}')
    print(f'{"Pose (1 person)":<20} {pose_ms:<12.1f} {1000/pose_ms:<10.1f}')
    print(f'{"Recognition":<20} {recog_ms:<12.1f} {1000/recog_ms:<10.1f}')
    print("-" * 42)
    print(f'{"TOTAL":<20} {total_ms:<12.1f} {fps:<10.1f}')
    print(f"\n→ Estimated pipeline: {fps:.1f} FPS ({args.device})")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure-ONNX Real-Time Skeleton Action Recognition"
    )

    # Modes
    p.add_argument(
        "--mode",
        choices=["stream", "clip"],
        default="stream",
        help="stream = continuous live recognition; "
        'clip = press "r" to record then infer',
    )
    p.add_argument(
        "--clip", type=str, default=None, help="Path to video file (default: webcam)"
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode instead of live demo",
    )
    p.add_argument(
        "--record-seconds",
        type=float,
        default=3.0,
        help="Seconds to record in clip mode (default: 3.0)",
    )

    # Device
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--recog-device",
        choices=["cuda", "cpu"],
        default="cpu",
        help="Device for STGCN++ recognizer (default: cpu, avoids cuDNN issues)",
    )

    # Model paths
    p.add_argument(
        "--model-dir",
        default="demo/onnx_models",
        help="Directory containing ONNX models",
    )
    p.add_argument("--det-model", default="yolox_tiny.onnx")
    p.add_argument("--pose-model", default="rtmpose_m.onnx")
    p.add_argument("--recog-model", default="stgcnpp_ntu120_xsub_hrnet_j.onnx")
    p.add_argument("--label-map", default="tools/data/label_map/nturgbd_120.txt")

    # Pipeline
    p.add_argument(
        "--window-frames",
        type=int,
        default=30,
        help="Sliding window size (frames to accumulate)",
    )
    p.add_argument(
        "--clip-len",
        type=int,
        default=100,
        help="Temporal length for STGCN++ (uniform sampled)",
    )
    p.add_argument("--num-person", type=int, default=2, help="Max persons for STGCN++")
    p.add_argument(
        "--min-frames", type=int, default=8, help="Min frames before first recognition"
    )

    # Detection
    p.add_argument("--det-score-thr", type=float, default=0.5)
    p.add_argument(
        "--det-every",
        type=int,
        default=1,
        help="Run detection every N frames (1=every frame)",
    )
    p.add_argument(
        "--recog-every",
        type=int,
        default=1,
        help="Run STGCN++ recognition every N frames (1=every frame, try 4-8 on slow machines)",
    )

    # Resolution
    p.add_argument(
        "--short-side",
        type=int,
        default=0,
        help="Resize short side to this (0=no resize)",
    )

    # Recording
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save annotated output to video file (e.g. demo_recording.mp4)",
    )
    p.add_argument(
        "--output-fps",
        type=float,
        default=0,
        help="FPS for output video (0=match source or 25 for webcam)",
    )

    # Performance
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads (0=auto, try 4 on laptops)",
    )

    # Benchmark
    p.add_argument("--benchmark-iters", type=int, default=100)

    # Fast preset
    p.add_argument(
        "--fast",
        action="store_true",
        help="Enable speed optimizations: det-every=2, recog-every=4, short-side=320",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply --fast preset (only override if user didn't set explicitly)
    if args.fast:
        if args.det_every == 1:      # default
            args.det_every = 2
        if args.recog_every == 1:     # default
            args.recog_every = 4
        if args.short_side == 0:      # default
            args.short_side = 320
        print("[--fast] det-every=2, recog-every=4, short-side=320")

    if args.benchmark:
        run_benchmark(args)
    elif args.mode == "clip":
        run_clip(args)
    else:
        run_stream(args)
