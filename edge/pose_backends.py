#!/usr/bin/env python
"""Pluggable pose backends for the edge emergency node.

All backends present the SAME call signature as demo_onnx.RTMPoseEstimator:
    __call__(img, bboxes=None) -> (keypoints (N,17,2) pixel-space, scores (N,17))
and emit COCO-17 keypoint ordering, so the trained STGCN++ medical model works
downstream unchanged regardless of which front-end produced the skeleton.

Backends
--------
rtmpose  : two-stage  YOLOX-tiny detector -> RTMPose top-down  (demo_onnx classes)
movenet  : single-stage MoveNet SinglePose (TFLite) — no separate detector,
           single-person. Validated 2026-06-10 on real falls (URFD): the
           *Thunder* int8 variant preserves fall detection (peak P(fall) 99-100%,
           matching RTMPose) at ~4x lower latency; the *Lightning* variant is too
           lossy on falls and must NOT be used for emergency detection.
"""

from pathlib import Path

import cv2
import numpy as np


class MoveNetEstimator:
    """MoveNet SinglePose (Lightning/Thunder) TFLite wrapper via ai-edge-litert.

    Converts MoveNet's native (y, x, score) in [0,1] to pixel-space (x, y),
    COCO-17 order, and returns a single-person batch shaped (1,17,2)/(1,17) so
    the recognizer (which pads to its fixed M=2) sees a normal skeleton.
    Incoming bboxes are ignored — MoveNet localises one person on the full frame.
    Input size is read from the model (Lightning=192, Thunder=256, square).
    """

    def __init__(self, model_path, device="cpu", threads=0):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MoveNet backend needs ai-edge-litert: "
                "uv pip install --python <venv-python> ai-edge-litert"
            ) from e
        n_threads = threads if threads and threads > 0 else None
        self.interp = Interpreter(model_path=str(model_path), num_threads=n_threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_size = int(self.inp["shape"][1])  # square H==W
        self.in_dtype = self.inp["dtype"]
        print(f"  MoveNet: {Path(model_path).name} in={self.in_size}px "
              f"dtype={np.dtype(self.in_dtype).name} threads={n_threads or 'auto'}")

    def _preprocess(self, img):
        h, w = img.shape[:2]
        scale = self.in_size / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.in_size, self.in_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = canvas[None]  # (1,H,W,3) NHWC
        if self.in_dtype == np.float32:
            blob = blob.astype(np.float32)
        return blob, scale

    def __call__(self, img, bboxes=None):
        blob, scale = self._preprocess(img)
        self.interp.set_tensor(self.inp["index"], blob)
        self.interp.invoke()
        kpts = self.interp.get_tensor(self.out["index"])[0, 0]  # (17,3) = (y,x,score) norm
        ys = kpts[:, 0] * self.in_size / scale
        xs = kpts[:, 1] * self.in_size / scale
        scores = kpts[:, 2].astype(np.float32)
        xy = np.stack([xs, ys], axis=-1).astype(np.float32)  # (17,2) pixel (x,y)
        return xy[None], scores[None]


class MoveNetMultiPose:
    """MoveNet MultiPose Lightning (TFLite) — up to 6 people in ONE model pass.

    Same call signature + COCO-17 output as the single-person estimators, but it
    detects multiple people AND gives each a confidence score, so non-human
    clutter is filtered out (fixes the single-pose "locks onto human-shaped
    objects" problem). Returns instances sorted by confidence (most confident
    first), so the recognizer's first M slots get the strongest detections.

    Output per instance (56 floats): [51 = 17*(y,x,score)] + [ymin,xmin,ymax,xmax] + [score].
    """

    def __init__(self, model_path, device="cpu", threads=0, in_size=256, score_thr=0.2):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as e:  # pragma: no cover
            raise ImportError("MultiPose needs ai-edge-litert") from e
        n_threads = threads if threads and threads > 0 else None
        self.in_size = (in_size // 32) * 32 or 256          # must be a multiple of 32
        self.score_thr = score_thr
        self.interp = Interpreter(model_path=str(model_path), num_threads=n_threads)
        inp = self.interp.get_input_details()[0]
        self.interp.resize_tensor_input(inp["index"], [1, self.in_size, self.in_size, 3])
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.in_dtype = self.inp["dtype"]
        print(f"  MoveNet-MultiPose: {Path(model_path).name} in={self.in_size}px "
              f"thr={score_thr} threads={n_threads or 'auto'}")

    def __call__(self, img, bboxes=None):
        h, w = img.shape[:2]
        scale = self.in_size / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.zeros((self.in_size, self.in_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[None]
        if self.in_dtype == np.float32:
            blob = blob.astype(np.float32)
        self.interp.set_tensor(self.inp["index"], blob)
        self.interp.invoke()
        out = self.interp.get_tensor(self.out["index"])[0]   # (6, 56)

        f = self.in_size / scale  # normalized [0,1] -> original pixels
        kpts, scores, inst_sc = [], [], []
        for inst in out:
            if float(inst[55]) < self.score_thr:
                continue
            kp = inst[:51].reshape(17, 3)
            xs = kp[:, 1] * f
            ys = kp[:, 0] * f
            kpts.append(np.stack([xs, ys], axis=-1).astype(np.float32))
            scores.append(kp[:, 2].astype(np.float32))
            inst_sc.append(float(inst[55]))
        if not kpts:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        order = np.argsort(inst_sc)[::-1]                    # most confident first
        return (np.stack([kpts[i] for i in order]),
                np.stack([scores[i] for i in order]))


def has_person(scores, kpt_thr=0.2, min_kpts=5):
    """Presence gate: True only if the top person has >= min_kpts confident joints.

    Guards against empty/garbage skeletons being recognised as emergencies. This
    matters because an all-zeros skeleton scores 'staggering' ~0.86 on the cent2d
    STGCN++, and MoveNet ALWAYS emits one (low-confidence) skeleton even on an
    empty scene — so without this gate an empty room raises false HIGH alerts.
    """
    if scores is None or len(scores) == 0:
        return False
    s = np.asarray(scores)
    if s.ndim == 1:
        s = s[None]
    return int((s[0] > kpt_thr).sum()) >= min_kpts


def run_frontend(detector, pose, frame, frame_count, det_every, cached_bboxes):
    """Detect (with skip cadence) + estimate pose for one frame.

    Handles both pipelines transparently:
      - two-stage (detector set): YOLOX every `det_every` frames -> RTMPose on boxes
      - single-stage (detector is None): MoveNet localises + poses one person

    Returns (keypoints, scores, bboxes, cached_bboxes).
    """
    if detector is not None:
        if frame_count % det_every == 1 or det_every == 1:
            cached_bboxes = detector(frame)
        bboxes = cached_bboxes
        if bboxes:
            keypoints, scores = pose(frame, bboxes)
        else:
            keypoints, scores = np.zeros((0, 17, 2)), np.zeros((0, 17))
    else:
        bboxes = []
        keypoints, scores = pose(frame)
    return keypoints, scores, bboxes, cached_bboxes
