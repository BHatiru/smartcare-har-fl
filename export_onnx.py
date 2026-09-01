"""
export_onnx.py — Export the federated STGCN model to ONNX.

Produces an ONNX file compatible with Demo 2's real-time inference pipeline.
Follows the same wrapper pattern as ``tools/export_stgcnpp_onnx.py`` in the
main pyskl repo so that the resulting graph has a clean signature:

    Input : skeleton  (N, M=2, T=100, V=17, C=3)   float32   (pyskl convention)
    Output: logits    (N, num_classes)               float32

Usage
-----
python export_onnx.py --checkpoint outputs/global_model.pt \
                      --output outputs/model.onnx \
                      --num-classes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.stgcn import build_model  # noqa: E402


class STGCNONNXWrapper(nn.Module):
    """Thin wrapper that accepts (N, M, T, V, C) and returns logits."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def main():
    parser = argparse.ArgumentParser(description="Export STGCN to ONNX")
    parser.add_argument("--checkpoint", type=str, default="outputs/global_model.pt",
                        help="Path to .pt state-dict checkpoint.")
    parser.add_argument("--output", type=str, default="outputs/model.onnx")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-person", type=int, default=2,
                        help="Number of person slots (M dimension).")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=6)
    parser.add_argument("--T", type=int, default=100, help="Temporal length")
    parser.add_argument("--simplify", action="store_true",
                        help="Run onnx-simplifier after export.")
    parser.add_argument("--verify", action="store_true",
                        help="Verify with ONNX Runtime.")
    args = parser.parse_args()

    # Build model
    model = build_model(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        num_person=args.num_person,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    wrapper = STGCNONNXWrapper(model)
    wrapper.eval()

    # Dummy input: (N=1, M, T, V, C) — pyskl convention
    dummy = torch.randn(1, args.num_person, args.T, 17, args.in_channels)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["skeleton"],
        output_names=["logits"],
        dynamic_axes={
            "skeleton": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"ONNX model exported → {out_path}")

    # Optional: simplify
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            onnx_model = onnx.load(str(out_path))
            simplified, ok = onnx_simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(out_path))
                print("ONNX model simplified ✓")
            else:
                print("Warning: onnx-simplifier reported failure")
        except ImportError:
            print("onnx / onnxsim not installed — skipping simplification")

    # Optional: verify with ONNX Runtime
    if args.verify:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(out_path))
            dummy_np = dummy.numpy()
            ort_out = sess.run(None, {"skeleton": dummy_np})[0]
            torch_out = wrapper(dummy).detach().numpy()
            diff = np.abs(ort_out - torch_out).max()
            print(f"Max diff (torch vs ORT): {diff:.6e}")
            assert diff < 1e-4, f"Verification failed! diff={diff}"
            print("ONNX Runtime verification passed ✓")
        except ImportError:
            print("onnxruntime not installed — skipping verification")


if __name__ == "__main__":
    main()
