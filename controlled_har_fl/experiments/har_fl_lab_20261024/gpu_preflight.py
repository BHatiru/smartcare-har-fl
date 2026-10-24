"""Exercise the actual HAR backbone forward/backward; no research accuracy claim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.threads) < 1 or args.warmup < 0:
        raise ValueError("Positive batch, step and thread counts required")
    if args.output.exists():
        raise FileExistsError(args.output)

    import numpy as np
    import torch
    from demo1_fed_skeleton.models.stgcn import build_model, count_parameters

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    device = torch.device(args.device)
    model = build_model(num_classes=15, in_channels=3, graph="coco", base_channels=64,
                        num_stages=6, num_person=2, dropout=.3).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
    x = torch.randn(args.batch_size, 2, 100, 17, 3, device=device)
    y = torch.arange(args.batch_size, device=device) % 15
    first = next(model.parameters())
    before = first.detach().clone()

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.shape != (args.batch_size, 15):
            raise AssertionError(f"Unexpected output: {logits.shape}")
        loss = torch.nn.functional.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise FloatingPointError("Non-finite gradient")
        optimizer.step()
        return loss.item()

    for _ in range(args.warmup):
        step()
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times, losses = [], []
    for _ in range(args.steps):
        synchronize()
        started = time.perf_counter()
        losses.append(step())
        synchronize()
        times.append(time.perf_counter() - started)
    if torch.equal(before, first.detach()):
        raise AssertionError("Model weights did not change")

    result = {
        "passed": True, "scope": "Synthetic kernel/backward compatibility probe of the actual backbone; includes gradient checks; not full-data throughput, convergence or test accuracy.",
        "device": args.device, "parameters": count_parameters(model),
        "input_shape": list(x.shape), "dtype": str(x.dtype), "steps": args.steps,
        "step_seconds": times, "median_step_seconds": statistics.median(times),
        "finite_losses": losses, "weight_update_verified": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
        "compiled_cuda_architectures": torch.cuda.get_arch_list() if device.type == "cuda" else [],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None,
        "model_source_sha256": hashlib.sha256((ROOT / "demo1_fed_skeleton/models/stgcn.py").read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
