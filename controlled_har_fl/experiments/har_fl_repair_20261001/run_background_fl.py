"""Bounded local full-data queue; checkpoints permit exact round-boundary resume."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(torch, path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def power_status():
    if os.name != "nt":
        return None

    class Power(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    value = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        return None
    return {"ac_online": value.ACLineStatus == 1 if value.ACLineStatus != 255 else None,
            "battery_percent": value.BatteryLifePercent if value.BatteryLifePercent <= 100 else None}


def acquire_lock(path):
    stream = Path(path).open("a+b")
    if stream.tell() == 0:
        stream.write(b"1")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another process owns this experiment's RUN.lock") from None
    return stream


class StopRequested(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=12.)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_hours <= 0 or min(args.rounds, args.batch_size, args.threads) < 1:
        raise ValueError("Positive time and training budgets required")
    args.code_dir = args.code_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output_dir / "RUN.lock")
    deadline = time.monotonic() + args.max_hours * 3600
    sys.path.insert(0, str(args.code_dir))
    import numpy as np
    import torch
    from demo1_fed_skeleton import corrected_fl as fl
    from demo1_fed_skeleton import development_fl as dev
    from demo1_fed_skeleton.models.stgcn import build_model

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    methods = ["centralized", "fedavg", "fedbn", "fedprox", "local"]
    source_hashes = {name: fl.sha256(args.code_dir / "demo1_fed_skeleton" / name)
                     for name in ("corrected_fl.py", "development_fl.py", "models/stgcn.py")}
    source_hashes["background_worker"] = fl.sha256(__file__)
    binding = {"protocol_sha256": fl.sha256(args.data_dir / "protocol.json"),
               "source_sha256": source_hashes, "seed": args.seed, "batch_size": args.batch_size,
               "threads": args.threads, "methods": methods, "lr": .01, "local_epochs": 1,
               "proximal_mu": .01, "device": "cpu", "torch": str(torch.__version__),
               "numpy": np.__version__, "python": sys.version.split()[0]}
    manifest_path = args.output_dir / "job.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["binding"] != binding:
            raise ValueError("Source, data, environment or training configuration changed; use a new run")
    manifest = {"binding": binding, "target_rounds_per_method": args.rounds,
                "max_hours_this_invocation": args.max_hours,
                "scheduling": "one complete round of each method in order, repeated to target",
                "resume": "Completed round only. Interrupted round replays deterministically from its start.",
                "test_evaluated": False, "publication_accuracy_claim_allowed": False,
                "data_dir": str(args.data_dir), "code_dir": str(args.code_dir)}
    atomic_json(manifest_path, manifest)
    current = {"state": "loading_data", "pid": os.getpid(), "round": 0, "method": None,
               "target_rounds": args.rounds, "publication_accuracy_claim_allowed": False}

    def status(**updates):
        current.update(updates)
        current["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output_dir / "status.json", current)
        text = (f"State: {current['state']}\nMethod: {current.get('method')}\n"
                f"Round: {current['round']} / {args.rounds}\nPID: {os.getpid()}\n"
                f"Updated UTC: {current['updated_utc']}\n"
                f"Progress: {json.dumps(current.get('progress', {}))}\n"
                f"Detail: {current.get('detail', '')}\n\n"
                "Full-data development experiment. Test data is not evaluated.\n"
                "Create STOP_REQUESTED in this directory to stop after a training batch.\n"
                "Completed-round checkpoints remain; an unfinished round restarts on resume.\n")
        temporary = args.output_dir / "STATUS.txt.tmp"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.output_dir / "STATUS.txt")

    last_power_check, last_power = 0., None

    def check_stop():
        nonlocal last_power_check, last_power
        now = time.monotonic()
        if (args.output_dir / "STOP_REQUESTED").exists():
            raise StopRequested("STOP_REQUESTED file found")
        if now >= deadline:
            raise StopRequested("This invocation's wall-time budget was reached")
        if now - last_power_check > 30:
            last_power = power_status()
            last_power_check = now
        if (last_power and last_power["ac_online"] is False and
                last_power["battery_percent"] is not None and last_power["battery_percent"] <= 20):
            raise StopRequested("Battery at or below20% with AC disconnected")

    keep_awake = False
    try:
        status()
        protocol, train, development = dev.load_development(args.data_dir)
        status(state="ready", training_clips=sum(map(len, train)), development_clips=sum(map(len, development)))
        print(f"DATA train={sum(map(len, train))} development={sum(map(len, development))}; no test input", flush=True)
        if os.name == "nt":
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        factory = lambda: build_model(num_classes=15, in_channels=3, graph="coco", base_channels=64,
                                      num_stages=6, num_person=2, dropout=.3)
        for target_round in range(1, args.rounds + 1):
            for method in methods:
                check_stop()
                folder = args.output_dir / method
                folder.mkdir(exist_ok=True)
                result_path = folder / f"round_{target_round:03d}.json"
                if result_path.exists():
                    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
                    if saved_result["binding"] != binding:
                        raise ValueError("Saved round belongs to another experiment")
                    for filename, expected in saved_result["artifact_sha256"].items():
                        if fl.sha256(folder / filename) != expected:
                            raise ValueError(f"Saved artifact changed: {filename}")
                    continue
                status(state="training", method=method, round=target_round, progress={})
                print(f"START {method} round={target_round}", flush=True)
                resume = None
                for saved_round in (target_round, target_round - 1):
                    path = folder / f"checkpoint_{saved_round:03d}.pt"
                    if path.exists():
                        resume = torch.load(path, map_location="cpu", weights_only=True)
                        if resume["binding"] != binding:
                            raise ValueError("Checkpoint source/data binding changed")
                        break
                if target_round > 1 and resume is None:
                    raise ValueError("Missing prior round checkpoint")
                last_reported = 0.

                def progress(entry):
                    nonlocal last_reported
                    check_stop()
                    if entry["optimizer_steps"] == 1 or time.monotonic() - last_reported >= 45:
                        status(state="training", progress=entry, power=last_power)
                        print(f"PROGRESS {method} R{entry['round']} C{entry['client']} "
                              f"{entry['examples_processed']}/{entry['total_examples']} "
                              f"loss={entry['training_loss']:.4f}", flush=True)
                        last_reported = time.monotonic()

                def save(state):
                    state["binding"] = binding
                    completed = state["current"]["round"]
                    path = folder / f"checkpoint_{completed:03d}.pt"
                    atomic_checkpoint(torch, path, state)
                    status(last_saved_checkpoint=str(path), last_completed_round_for_method=completed)

                result, endpoints = dev.run_method(
                    factory, train, development, method=method, rounds=target_round, local_epochs=1,
                    batch_size=args.batch_size, lr=.01, seed=args.seed, device="cpu", num_classes=15,
                    proximal_mu=.01, resume_state=resume, on_checkpoint=save, on_progress=progress)
                hashes = {}
                for endpoint, (predictions, _) in endpoints.items():
                    path = folder / f"round_{target_round:03d}_{endpoint}.npz"
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, **predictions)
                    os.replace(temporary, path)
                    hashes[path.name] = fl.sha256(path)
                checkpoint_path = folder / f"checkpoint_{target_round:03d}.pt"
                hashes[checkpoint_path.name] = fl.sha256(checkpoint_path)
                result.update(binding=binding, artifact_sha256=hashes, protocol=protocol,
                              publication_accuracy_claim_allowed=False, test_evaluated=False)
                atomic_json(result_path, result)
                metric = result["final_development"]["pooled"]
                status(state="round_complete", progress={}, development_accuracy=metric["accuracy"],
                       development_macro_f1=metric["macro_f1_all_declared_classes"])
                print(f"DONE {method} R{target_round} development_accuracy={metric['accuracy']:.4f} "
                      f"macro_f1={metric['macro_f1_all_declared_classes']:.4f}", flush=True)
                del result, endpoints, resume
                gc.collect()
        status(state="complete", detail="All scheduled development rounds finished")
        print("COMPLETE", flush=True)
    except StopRequested as error:
        status(state="stopped_resumable", detail=str(error))
        print(f"STOPPED: {error}; resume from saved completed rounds", flush=True)
    except BaseException as error:
        status(state="failed_resumable", detail=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == "__main__":
    main()
