"""Matched HAR training with development-only selection; this runner has no test input."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import corrected_fl as fl
from .models.stgcn import build_model, count_parameters
from .prepare_corrected_fl import class_contract

METHODS = ("centralized", "local", "fedavg", "fedbn", "fedprox")


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_bytes(state):
    return sum(value.numel() * value.element_size() for value in state.values())


def shard_hash(shards):
    digest = hashlib.sha256()
    for shard in shards:
        for array in (shard.x.numpy(), shard.y.numpy(), shard.sequence_ids, shard.subject_ids):
            array = np.ascontiguousarray(array)
            digest.update(f"{array.dtype}:{array.shape}".encode())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def pack_predictions(predictions):
    if predictions is None:
        return None
    return {key: value.tolist() if value.dtype.kind in "US" else torch.from_numpy(value.copy())
            for key, value in predictions.items()}


def unpack_predictions(predictions):
    if predictions is None:
        return None
    return {key: value.numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in predictions.items()}


def merge_shards(shards):
    return fl.Shard(torch.cat([s.x for s in shards]), torch.cat([s.y for s in shards]),
                    np.concatenate([s.sequence_ids for s in shards]),
                    np.concatenate([s.subject_ids for s in shards]))


def validate_shards(train, development):
    if not train or len(train) != len(development) or any(len(s) == 0 for s in train + development):
        raise ValueError("Every client needs nonempty train and development data")
    identities = []
    for shards in (train, development):
        ids = np.concatenate([s.sequence_ids for s in shards])
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Duplicate sequence identity across clients")
        identities.append((set(ids.tolist()), set(np.concatenate([s.subject_ids for s in shards]).tolist())))
    if any(identities[0][i] & identities[1][i] for i in (0, 1)):
        raise ValueError("Training and development sequences and performers must be disjoint")


def load_development(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    mapping, names = class_contract(15)
    if (protocol.get("schema_version") != 2 or
            protocol.get("purpose") not in {"development_pilot", "development_training"} or
            protocol["label_names"] != names or protocol["ntu_zero_based_ids"] != list(mapping) or
            protocol["sample_shape"] != [2, 100, 17, 3] or protocol["graph"] != "coco"):
        raise ValueError("Expected version2 development-only medical15 COCO protocol")
    manifest_path = directory / "split_manifest.json"
    if fl.sha256(manifest_path) != protocol["split_manifest_sha256"]:
        raise ValueError("Split manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = {}
    for split in ("train", "development"):
        path = directory / f"{split}.npz"
        entry = protocol["files"][split]
        if fl.sha256(path) != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as data:
            x, y = data["x"], data["y"]
            ids, subjects, clients = (data[k] for k in ("sequence_ids", "subject_ids", "client_ids"))
        if (x.shape != (entry["samples"], 2, 100, 17, 3) or not np.isfinite(x).all() or
                y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= 15) or
                any(a.shape != (len(x),) for a in (ids, subjects, clients)) or
                clients.dtype.kind not in "iu" or subjects.dtype.kind not in "iu" or
                set(clients.tolist()) != set(range(protocol["num_clients"]))):
            raise ValueError(f"Invalid arrays: {split}")
        if (set(subjects) & set(protocol["excluded_official_test_subjects"]) or
                not set(ids).issubset(manifest[split]["all_eligible_sequence_ids"]) or
                not set(subjects).issubset(manifest[split]["all_eligible_subjects"])):
            raise ValueError("Prepared examples violate frozen performer split")
        shards[split] = [fl.Shard(torch.from_numpy(x[clients == c].astype(np.float32)),
                                  torch.from_numpy(y[clients == c].astype(np.int64)),
                                  ids[clients == c], subjects[clients == c])
                         for c in range(protocol["num_clients"])]
    validate_shards(shards["train"], shards["development"])
    return protocol, shards["train"], shards["development"]


def evaluate(models, development, *, centralized, device, batch_size, num_classes):
    matrices, logits, labels, ids, subjects, client_ids = [], [], [], [], [], []
    for cid, shard in enumerate(development):
        matrix, scores = fl.score_client(models[0] if centralized else models[cid], shard,
                                         device=device, batch_size=batch_size, num_classes=num_classes)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite development logits")
        matrices.append(matrix)
        logits.append(scores)
        labels.append(shard.y.numpy())
        ids.append(shard.sequence_ids)
        subjects.append(shard.subject_ids)
        client_ids.append(np.full(len(shard), cid, dtype=np.int64))
    predictions = dict(logits=np.concatenate(logits), y=np.concatenate(labels),
                       sequence_ids=np.concatenate(ids), subject_ids=np.concatenate(subjects),
                       client_ids=np.concatenate(client_ids))
    pooled = fl.summarize(np.sum(matrices, axis=0))
    pooled["cross_entropy"] = float(torch.nn.functional.cross_entropy(
        torch.from_numpy(predictions["logits"]), torch.from_numpy(predictions["y"])))
    client_metrics = [fl.summarize(m) for m in matrices]
    subject_metrics = []
    for subject in np.unique(predictions["subject_ids"]):
        mask = predictions["subject_ids"] == subject
        correct = predictions["logits"][mask].argmax(1) == predictions["y"][mask]
        subject_metrics.append({"subject_id": int(subject), "n": int(mask.sum()),
                                "accuracy": float(correct.mean())})
    return {"pooled": pooled, "clients": client_metrics, "subjects": subject_metrics,
            "mean_client_accuracy": float(np.mean([m["accuracy"] for m in client_metrics])),
            "worst_client_accuracy": min(m["accuracy"] for m in client_metrics),
            "mean_subject_accuracy": float(np.mean([m["accuracy"] for m in subject_metrics]))}, predictions


def run_method(factory, train, development, *, method, rounds, local_epochs, batch_size,
               lr, seed, device, num_classes, proximal_mu=.01, on_round=None,
               on_checkpoint=None, resume_state=None, on_progress=None):
    validate_shards(train, development)
    if method not in METHODS or min(rounds, local_epochs, batch_size) < 1 or lr <= 0 or proximal_mu < 0:
        raise ValueError("Invalid method or training budget")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    template = factory()
    server = fl.state_copy(template)
    initial_hash = state_hash(server)
    centralized = method == "centralized"
    federated = method in {"fedavg", "fedbn", "fedprox"}
    models = [copy.deepcopy(template) for _ in range(1 if centralized else len(train))]
    training_shards = [merge_shards(train)] if centralized else train
    excluded = fl.batchnorm_keys(template) if method == "fedbn" else set()
    del template
    best_key, best_checkpoint, best_predictions, best_metrics = None, None, None, None
    history = []
    total_upload, total_download = 0, 0
    completed_round, previous_seconds = 0, 0.
    metrics, predictions = None, None
    contract = {"method": method, "local_epochs": local_epochs, "batch_size": batch_size,
                "lr": lr, "seed": seed, "device": str(device), "num_classes": num_classes,
                "proximal_mu": proximal_mu, "initial_state_sha256": initial_hash,
                "train_sha256": shard_hash(train), "development_sha256": shard_hash(development)}
    if resume_state is not None:
        if resume_state.get("schema_version") != 1 or resume_state["contract"] != contract:
            raise ValueError("Resume configuration or training/development data differs")
        current = resume_state["current"]
        completed_round = current["round"]
        if not 0 <= completed_round <= rounds or len(current["clients"]) != len(models):
            raise ValueError("Invalid resumed round or client count")
        history = copy.deepcopy(resume_state["history"])
        if [r["round"] for r in history] != list(range(1, completed_round + 1)):
            raise ValueError("Resume history is not consecutive")
        for model, saved in zip(models, current["clients"]):
            model.load_state_dict(saved, strict=True)
        if federated:
            server = {k: v.clone() for k, v in current["server"].items()}
        best_key = resume_state["best_key"]
        best_checkpoint = copy.deepcopy(resume_state["best_checkpoint"])
        best_predictions = unpack_predictions(resume_state["best_predictions"])
        best_metrics = copy.deepcopy(resume_state["best_metrics"])
        predictions = unpack_predictions(resume_state["last_predictions"])
        metrics = copy.deepcopy(resume_state["last_metrics"])
        previous_seconds = resume_state["completed_seconds"]
        total_upload = sum(r["logical_upload_bytes"] for r in history)
        total_download = sum(r["logical_download_bytes"] for r in history)
    start = time.perf_counter()

    def checkpoint(round_index):
        return {"method": method, "round": round_index, "seed": seed,
                "initial_state_sha256": initial_hash,
                "server": {k: v.clone() for k, v in server.items()} if federated else None,
                "clients": [fl.state_copy(model) for model in models]}

    def save_progress(round_index):
        if on_checkpoint:
            on_checkpoint({"schema_version": 1, "contract": contract, "current": checkpoint(round_index),
                           "history": copy.deepcopy(history), "best_key": best_key,
                           "best_checkpoint": copy.deepcopy(best_checkpoint),
                           "best_predictions": pack_predictions(best_predictions), "best_metrics": best_metrics,
                           "last_predictions": pack_predictions(predictions), "last_metrics": metrics,
                           "completed_seconds": previous_seconds + time.perf_counter() - start})

    if resume_state is None:
        save_progress(0)
    for round_index in range(completed_round + 1, rounds + 1):
        round_start = time.perf_counter()
        current_lr = lr * (.01 if round_index > 35 else .1 if round_index > 20 else 1.)
        training = []
        # Clients already hold the previous round's aggregate from development synchronization.
        for cid, (model, shard) in enumerate(zip(models, training_shards)):
            client_seed = seed + 100_000 * round_index + cid
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(client_seed)

            def step_progress(progress):
                if on_progress:
                    on_progress({"round": round_index, "client": cid, **progress})

            training.append(fl.train_client(
                model, shard, device=device, batch_size=batch_size, epochs=local_epochs,
                lr=current_lr, seed=client_seed, proximal_mu=proximal_mu if method == "fedprox" else 0.,
                reference_state=server if method == "fedprox" else None,
                on_step=step_progress if on_progress else None))
        upload, download = 0, 0
        if federated:
            updates = [{k: v for k, v in fl.state_copy(model).items() if k not in excluded} for model in models]
            upload = sum(state_bytes(update) for update in updates)
            server = fl.aggregate(server, updates, [len(s) for s in train], excluded=excluded)
            payload = {k: v for k, v in server.items() if k not in excluded}
            for model in models:
                fl.receive_shared(model, server, local_bn=method == "fedbn")
                download += state_bytes(payload)
            total_upload += upload
            total_download += download
        eval_start = time.perf_counter()
        metrics, predictions = evaluate(models, development, centralized=centralized,
                                        device=device, batch_size=batch_size, num_classes=num_classes)
        evaluation_seconds = time.perf_counter() - eval_start
        key = (metrics["pooled"]["macro_f1_all_declared_classes"], -metrics["pooled"]["cross_entropy"])
        if best_key is None or key > best_key:
            best_key, best_checkpoint = key, checkpoint(round_index)
            best_predictions, best_metrics = predictions, metrics
        entry = {"round": round_index, "lr": current_lr, "training": training, "development": metrics,
                 "logical_upload_bytes": upload, "logical_download_bytes": download,
                 "evaluation_seconds": evaluation_seconds, "seconds": time.perf_counter() - round_start}
        history.append(entry)
        save_progress(round_index)
        if on_round:
            on_round(entry)
    result = {
        "method": method, "seed": seed, "initial_state_sha256": initial_hash,
        "selection": "Maximum pooled development macro-F1; then lower cross-entropy; then earliest round",
        "selected_round": best_checkpoint["round"], "best_development": best_metrics,
        "final_development": metrics, "history": history,
        "training_examples_processed": sum(t["examples_processed"] for r in history for t in r["training"]),
        "optimizer_steps": sum(t["optimizer_steps"] for r in history for t in r["training"]),
        "training_seconds": sum(t["seconds"] for r in history for t in r["training"]),
        "total_seconds": previous_seconds + time.perf_counter() - start,
        "logical_upload_bytes": total_upload, "logical_download_bytes": total_download,
        "common_initial_state_bytes_per_model": state_bytes(server),
        "communication_scope": "Tensor payloads per simulated client, uploads and post-aggregation downloads each round; identical initialization pre-provisioned. Excludes transport/serialization, metrics and centralized raw-data transfer; not measured network traffic.",
    }
    return result, {"best_development": (best_predictions, best_checkpoint),
                    "final": (predictions, checkpoint(rounds))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--proximal-mu", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory; saved runs are never overwritten")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("Repeated methods")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no silent CPU fallback")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    protocol, train, development = load_development(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factory = lambda: build_model(num_classes=15, in_channels=3, graph="coco", base_channels=64,
                                  num_stages=6, num_person=2, dropout=.3)
    report = {
        "schema_version": 2, "purpose": protocol["purpose"], "publication_accuracy_claim_allowed": False,
        "test_evaluated": False, "protocol": protocol,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "numpy": np.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
        "parameters": count_parameters(factory()),
        "source_sha256": {name: fl.sha256(Path(__file__).parent / name) for name in
                          ("development_fl.py", "corrected_fl.py", "models/stgcn.py")},
        "invariants": {"pooled_bn_calibration": False, "centralized_warm_start": False,
                       "persistent_client_bn_for_fedbn": True, "test_checkpoint_selection": False,
                       "optimizer": "SGD momentum0.9 weight_decay0.0005; reset each local round/central epoch-block",
                       "schedule": "base through R20; base*0.1 R21-35; base*0.01 from R36",
                       "matched_budget": "Same training clips and passes; actual steps differ with client batch rounding",
                       "deterministic_algorithms": True,
                       "fedprox": "mu/2 times squared distance of all trainable parameters to fixed round server; same SGD as other methods"},
        "methods": [], "complete": False,
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for method in args.methods:
        print(f"START {method}", flush=True)

        def on_round(entry):
            with (args.output_dir / f"{method}_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            metric = entry["development"]["pooled"]
            print(f"{method} round={entry['round']} seconds={entry['seconds']:.1f} "
                  f"development_accuracy={metric['accuracy']:.4f} macro_f1={metric['macro_f1_all_declared_classes']:.4f}", flush=True)

        result, artifacts = run_method(factory, train, development, method=method, rounds=args.rounds,
                                        local_epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr,
                                        seed=args.seed, device=args.device, num_classes=15,
                                        proximal_mu=args.proximal_mu, on_round=on_round)
        result["artifacts"] = {}
        for endpoint, (predictions, checkpoint) in artifacts.items():
            checkpoint["protocol"] = protocol
            checkpoint["source_sha256"] = report["source_sha256"]
            for suffix in ("npz", "pt"):
                path = args.output_dir / f"{method}_{endpoint}.{suffix}"
                if suffix == "npz":
                    np.savez_compressed(path, **predictions)
                else:
                    torch.save(checkpoint, path)
                result["artifacts"][path.name] = fl.sha256(path)
        report["methods"].append(result)
        result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["complete"] = True
    result_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("COMPLETE: development-only results, not publication test estimates", flush=True)


if __name__ == "__main__":
    main()
