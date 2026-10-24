"""Fixed-budget FedAvg/FedBN/local comparison; historical notebooks stay unchanged.

Run with ``python -m demo1_fed_skeleton.corrected_fl --help``. FedBN keeps every
BatchNorm parameter and buffer on its owning client. No pooled BN calibration,
warm start, test-selected checkpoints, or test-time adaptation is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models.stgcn import build_model, count_parameters


@dataclass
class Shard:
    x: torch.Tensor
    y: torch.Tensor
    sequence_ids: np.ndarray
    subject_ids: np.ndarray

    def __len__(self):
        return len(self.y)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batchnorm_keys(model):
    """Include affine parameters, running statistics and counters, by ownership."""
    return {
        f"{name}.{key}" if name else key
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        for key in module.state_dict()
    }


def state_copy(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def receive_shared(model, server_state, local_bn=False):
    local_keys = batchnorm_keys(model) if local_bn else set()
    own_state = model.state_dict()
    model.load_state_dict({key: own_state[key] if key in local_keys else value
                           for key, value in server_state.items()}, strict=True)


def aggregate(server_state, client_states, sizes, excluded=()):
    """Sample-weight floats; retain local BN; take max of integer BN counters."""
    if not client_states or len(client_states) != len(sizes) or min(sizes) <= 0:
        raise ValueError("Aggregation needs nonempty clients with positive training sizes")
    expected_keys = set(server_state) - set(excluded)
    if any(set(state) != expected_keys for state in client_states):
        raise ValueError("Client/server state keys differ")
    weights = np.asarray(sizes, dtype=np.float64) / sum(sizes)
    result = {}
    for key, reference in server_state.items():
        if key in excluded:
            result[key] = reference.clone()
        elif reference.is_floating_point():
            result[key] = sum(state[key].to(torch.float64) * float(weight)
                              for state, weight in zip(client_states, weights)).to(reference.dtype)
        elif key.endswith("num_batches_tracked"):
            result[key] = torch.stack([state[key] for state in client_states]).max(dim=0).values
        else:
            if any(not torch.equal(reference, state[key]) for state in client_states):
                raise ValueError(f"Unequal non-floating structural buffer: {key}")
            result[key] = reference.clone()
    return result


def train_client(model, shard, *, device, batch_size, epochs, lr, seed, momentum=.9,
                 weight_decay=5e-4, proximal_mu=0., reference_state=None, on_step=None):
    """This API receives training data only; optimizer state resets each round."""
    if proximal_mu < 0 or (proximal_mu and reference_state is None):
        raise ValueError("FedProx requires nonnegative mu and a fixed server reference")
    model.to(device).train()
    reference = ({name: reference_state[name].detach().to(device).clone()
                  for name, parameter in model.named_parameters() if parameter.requires_grad}
                 if proximal_mu else {})
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(shard.x, shard.y), batch_size=batch_size,
                        shuffle=True, generator=generator, drop_last=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    losses, penalties, correct, seen, steps = 0., 0., 0, 0, 0
    started = time.perf_counter()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            cross_entropy = nn.functional.cross_entropy(logits, y)
            penalty = (.5 * proximal_mu * sum((parameter - reference[name]).square().sum()
                       for name, parameter in model.named_parameters() if parameter.requires_grad)
                       if proximal_mu else cross_entropy.new_zeros(()))
            loss = cross_entropy + penalty
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            loss.backward()
            optimizer.step()
            losses += float(cross_entropy.detach()) * len(y)
            penalties += float(penalty.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
            steps += 1
            if on_step:
                on_step({"optimizer_steps": steps, "examples_processed": seen,
                         "total_examples": len(shard) * epochs, "training_loss": losses / seen,
                         "seconds": time.perf_counter() - started})
    model.cpu()
    return {"online_training_accuracy": correct / seen, "training_loss": losses / seen,
            "mean_proximal_penalty": penalties / seen, "optimizer_steps": steps,
            "examples_processed": seen, "seconds": time.perf_counter() - started}


def score_client(model, heldout, *, device, batch_size, num_classes):
    """Evaluate a fixed final model; no gradients, optimizer or BN adaptation."""
    model.to(device).eval()
    loader = DataLoader(TensorDataset(heldout.x, heldout.y), batch_size=batch_size,
                        shuffle=False)
    logits = []
    with torch.inference_mode():
        for x, _ in loader:
            logits.append(model(x.to(device)).cpu())
    model.cpu()
    logits = torch.cat(logits).numpy()
    predictions = logits.argmax(1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (heldout.y.numpy(), predictions), 1)
    return matrix, logits


def summarize(matrix):
    matrix = np.asarray(matrix)
    support, predicted = matrix.sum(1), matrix.sum(0)
    true_positive = matrix.diagonal()
    recall = np.divide(true_positive, support, out=np.zeros(len(support), float), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(len(support), float), where=predicted > 0)
    f1 = np.divide(2 * true_positive, support + predicted,
                   out=np.zeros(len(support), float), where=(support + predicted) > 0)
    return {"n": int(matrix.sum()), "accuracy": float(true_positive.sum() / matrix.sum()),
            "macro_f1_all_declared_classes": float(f1.mean()),
            "per_class_support": support.tolist(), "per_class_recall": recall.tolist(),
            "per_class_precision": precision.tolist(), "per_class_f1": f1.tolist(),
            "fall_recall": float(recall[0]) if support[0] else None,
            "fall_precision": float(precision[0]) if predicted[0] else None,
            "confusion_matrix": matrix.tolist()}


def run_method(model_factory, train_shards, test_shards, *, method, rounds, local_epochs,
               batch_size, lr, seed, device, num_classes):
    """All methods use one initial state and evaluate only after fixed training."""
    if method not in {"fedavg", "fedbn", "local"}:
        raise ValueError(f"Unknown method: {method}")
    if len(train_shards) != len(test_shards) or not train_shards:
        raise ValueError("Every client requires separate train and heldout shards")
    if rounds < 1 or local_epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("Round, epoch, batch and learning-rate budgets must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    server = model_factory()
    clients = [copy.deepcopy(server) for _ in train_shards]
    server_state = state_copy(server)
    local_keys = batchnorm_keys(server) if method == "fedbn" else set()
    history = []
    start = time.perf_counter()
    for round_index in range(1, rounds + 1):
        current_lr = lr * (.01 if round_index > 35 else .1 if round_index > 20 else 1.)
        training = []
        for cid, (model, shard) in enumerate(zip(clients, train_shards)):
            if method != "local":
                receive_shared(model, server_state, local_bn=method == "fedbn")
            # Match stochastic local passes across methods; order cannot consume another client's RNG.
            client_seed = seed + 100_000 * round_index + cid
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(client_seed)
            training.append(train_client(model, shard, device=device, batch_size=batch_size,
                                         epochs=local_epochs, lr=current_lr, seed=client_seed))
        if method != "local":
            updates = [{key: tensor for key, tensor in state_copy(model).items() if key not in local_keys}
                       for model in clients]
            server_state = aggregate(server_state, updates,
                                     [len(s) for s in train_shards], excluded=local_keys)
        history.append({"round": round_index, "lr": current_lr, "clients_training": training})
    train_seconds = time.perf_counter() - start
    matrices, all_logits, all_ids, all_y, all_clients = [], [], [], [], []
    for cid, (model, heldout) in enumerate(zip(clients, test_shards)):
        if method != "local":
            receive_shared(model, server_state, local_bn=method == "fedbn")
        matrix, logits = score_client(model, heldout, device=device, batch_size=batch_size,
                                      num_classes=num_classes)
        matrices.append(matrix)
        all_logits.append(logits)
        all_ids.append(heldout.sequence_ids)
        all_y.append(heldout.y.numpy())
        all_clients.append(np.full(len(heldout), cid, dtype=np.int64))
    client_metrics = [summarize(m) for m in matrices]
    result = {"method": method, "selection": "fixed final communication round; test scored once",
              "prediction_rule": {"fedavg": "same final global model for every test client",
                                  "fedbn": "final shared weights plus each client's persistent BN",
                                  "local": "independent final local model per client"}[method],
              "heldout_client_metrics": client_metrics,
              "heldout_sample_weighted": summarize(np.sum(matrices, axis=0)),
              "heldout_unweighted_mean_client_accuracy": float(np.mean([m["accuracy"] for m in client_metrics])),
              "heldout_worst_client_accuracy": min(m["accuracy"] for m in client_metrics),
              "train_seconds": train_seconds, "total_seconds": time.perf_counter() - start,
              "history_training_only": history}
    predictions = {"sequence_ids": np.concatenate(all_ids), "client_ids": np.concatenate(all_clients),
                   "y": np.concatenate(all_y), "logits": np.concatenate(all_logits)}
    checkpoint = {"server": server_state, "clients": [state_copy(m) for m in clients],
                  "method": method, "round": rounds, "seed": seed}
    return result, predictions, checkpoint


def load_protocol(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("purpose") not in {"research", "synthetic_smoke"}:
        raise ValueError("Expected corrected FL protocol schema 1 and explicit purpose")
    if protocol["graph"] not in {"coco", "ntu"}:
        raise ValueError("Unsupported skeleton graph")
    expected = (2, 100, 17 if protocol["graph"] == "coco" else 25, 3)
    if tuple(protocol["sample_shape"]) != expected:
        raise ValueError("Requires M=2,T=100,C=3 and graph-matched joint count")
    classes = len(protocol["label_names"])
    if classes not in {10, 15} or len(set(protocol["label_names"])) != classes:
        raise ValueError("Expected declared unique 10- or 15-class medical labels")
    from .prepare_corrected_fl import class_contract
    mapping, names = class_contract(classes)
    if protocol["label_names"] != names or protocol["ntu_zero_based_ids"] != list(mapping):
        raise ValueError("Class ordering must match the explicit 10/15-class NTU contract")
    shards = {}
    for split in ("train", "test"):
        entry = protocol["files"][split]
        path = directory / f"{split}.npz"
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {split}")
        with np.load(path, allow_pickle=False) as data:
            required = {"x", "y", "sequence_ids", "subject_ids", "client_ids"}
            if not required.issubset(data.files):
                raise ValueError(f"Missing identities or tensor arrays: {split}")
            x, y = data["x"], data["y"]
            ids, subjects, clients = data["sequence_ids"], data["subject_ids"], data["client_ids"]
        if (x.shape != (len(y), *expected) or len(y) != entry["samples"] or
                y.ndim != 1 or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= classes)):
            raise ValueError(f"Invalid data/label shape or class range: {split}")
        if not np.isfinite(x).all() or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Non-finite inputs or duplicate sequence IDs: {split}")
        if any(a.shape != (len(y),) for a in (ids, subjects, clients)):
            raise ValueError(f"Identity vectors have wrong lengths: {split}")
        if set(clients.tolist()) != set(range(protocol["num_clients"])):
            raise ValueError(f"Client IDs must cover every declared client: {split}")
        shards[split] = [Shard(torch.from_numpy(x[clients == cid].astype(np.float32)),
                               torch.from_numpy(y[clients == cid].astype(np.int64)),
                               ids[clients == cid], subjects[clients == cid])
                           for cid in range(protocol["num_clients"])]
    train_ids = set(np.concatenate([s.sequence_ids for s in shards["train"]]).tolist())
    test_ids = set(np.concatenate([s.sequence_ids for s in shards["test"]]).tolist())
    train_subjects = set(np.concatenate([s.subject_ids for s in shards["train"]]).tolist())
    test_subjects = set(np.concatenate([s.subject_ids for s in shards["test"]]).tolist())
    if train_ids & test_ids or train_subjects & test_subjects:
        raise ValueError("Train/test sequence and subject identities must be disjoint")
    return protocol, shards["train"], shards["test"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["fedavg", "fedbn", "local"],
                        default=["fedavg", "fedbn", "local"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Use a new empty output directory; historical artifacts are never overwritten")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("Methods must be unique")
    torch.set_num_threads(args.threads)
    protocol, train, test = load_protocol(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factory = lambda: build_model(num_classes=len(protocol["label_names"]), in_channels=3,
                                  base_channels=64, num_stages=6, num_person=2,
                                  dropout=.3, graph=protocol["graph"])
    ref = factory()
    report = {"schema_version": 1, "purpose": protocol["purpose"],
              "accuracy_claim_allowed": protocol["purpose"] == "research",
              "protocol": protocol, "configuration": {k: str(v) if isinstance(v, Path) else v
                                                       for k, v in vars(args).items()},
              "torch_version": torch.__version__, "parameters": count_parameters(ref),
              "bn_state_keys": len(batchnorm_keys(ref)),
              "optimizer": {"name": "SGD", "momentum": .9, "weight_decay": 5e-4},
              "learning_rate_schedule": "base through R20; base*0.1 R21-35; base*0.01 R36 onward",
              "source_sha256": {"runner": sha256(__file__),
                                "model": sha256(Path(__file__).parent / "models/stgcn.py")},
              "invariants": {"central_warm_start": False, "pooled_bn_calibration": False,
                             "test_checkpoint_selection": False, "client_participation": "all",
                             "fedbn_transmits_bn_state": False,
                             "optimizer_state": "reset each client round",
                             "integer_bn_counter_aggregation": "maximum for FedAvg",
                             "client_definition": "simulated label-skew cohorts, not observed hospitals"},
              "methods": []}
    for method in args.methods:
        result, predictions, checkpoint = run_method(
            factory, train, test, method=method, rounds=args.rounds, local_epochs=args.local_epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device,
            num_classes=len(protocol["label_names"]))
        np.savez_compressed(args.output_dir / f"{method}_final_predictions.npz", **predictions)
        torch.save(checkpoint, args.output_dir / f"{method}_final.pt")
        report["methods"].append(result)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{protocol['purpose']} {method}: {result['total_seconds']:.2f}s; "
              f"final heldout n={result['heldout_sample_weighted']['n']}", flush=True)


if __name__ == "__main__":
    main()
