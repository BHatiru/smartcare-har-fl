"""
train_federated.py — Main entry-point for Demo 1 federated skeleton HAR.

Runs a Flower simulation (``flwr.simulation``) with K virtual clients,
each backed by its own NPZ shard.

Usage
-----
# 1) Prepare data (synthetic example)
python prepare_clients.py --synthetic --num-clients 5 --split dirichlet --alpha 0.3

# 2) Train — FedAvg baseline
python train_federated.py --data-dir data/fed_skeleton --num-rounds 30

# 3) Train — FedBN (BN params kept local)
python train_federated.py --data-dir data/fed_skeleton --num-rounds 30 --mode fedbn

# 4) Train — Clustered FedBN
python train_federated.py --data-dir data/fed_skeleton --num-rounds 30 \
       --mode cluster --num-clusters 2 --cluster-every 5

# 5) Export final model to ONNX
python export_onnx.py --checkpoint outputs/global_model.pt --output outputs/model.onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.stgcn import build_model, count_parameters  # noqa: E402
from fl.client import SkeletonClient                      # noqa: E402
from fl.strategy_fsar import FSARStrategy                  # noqa: E402
from prepare_medical_data import MEDICAL_LABELS            # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_federated")


# ─────────────────── centralized test evaluation ─────────────────────────────

def evaluate_global(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, list, list]:
    """Evaluate on global test set, returning loss, acc, preds, labels."""
    model.eval()
    correct, total, cum_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            cum_loss += criterion(logits, yb).item() * yb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(yb.cpu().tolist())
    return cum_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


# ─────────────────── client factory for Flower simulation ────────────────────

def make_client_fn(
    data_dir: Path,
    num_classes: int,
    in_channels: int,
    device: str,
    local_bn: bool,
    epochs_per_round: int,
    lr: float,
    batch_size: int,
    mu: float,
    model_kwargs: dict,
):
    """Return a factory ``fn(context) -> fl.client.Client``.

    Supports both the legacy ``cid: str`` and the new ``Context`` signatures.
    """

    def client_fn(context):
        # Flower >= 1.8 passes a Context object; extract cid from it
        if hasattr(context, "node_config"):
            cid = str(context.node_config.get("partition-id", 0))
        else:
            cid = str(context)  # legacy fallback

        client_path = data_dir / f"client_{cid}.npz"
        return SkeletonClient(
            cid=int(cid),
            data_path=client_path,
            num_classes=num_classes,
            in_channels=in_channels,
            device=device,
            local_bn=local_bn,
            epochs_per_round=epochs_per_round,
            lr=lr,
            batch_size=batch_size,
            mu=mu,
            model_kwargs=model_kwargs,
        ).to_client()

    return client_fn


# ─────────────────── main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Federated skeleton HAR training")
    parser.add_argument("--data-dir", type=str, default="data/fed_skeleton")
    parser.add_argument("--num-rounds", type=int, default=30)
    parser.add_argument("--num-clients", "-K", type=int, default=None,
                        help="Auto-detected from data dir if omitted.")
    parser.add_argument("--mode", choices=["fedavg", "fedbn", "cluster", "fedprox"],
                        default="fedbn")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-person", type=int, default=2,
                        help="Person slots (M dim). Use 2 for Demo 2 compat.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs-per-round", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.01,
                        help="FedProx proximal term coefficient (only used with --mode fedprox)")
    parser.add_argument("--graph", type=str, default="coco",
                        choices=["coco", "ntu"],
                        help="Skeleton graph: coco (17j) or ntu (25j)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-clusters", type=int, default=2)
    parser.add_argument("--cluster-every", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=6)
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag for output naming (auto-generated if omitted)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Experiment tag for file naming
    skeleton = '3d' if args.graph == 'ntu' else '2d'
    tag = args.tag or f"fl_{args.mode}_{skeleton}_{args.num_classes}cls"
    if args.mode == 'fedprox':
        tag_default = f"fl_fedprox_mu{args.mu}_{skeleton}_{args.num_classes}cls"
        tag = args.tag or tag_default

    # ── Detect number of clients ──
    if args.num_clients is None:
        client_files = sorted(data_dir.glob("client_*.npz"))
        num_clients = len(client_files)
        if num_clients == 0:
            raise FileNotFoundError(
                f"No client_*.npz found in {data_dir}. "
                "Run prepare_clients.py first.")
    else:
        num_clients = args.num_clients
    logger.info("Number of clients: %d", num_clients)

    # ── Model kwargs (shared across clients) ──
    model_kwargs = dict(
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        num_person=args.num_person,
        graph=args.graph,
    )

    device_str = args.device
    device = torch.device(device_str)
    local_bn = args.mode in ("fedbn", "cluster")
    mu = args.mu if args.mode == "fedprox" else 0.0

    # ── Build a reference model to get initial parameters ──
    ref_model = build_model(num_classes=args.num_classes,
                            in_channels=args.in_channels, **model_kwargs)
    logger.info("Model parameters: %s", f"{count_parameters(ref_model):,}")

    if local_bn:
        from fl.client import _split_param_names
        global_keys, local_keys = _split_param_names(ref_model)
        logger.info("FedBN mode — %d global params, %d local (BN) params",
                     len(global_keys), len(local_keys))
        init_ndarrays = [ref_model.state_dict()[k].cpu().numpy()
                         for k in global_keys]
    else:
        init_ndarrays = [v.cpu().numpy() for v in ref_model.state_dict().values()]

    initial_params = ndarrays_to_parameters(init_ndarrays)

    # ── Strategy ──
    strategy = FSARStrategy(
        mode=args.mode,
        initial_parameters=initial_params,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        num_clusters=args.num_clusters,
        cluster_every=args.cluster_every,
    )

    # ── Client factory ──
    client_fn = make_client_fn(
        data_dir=data_dir,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        device=device_str,
        local_bn=local_bn,
        epochs_per_round=args.epochs_per_round,
        lr=args.lr,
        batch_size=args.batch_size,
        mu=mu,
        model_kwargs=model_kwargs,
    )

    # ── Run simulation ──
    logger.info("Starting Flower simulation  |  mode=%s  rounds=%d  clients=%d  tag=%s",
                args.mode, args.num_rounds, num_clients, tag)

    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # ── Post-training: load aggregated weights into model ──
    if strategy.last_aggregated_ndarrays is not None:
        sd = ref_model.state_dict()
        if local_bn:
            from fl.client import _split_param_names
            g_keys, _ = _split_param_names(ref_model)
            for k, v in zip(g_keys, strategy.last_aggregated_ndarrays):
                sd[k] = torch.tensor(v)
        else:
            for k, v in zip(sd.keys(), strategy.last_aggregated_ndarrays):
                sd[k] = torch.tensor(v)
        ref_model.load_state_dict(sd)
        logger.info("Loaded aggregated weights from last round.")

    # ── Post-training: evaluate on global test set ──
    test_loss, test_acc = None, None
    preds, labels = [], []
    test_path = data_dir / "test.npz"
    if test_path.exists():
        logger.info("Evaluating final model on global test set \u2026")
        test_data = np.load(test_path)
        x_test = torch.from_numpy(test_data["x"].astype(np.float32))
        y_test = torch.from_numpy(test_data["y"].astype(np.int64))
        test_loader = DataLoader(TensorDataset(x_test, y_test),
                                 batch_size=args.batch_size)

        ref_model.to(device)
        test_loss, test_acc, preds, labels = evaluate_global(
            ref_model, test_loader, device)
        logger.info("Global test \u2014 loss: %.4f  acc: %.4f", test_loss, test_acc)

    total_time = time.time() - start_time

    # ── Save model checkpoint ──
    ckpt_path = out_dir / f"{tag}_global.pt"
    torch.save(ref_model.state_dict(), ckpt_path)
    logger.info("Model saved → %s", ckpt_path)

    # ── Per-class metrics and confusion matrix ──
    num_cls = args.num_classes
    label_names = MEDICAL_LABELS[:num_cls]
    per_class_acc = {}
    confusion = np.zeros((num_cls, num_cls), dtype=int)
    if preds and labels:
        class_correct = Counter()
        class_total = Counter()
        for p, l in zip(preds, labels):
            class_total[l] += 1
            confusion[l][p] += 1
            if p == l:
                class_correct[l] += 1
        for c in sorted(class_total.keys()):
            per_class_acc[int(c)] = {
                'name': label_names[c] if c < len(label_names) else f'class_{c}',
                'accuracy': round(class_correct[c] / max(class_total[c], 1), 4),
                'correct': class_correct[c],
                'total': class_total[c],
            }

        # Save confusion matrix CSV
        cm_path = out_dir / f"{tag}_confusion_matrix.csv"
        with open(cm_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([''] + label_names)
            for i, row in enumerate(confusion):
                writer.writerow([label_names[i]] + row.tolist())
        logger.info("Confusion matrix saved → %s", cm_path)

    # ── Save strategy round history with full metadata ──
    hist_path = out_dir / f"{tag}_results.json"
    results_obj = {
        'tag': tag,
        'mode': args.mode,
        'graph': args.graph,
        'num_classes': args.num_classes,
        'num_clients': num_clients,
        'num_rounds': args.num_rounds,
        'epochs_per_round': args.epochs_per_round,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'mu': mu,
        'base_channels': args.base_channels,
        'num_stages': args.num_stages,
        'num_person': args.num_person,
        'params': count_parameters(ref_model),
        'final_test_acc': round(test_acc, 4) if test_acc is not None else None,
        'final_test_loss': round(test_loss, 4) if test_loss is not None else None,
        'total_time_s': round(total_time, 1),
        'label_names': label_names,
        'per_class_acc': per_class_acc,
        'confusion_matrix': confusion.tolist() if preds else None,
        'round_history': strategy.history,
    }
    with open(hist_path, "w") as f:
        json.dump(results_obj, f, indent=2)
    logger.info("Results saved → %s", hist_path)

    # ── Print summary ──
    print("\n" + "═" * 60)
    print(f"  Federated Training Complete — {tag}")
    print("═" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Rounds:     {args.num_rounds}")
    print(f"  Clients:    {num_clients}")
    print(f"  Time:       {total_time / 60:.1f} min")
    if strategy.history:
        last = strategy.history[-1]
        print(f"  Final round {last['round']}:")
        print(f"    Global acc      : {last.get('global_acc', 'N/A'):.4f}")
        print(f"    Mean client acc : {last.get('mean_client_acc', 'N/A'):.4f}")
        print(f"    Worst client acc: {last.get('worst_client_acc', 'N/A'):.4f}")
    if test_acc is not None:
        print(f"  Global test acc:  {test_acc:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Results:    {hist_path}")
    if per_class_acc:
        print(f"\n  Per-class accuracy:")
        for c in sorted(per_class_acc.keys()):
            info = per_class_acc[c]
            print(f"    [{c:2d}] {info['name']:20s} {info['accuracy']:.4f}  "
                  f"({info['correct']}/{info['total']})")
    print("═" * 60)


if __name__ == "__main__":
    main()
