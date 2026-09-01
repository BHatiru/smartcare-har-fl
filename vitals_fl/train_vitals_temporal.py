"""
Federated Vitals Anomaly Detection — Temporal 1D-CNN
=====================================================

Replaces the point-wise MLP with a temporal model that processes
entire vitals sequences (pad/truncate to fixed window).

Key improvements over federated_2.py:
  1. Temporal model (1D-CNN) instead of point-wise MLP
  2. NaN handling via zero-fill + mask channels (not row dropping)
  3. Subject-aware non-IID split (Dirichlet α) instead of IID array_split
  4. Proper train/test separation (subject-level)
  5. Multiple FL strategies: FedAvg, FedProx, FedBN

Usage:
    python federated_v2/train_vitals_temporal.py [--mode centralized|fedavg|fedprox|fedbn|all]

Data:
    ntu_action_dataset/integrated_v2/*.parquet
"""

import os
import sys
import glob
import json
import time
import argparse
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ntu_action_dataset" / "integrated_v2"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_temporal"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────────
VITALS_CHANNELS = ["HR", "SpO2", "awRR"]            # 3 most informative vitals
MASK_CHANNELS   = ["HR_mask", "SpO2_mask", "awRR_mask"]  # corresponding masks
N_CHANNELS      = len(VITALS_CHANNELS) + len(MASK_CHANNELS)  # 6
SEQ_LEN         = 200              # pad/truncate to this length (100 seconds @ 0.5s dt)
N_CLASSES       = 6                # Bradycardia, Tachycardia, Normal, Low/High pressure, Nitroglycerine
CLASS_NAMES     = ["Bradycardia", "High pressure", "Low pressure",
                   "Nitroglycerine", "Normal", "Tachycardia"]

# FL settings (aligned with skeleton HAR setup)
NUM_CLIENTS     = 5
DIRICHLET_ALPHA = 0.5
LOCAL_EPOCHS    = 2
BATCH_SIZE      = 32
LR              = 0.001

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPROCESSING — Sequence-level with temporal windows
# ═════════════════════════════════════════════════════════════════════════════

def load_sequences():
    """Load integrated_v2, group by sequence, build temporal tensors."""
    print("Loading data...")
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    iv2 = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {iv2.shape[0]:,} rows, {iv2.sequence_id.nunique()} sequences, "
          f"{iv2.subject_id.nunique()} subjects")

    # Sort by sequence + time
    iv2 = iv2.sort_values(["sequence_id", "timestamp_s"]).reset_index(drop=True)

    sequences = []
    labels = []
    subject_ids = []
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    for seq_id, grp in iv2.groupby("sequence_id"):
        # Extract vitals + masks
        vitals = grp[VITALS_CHANNELS].values.astype(np.float32)   # (T, 3)
        masks  = grp[MASK_CHANNELS].values.astype(np.float32)     # (T, 3)

        # Zero-fill NaN (mask channels indicate where data is valid)
        vitals = np.nan_to_num(vitals, nan=0.0)

        # Combine: (T, 6)
        combined = np.concatenate([vitals, masks], axis=1)

        # Pad or truncate to SEQ_LEN
        T = combined.shape[0]
        if T >= SEQ_LEN:
            # Center crop
            start = (T - SEQ_LEN) // 2
            combined = combined[start:start + SEQ_LEN]
        else:
            # Zero-pad at end
            pad = np.zeros((SEQ_LEN - T, N_CHANNELS), dtype=np.float32)
            combined = np.concatenate([combined, pad], axis=0)

        sequences.append(combined)
        labels.append(class_to_idx[grp["class_name"].iloc[0]])
        subject_ids.append(grp["subject_id"].iloc[0])

    X = np.stack(sequences)                      # (N, SEQ_LEN, 6)
    X = X.transpose(0, 2, 1)                     # (N, 6, SEQ_LEN) — channels first for Conv1d
    y = np.array(labels, dtype=np.int64)
    subjects = np.array(subject_ids)

    print(f"  Tensor: X={X.shape}, y={y.shape}")
    print(f"  Classes: {dict(zip(CLASS_NAMES, np.bincount(y)))}")

    # ── Normalize vitals channels (per-channel, on non-zero values) ──
    for ch in range(len(VITALS_CHANNELS)):
        vals = X[:, ch, :]
        mask_vals = X[:, ch + len(VITALS_CHANNELS), :]  # corresponding mask
        valid = vals[mask_vals > 0.5]
        if len(valid) > 0:
            mu, sigma = valid.mean(), valid.std() + 1e-8
            X[:, ch, :] = np.where(mask_vals > 0.5, (vals - mu) / sigma, 0.0)

    return X, y, subjects


def subject_split(X, y, subjects, test_frac=0.2, seed=42):
    """Subject-aware train/test split."""
    rng = np.random.RandomState(seed)
    unique_subj = np.unique(subjects)
    rng.shuffle(unique_subj)
    n_test = int(len(unique_subj) * test_frac)
    test_subj = set(unique_subj[:n_test])

    train_mask = np.array([s not in test_subj for s in subjects])
    test_mask = ~train_mask

    print(f"  Train: {train_mask.sum()} seqs ({len(unique_subj) - n_test} subjects), "
          f"Test: {test_mask.sum()} seqs ({n_test} subjects)")

    return (X[train_mask], y[train_mask], subjects[train_mask],
            X[test_mask],  y[test_mask],  subjects[test_mask])


def dirichlet_split(X_train, y_train, subjects_train, n_clients, alpha, seed=42):
    """
    Label-based Dirichlet non-IID split (standard FL method).

    For each class, draws a Dirichlet(α) distribution to decide what
    fraction of that class's samples go to each client. Subjects are
    kept intact (all sequences from one subject go to the same client).
    Lower α = more heterogeneous.
    """
    rng = np.random.RandomState(seed)

    # First, group subjects by their dominant class
    subject_class = {}
    for subj in np.unique(subjects_train):
        mask = subjects_train == subj
        classes = y_train[mask]
        # Assign subject to most frequent class among their sequences
        subject_class[subj] = np.bincount(classes, minlength=N_CLASSES).argmax()

    # For each class, partition its subjects across clients via Dirichlet
    subject_assignment = {}
    for cls in range(N_CLASSES):
        cls_subjects = [s for s, c in subject_class.items() if c == cls]
        rng.shuffle(cls_subjects)
        if len(cls_subjects) == 0:
            continue
        # Draw Dirichlet proportions for this class
        proportions = rng.dirichlet([alpha] * n_clients)
        # Clip to ensure minimum representation
        proportions = np.maximum(proportions, 0.05)
        proportions = proportions / proportions.sum()
        # Split subjects according to proportions
        split_points = (np.cumsum(proportions)[:-1] * len(cls_subjects)).astype(int)
        splits = np.array_split(cls_subjects, split_points)
        for cid, subj_list in enumerate(splits):
            for s in subj_list:
                subject_assignment[s] = cid

    # Build per-client indices
    client_indices = {i: [] for i in range(n_clients)}
    for idx, s in enumerate(subjects_train):
        cid = subject_assignment.get(s, 0)
        client_indices[cid].append(idx)

    print(f"  Client sizes: {[len(v) for v in client_indices.values()]}")
    for cid in range(n_clients):
        idxs = client_indices[cid]
        dist = np.bincount(y_train[idxs], minlength=N_CLASSES)
        print(f"    Client {cid}: {len(idxs)} seqs, class dist={dist.tolist()}")

    return client_indices


# ═════════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL 1D-CNN MODEL
# ═════════════════════════════════════════════════════════════════════════════

class VitalsTemporalCNN(nn.Module):
    """
    1D-CNN for vitals time series classification.

    Input:  (batch, 6, 200)  — 3 vitals + 3 masks, 200 timesteps
    Output: (batch, 6)       — 6 class logits
    """
    def __init__(self, in_channels=N_CHANNELS, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (6, 200) → (32, 100)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: (32, 100) → (64, 50)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: (64, 50) → (128, 25)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4: (128, 25) → (128, 12)
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (128, 1)
            nn.Flatten(),              # (128,)
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return correct / total, np.array(all_preds), np.array(all_labels)


def train_one_epoch(model, loader, optimizer, criterion, device="cpu",
                    proximal_mu=0.0, global_params=None):
    """Train for one epoch. If proximal_mu > 0, adds FedProx term."""
    model.train()
    epoch_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        # FedProx proximal term
        if proximal_mu > 0.0 and global_params is not None:
            prox = 0.0
            for p, gp in zip(model.parameters(), global_params):
                prox += ((p - gp) ** 2).sum()
            loss += (proximal_mu / 2.0) * prox

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
    return epoch_loss / len(loader.dataset)


# ═════════════════════════════════════════════════════════════════════════════
# 4. CENTRALIZED TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_centralized(X_train, y_train, X_test, y_test, epochs=50):
    print("\n" + "=" * 60)
    print(" CENTRALIZED TRAINING")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VitalsTemporalCNN().to(device)
    print(f"  Model params: {count_parameters(model):,}")
    print(f"  Device: {device}")

    train_loader = make_loader(X_train, y_train)
    test_loader  = make_loader(X_test, y_test, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "test_acc": [], "loss": []}
    best_acc = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        train_acc, _, _ = evaluate(model, train_loader, device)
        test_acc, _, _  = evaluate(model, test_loader, device)

        history["loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "centralized_best.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}  train={train_acc:.1%}  test={test_acc:.1%}")

    print(f"  Best test accuracy: {best_acc:.1%}")

    # Final evaluation with confusion matrix
    model.load_state_dict(torch.load(OUTPUT_DIR / "centralized_best.pt", weights_only=True))
    _, preds, labels = evaluate(model, test_loader, device)

    return {"best_acc": best_acc, "history": history, "preds": preds, "labels": labels}


# ═════════════════════════════════════════════════════════════════════════════
# 5. FEDERATED TRAINING (Manual simulation — no Flower dependency needed)
# ═════════════════════════════════════════════════════════════════════════════

def get_model_params(model):
    return [p.clone().detach() for p in model.parameters()]


def set_model_params(model, params):
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(new_p)


def fedavg_aggregate(client_params_list, client_sizes):
    """Weighted average of client parameters."""
    total = sum(client_sizes)
    avg_params = []
    for param_idx in range(len(client_params_list[0])):
        weighted_sum = sum(
            client_params_list[c][param_idx] * (client_sizes[c] / total)
            for c in range(len(client_params_list))
        )
        avg_params.append(weighted_sum)
    return avg_params


def fedbn_aggregate(client_state_dicts, client_sizes):
    """FedAvg but skip BatchNorm parameters (keep them local)."""
    total = sum(client_sizes)
    n_clients = len(client_state_dicts)
    avg_state = OrderedDict()

    for key in client_state_dicts[0]:
        if "bn" in key or "num_batches_tracked" in key:
            # BN params stay local — just use first client's as placeholder
            avg_state[key] = client_state_dicts[0][key].clone()
        else:
            avg_state[key] = sum(
                client_state_dicts[c][key] * (client_sizes[c] / total)
                for c in range(n_clients)
            )
    return avg_state


def train_federated(X_train, y_train, subjects_train, X_test, y_test,
                    mode="fedavg", num_rounds=30, mu=0.01):
    """
    Federated training with manual simulation.
    mode: 'fedavg' | 'fedprox' | 'fedbn'
    """
    mode_label = mode.upper()
    if mode == "fedprox":
        mode_label = f"FEDPROX (μ={mu})"

    print(f"\n{'=' * 60}")
    print(f" FEDERATED TRAINING — {mode_label}")
    print(f" {NUM_CLIENTS} clients, Dirichlet α={DIRICHLET_ALPHA}, {num_rounds} rounds")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Split data
    client_indices = dirichlet_split(X_train, y_train, subjects_train,
                                     NUM_CLIENTS, DIRICHLET_ALPHA)

    # Create client data loaders
    client_loaders = {}
    client_sizes = {}
    for cid in range(NUM_CLIENTS):
        idxs = client_indices[cid]
        client_loaders[cid] = make_loader(X_train[idxs], y_train[idxs])
        client_sizes[cid] = len(idxs)

    test_loader = make_loader(X_test, y_test, shuffle=False)

    # Initialize global model
    global_model = VitalsTemporalCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # For FedBN: keep per-client state dicts
    client_state_dicts = {cid: None for cid in range(NUM_CLIENTS)}

    history = {"global_acc": [], "client_accs": []}
    best_acc = 0

    for rnd in range(1, num_rounds + 1):
        round_params = []
        round_states = []
        round_sizes = []
        client_accs_round = []

        for cid in range(NUM_CLIENTS):
            # Create local model
            local_model = VitalsTemporalCNN().to(device)

            if mode == "fedbn" and client_state_dicts[cid] is not None:
                # Load global non-BN params + local BN params
                global_state = global_model.state_dict()
                local_state = client_state_dicts[cid]
                merged = OrderedDict()
                for key in global_state:
                    if "bn" in key or "num_batches_tracked" in key:
                        merged[key] = local_state[key]
                    else:
                        merged[key] = global_state[key]
                local_model.load_state_dict(merged)
            else:
                local_model.load_state_dict(global_model.state_dict())

            # FedProx: save global params for proximal term
            global_params = None
            if mode == "fedprox":
                global_params = [p.clone().detach().to(device) for p in global_model.parameters()]

            # Local training
            optimizer = optim.Adam(local_model.parameters(), lr=LR, weight_decay=1e-4)
            for _ in range(LOCAL_EPOCHS):
                train_one_epoch(local_model, client_loaders[cid], optimizer, criterion,
                                device, proximal_mu=mu if mode == "fedprox" else 0.0,
                                global_params=global_params)

            # Collect params
            round_params.append(get_model_params(local_model))
            round_states.append(OrderedDict({k: v.cpu() for k, v in local_model.state_dict().items()}))
            round_sizes.append(client_sizes[cid])

            # Save BN state for FedBN
            if mode == "fedbn":
                client_state_dicts[cid] = round_states[-1]

            # Client accuracy
            c_acc, _, _ = evaluate(local_model, client_loaders[cid], device)
            client_accs_round.append(c_acc)

        # Aggregate
        if mode == "fedbn":
            avg_state = fedbn_aggregate(round_states, round_sizes)
            global_model.load_state_dict(avg_state)
        else:
            avg_params = fedavg_aggregate(round_params, round_sizes)
            set_model_params(global_model, avg_params)

        # Evaluate global model
        global_acc, _, _ = evaluate(global_model, test_loader, device)
        history["global_acc"].append(global_acc)
        history["client_accs"].append(client_accs_round)

        if global_acc > best_acc:
            best_acc = global_acc
            torch.save(global_model.state_dict(), OUTPUT_DIR / f"{mode}_best.pt")

        if rnd % 5 == 0 or rnd == 1:
            client_str = " ".join(f"{a:.0%}" for a in client_accs_round)
            print(f"  R{rnd:3d}: global={global_acc:.1%}  clients=[{client_str}]")

    print(f"  Best global accuracy: {best_acc:.1%}")

    # Final evaluation
    global_model.load_state_dict(torch.load(OUTPUT_DIR / f"{mode}_best.pt", weights_only=True))
    _, preds, labels = evaluate(global_model, test_loader, device)

    return {"best_acc": best_acc, "history": history, "preds": preds, "labels": labels}


# ═════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(results_dict):
    """Generate comparison plots."""
    from sklearn.metrics import confusion_matrix as cm_fn

    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Vitals Anomaly Detection — Temporal 1D-CNN", fontsize=14, fontweight="bold")

    # ── Accuracy curves ──
    ax = axes[0]
    for name, res in results_dict.items():
        hist = res["history"]
        if "test_acc" in hist:
            ax.plot(hist["test_acc"], label=f"{name} (test)")
        if "global_acc" in hist:
            ax.plot(hist["global_acc"], label=f"{name} (global)")
    ax.set_xlabel("Epoch / Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Training Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Confusion matrices ──
    for i, (name, res) in enumerate(results_dict.items()):
        ax = axes[i + 1]
        cm = cm_fn(res["labels"], res["preds"])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(N_CLASSES))
        ax.set_xticklabels([c[:6] for c in CLASS_NAMES], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(N_CLASSES))
        ax.set_yticklabels([c[:6] for c in CLASS_NAMES], fontsize=7)
        ax.set_title(f"{name}\n{res['best_acc']:.1%}", fontsize=11, fontweight="bold")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=7,
                        color="white" if cm[r, c] > cm.max() * 0.5 else "black")

    plt.tight_layout()
    path = OUTPUT_DIR / "results_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path}")

    # ── Summary bar chart ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor("white")
    names = list(results_dict.keys())
    accs = [results_dict[n]["best_acc"] for n in names]
    colors = ["#1565c0", "#43a047", "#e65100", "#6a1b9a", "#d32f2f"]
    bars = ax2.bar(names, accs, color=colors[:len(names)])
    ax2.set_ylabel("Best Test Accuracy")
    ax2.set_title("Vitals Classification — Method Comparison", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = OUTPUT_DIR / "accuracy_comparison.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Saved: {path2}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Federated Vitals Temporal Training")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["centralized", "fedavg", "fedprox", "fedbn", "all"],
                        help="Training mode")
    parser.add_argument("--rounds", type=int, default=30, help="FL rounds")
    parser.add_argument("--epochs", type=int, default=50, help="Centralized epochs")
    parser.add_argument("--mu", type=float, default=0.01, help="FedProx proximal weight")
    args = parser.parse_args()

    print("=" * 60)
    print(" VITALS ANOMALY DETECTION — TEMPORAL 1D-CNN")
    print(f" Mode: {args.mode}")
    print("=" * 60)

    # Load and preprocess
    X, y, subjects = load_sequences()
    X_train, y_train, subj_train, X_test, y_test, subj_test = subject_split(X, y, subjects)

    results = {}
    t0 = time.time()

    if args.mode in ("centralized", "all"):
        res = train_centralized(X_train, y_train, X_test, y_test, epochs=args.epochs)
        results["Centralized"] = res

    if args.mode in ("fedavg", "all"):
        res = train_federated(X_train, y_train, subj_train, X_test, y_test,
                              mode="fedavg", num_rounds=args.rounds)
        results["FedAvg"] = res

    if args.mode in ("fedprox", "all"):
        res = train_federated(X_train, y_train, subj_train, X_test, y_test,
                              mode="fedprox", num_rounds=args.rounds, mu=args.mu)
        results["FedProx"] = res

    if args.mode in ("fedbn", "all"):
        res = train_federated(X_train, y_train, subj_train, X_test, y_test,
                              mode="fedbn", num_rounds=args.rounds)
        results["FedBN"] = res

    elapsed = time.time() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print(" RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, res in results.items():
        print(f"  {name:15s}: {res['best_acc']:.1%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Generate plots
    if len(results) > 1:
        plot_results(results)

    # Save results JSON
    summary = {name: {"best_acc": float(res["best_acc"])} for name, res in results.items()}
    summary["config"] = {
        "seq_len": SEQ_LEN, "n_channels": N_CHANNELS, "n_classes": N_CLASSES,
        "num_clients": NUM_CLIENTS, "dirichlet_alpha": DIRICHLET_ALPHA,
        "local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
