"""
prepare_clients.py — Split COCO-17 2D skeleton data into K federated clients.

Supports:
  - Subject-based non-IID split (each client gets entire subjects)
  - Dirichlet label-skew split (alpha controls heterogeneity)

Input : a single NPZ file with keys 'x' (N,M,T,V,C) and 'y' (N,)
        where C=3 (x,y,score), T=100, V=17, M=2  (pyskl convention)
Output: K per-client NPZ files  + 1 global test NPZ

If no real data is available, the script can generate synthetic skeletons
for quick prototyping (--synthetic flag).
"""

import argparse
import os
import numpy as np
from pathlib import Path


# ───────────────────────── synthetic data generator ──────────────────────────

# COCO-17 skeleton connectivity (parent joint for each joint index)
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # head
    (0, 5), (0, 6),                     # torso-top → shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12),                   # torso → hips
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
]

NUM_JOINTS = 17


def _random_skeleton_sequence(
    num_classes: int = 10,
    samples_per_class: int = 200,
    T: int = 100,
    num_person: int = 2,
    seed: int = 0,
) -> tuple:
    """Generate synthetic 2D skeleton sequences (random walks per class).

    Each class gets a different base pose + temporal pattern so that
    a classifier can learn to distinguish them.

    Format follows the pyskl convention used by Demo 2:
      (N, M, T, V, C)  with C=3 (x, y, score), V=17, M=num_person

    Returns
    -------
    x : ndarray  shape (N, M, T, 17, 3)
    y : ndarray  shape (N,)
    subjects : ndarray  shape (N,)   — simulated subject IDs
    """
    rng = np.random.RandomState(seed)
    N = num_classes * samples_per_class
    num_subjects = max(20, N // 50)  # ~50 samples per subject

    x_all, y_all, subj_all = [], [], []

    for cls_id in range(num_classes):
        # Base 2D pose unique to this class (centred around 0.5, 0.5)
        base_pose = rng.randn(2, NUM_JOINTS) * 0.05  # (2, V)
        # Class-specific temporal frequency
        freq = 0.5 + cls_id * 0.3

        for _ in range(samples_per_class):
            subj = rng.randint(0, num_subjects)
            # Temporal modulation
            t = np.linspace(0, 2 * np.pi * freq, T)  # (T,)
            modulation = np.stack([np.sin(t), np.cos(t)], axis=0)  # (2, T)

            # Skeleton = base_pose + subject jitter + temporal modulation + noise
            xy = (
                base_pose[:, np.newaxis, :] +                              # (2, 1, V)
                modulation[:, :, np.newaxis] * 0.04 +                      # (2, T, V)
                rng.randn(2, T, NUM_JOINTS) * 0.02 +                      # noise
                rng.randn(2, 1, NUM_JOINTS) * 0.01 * subj                 # subject bias
            )  # (2, T, V=17)

            # Build (M, T, V, C=3) — person 0 has the skeleton, rest zero-padded
            score = np.ones((1, T, NUM_JOINTS), dtype=np.float32)  # confidence = 1
            person0 = np.concatenate([xy, score], axis=0)          # (3, T, V)
            person0 = person0.transpose(1, 2, 0)                  # (T, V, 3)

            sample = np.zeros((num_person, T, NUM_JOINTS, 3), dtype=np.float32)
            sample[0] = person0  # only first person slot populated

            x_all.append(sample)
            y_all.append(cls_id)
            subj_all.append(subj)

    x = np.stack(x_all).astype(np.float32)  # (N, M, T, 17, 3)
    y = np.array(y_all, dtype=np.int64)
    subjects = np.array(subj_all, dtype=np.int64)
    return x, y, subjects


# ───────────────────── non-IID partitioning strategies ───────────────────────

def split_by_subject(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    num_clients: int,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Assign whole subjects to clients (natural non-IID)."""
    rng = np.random.RandomState(seed)
    unique_subjs = np.unique(subjects)
    rng.shuffle(unique_subjs)

    # Hold out subjects for global test set
    n_test_subj = max(1, int(len(unique_subjs) * test_ratio))
    test_subjs = set(unique_subjs[:n_test_subj])
    train_subjs = unique_subjs[n_test_subj:]

    # Split remaining subjects across clients round-robin
    client_subj_map = {i: [] for i in range(num_clients)}
    for idx, s in enumerate(train_subjs):
        client_subj_map[idx % num_clients].append(s)

    test_mask = np.isin(subjects, list(test_subjs))
    x_test, y_test = x[test_mask], y[test_mask]

    client_data = []
    for cid in range(num_clients):
        mask = np.isin(subjects, client_subj_map[cid])
        client_data.append((x[mask], y[mask]))

    return client_data, x_test, y_test


def split_dirichlet(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Dirichlet label-skew split (lower alpha → more heterogeneous)."""
    rng = np.random.RandomState(seed)
    num_classes = int(y.max()) + 1
    N = len(y)

    # Global train / test split (stratified)
    from collections import defaultdict
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[int(label)].append(i)

    train_indices, test_indices = [], []
    for cls, idxs in class_indices.items():
        rng.shuffle(idxs)
        n_test = max(1, int(len(idxs) * test_ratio))
        test_indices.extend(idxs[:n_test])
        train_indices.extend(idxs[n_test:])

    x_test, y_test = x[test_indices], y[test_indices]
    x_train, y_train = x[train_indices], y[train_indices]

    # Dirichlet allocation of training samples to clients
    label_indices = defaultdict(list)
    for i, label in enumerate(y_train):
        label_indices[int(label)].append(i)

    client_indices = {i: [] for i in range(num_clients)}
    for cls in range(num_classes):
        idxs = np.array(label_indices[cls])
        proportions = rng.dirichlet([alpha] * num_clients)
        # Balance proportions so every client gets at least something
        proportions = np.cumsum(proportions)
        proportions = proportions / proportions[-1]
        splits = np.split(idxs, (proportions[:-1] * len(idxs)).astype(int))
        for cid, chunk in enumerate(splits):
            client_indices[cid].extend(chunk.tolist())

    client_data = []
    for cid in range(num_clients):
        ci = np.array(client_indices[cid], dtype=int)
        if len(ci) == 0:
            client_data.append((np.zeros((0, *x_train.shape[1:]), dtype=x_train.dtype),
                                np.zeros((0,), dtype=y_train.dtype)))
        else:
            client_data.append((x_train[ci], y_train[ci]))

    return client_data, x_test, y_test


# ───────────────────────────── main CLI ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare federated skeleton-HAR client data.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input NPZ (keys: x, y, [subjects]).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic COCO-17 data instead of loading.")
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes (synthetic mode).")
    parser.add_argument("--samples-per-class", type=int, default=200,
                        help="Samples per class (synthetic mode).")
    parser.add_argument("--num-clients", "-K", type=int, default=5,
                        help="Number of FL clients.")
    parser.add_argument("--split", choices=["subject", "dirichlet"],
                        default="dirichlet",
                        help="Partitioning strategy.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration (lower → more skew).")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="data/fed_skeleton",
                        help="Directory to write per-client NPZ files.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load or generate data ────────────────────────────────────────────
    if args.synthetic:
        print(f"Generating synthetic data: {args.num_classes} classes × "
              f"{args.samples_per_class} samples …")
        x, y, subjects = _random_skeleton_sequence(
            num_classes=args.num_classes,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    else:
        if args.input is None:
            raise ValueError("Provide --input NPZ or use --synthetic")
        data = np.load(args.input)
        x, y = data["x"], data["y"]
        subjects = data.get("subjects", None)

    print(f"Dataset shape: x={x.shape}  y={y.shape}  "
          f"classes={int(y.max())+1}")

    # ── Partition ────────────────────────────────────────────────────────
    if args.split == "subject":
        if args.synthetic:
            # subjects array generated above
            pass
        elif subjects is None:
            raise ValueError("subject split requires 'subjects' key in NPZ")
        client_data, x_test, y_test = split_by_subject(
            x, y, subjects, args.num_clients,
            test_ratio=args.test_ratio, seed=args.seed)
    else:
        client_data, x_test, y_test = split_dirichlet(
            x, y, args.num_clients, alpha=args.alpha,
            test_ratio=args.test_ratio, seed=args.seed)

    # ── Save ─────────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cid, (xc, yc) in enumerate(client_data):
        path = out / f"client_{cid}.npz"
        np.savez_compressed(path, x=xc, y=yc)
        label_dist = np.bincount(yc.astype(int), minlength=int(y.max())+1)
        print(f"  Client {cid}: {len(yc):5d} samples | labels {label_dist.tolist()}")

    test_path = out / "test.npz"
    np.savez_compressed(test_path, x=x_test, y=y_test)
    print(f"  Test set:  {len(y_test):5d} samples → {test_path}")
    print("Done.")


if __name__ == "__main__":
    main()
