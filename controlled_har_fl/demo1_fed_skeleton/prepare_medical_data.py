"""
prepare_medical_data.py — Prepare 15-class medical action data from NTU RGB+D 60.

Downloads (if needed) and processes both 2D (HRNet COCO-17) and 3D (NTU-25) skeleton
annotations into train/test NPZ files + federated client shards.

Output format: (N, M=2, T=100, V, C) — compatible with STGCN++ model.
  - 2D: V=17, C=3 (x_norm, y_norm, score)
  - 3D: V=25, C=3 (x, y, z)

Usage:
    python demo1_fed_skeleton/prepare_medical_data.py --num-clients 5 --alpha 0.5
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

# ─────────────────── 15-class medical action mapping ──────────────────────────
# NTU-60 0-indexed label → our 15-class label
# Matches the TRAINING_HANDOFF.md spec exactly
MEDICAL_CLASS_MAP = {
    42: 0,   # A043: falling
    41: 1,   # A042: staggering
    47: 2,   # A048: nausea or vomiting
    43: 3,   # A044: touch head (headache)
    44: 4,   # A045: touch chest (heart pain)
    45: 5,   # A046: touch back (backache)
    46: 6,   # A047: touch neck (neckache)
    40: 7,   # A041: sneeze/cough
    8:  8,   # A009: standing up
    7:  9,   # A008: sitting down
    58: 10,  # A059: walking towards
    59: 11,  # A060: walking apart
    0:  12,  # A001: drinking
    1:  13,  # A002: eating
    27: 14,  # A028: phone call  (was 16 in old map, but A028 = make a phone call = 0-indexed 27)
}

MEDICAL_LABELS = [
    'falling',          # 0  — EMERGENCY
    'staggering',       # 1  — PAIN
    'nausea/vomiting',  # 2  — EMERGENCY
    'touch head',       # 3  — PAIN
    'touch chest',      # 4  — PAIN
    'touch back',       # 5  — PAIN
    'touch neck',       # 6  — PAIN
    'sneeze/cough',     # 7  — SYMPTOM
    'standing up',      # 8  — NORMAL
    'sitting down',     # 9  — NORMAL
    'walking towards',  # 10 — NORMAL
    'walking apart',    # 11 — NORMAL
    'drinking',         # 12 — NORMAL
    'eating',           # 13 — NORMAL
    'phone call',       # 14 — NORMAL
]

NUM_CLASSES = len(MEDICAL_LABELS)

# URLs for NTU-60 skeleton data
URL_2D = 'https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl'
URL_3D = 'https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl'


# ─────────────────── preprocessing helpers ───────────────────────────────────

def prenormalize_2d(keypoint, keypoint_score, img_shape=(1080, 1920), threshold=0.01):
    """Normalize 2D keypoints to [-1, 1] and append confidence score."""
    kp = keypoint.astype(np.float32).copy()
    score = keypoint_score.astype(np.float32)
    kp = np.concatenate([kp, score[..., None]], axis=-1)  # (M, T, V, 3)
    mask_out = kp[..., 2] <= threshold
    h, w = img_shape
    kp[..., 0] = (kp[..., 0] - w / 2) / (w / 2)
    kp[..., 1] = (kp[..., 1] - h / 2) / (h / 2)
    kp[..., 0][mask_out] = 0
    kp[..., 1][mask_out] = 0
    return kp


def prenormalize_3d(keypoint):
    """Center 3D skeleton at spine joint (joint 1) and scale to unit."""
    kp = keypoint.astype(np.float32).copy()  # (M, T, V, C)
    # Center at spine mid (joint 1 in NTU-25 = spine)
    spine = kp[:, :, 1:2, :]  # (M, T, 1, C)
    kp = kp - spine
    # Scale so max distance from center ≈ 1
    dist = np.sqrt((kp ** 2).sum(axis=-1, keepdims=True))  # (M, T, V, 1)
    max_dist = dist.max()
    if max_dist > 1e-6:
        kp = kp / max_dist
    return kp


def uniform_sample(keypoint, clip_len=100):
    """Temporally sample or pad to fixed clip_len."""
    M, T, V, C = keypoint.shape
    if T == clip_len:
        return keypoint
    if T < clip_len:
        inds = np.arange(clip_len) % T
    else:
        inds = np.linspace(0, T - 1, clip_len, dtype=int)
    return keypoint[:, inds]


def format_gcn_input(keypoint, num_person=2):
    """Pad or trim to num_person slots."""
    M, T, V, C = keypoint.shape
    if M < num_person:
        pad = np.zeros((num_person - M, T, V, C), dtype=keypoint.dtype)
        keypoint = np.concatenate([keypoint, pad], axis=0)
    return keypoint[:num_person]


def preprocess_2d(ann, clip_len=100, num_person=2):
    """Process a single 2D annotation → (M, T, 17, 3)."""
    img_shape = ann.get('img_shape', (1080, 1920))
    kp = prenormalize_2d(ann['keypoint'], ann['keypoint_score'], img_shape)
    kp = uniform_sample(kp, clip_len)
    return format_gcn_input(kp, num_person)


def preprocess_3d(ann, clip_len=100, num_person=2):
    """Process a single 3D annotation → (M, T, 25, 3)."""
    kp = prenormalize_3d(ann['keypoint'])
    kp = uniform_sample(kp, clip_len)
    return format_gcn_input(kp, num_person)


# ─────────────────── partitioning ────────────────────────────────────────────

def dirichlet_split(x, y, num_clients, alpha, seed=42):
    """Dirichlet non-IID split of training data across clients."""
    rng = np.random.RandomState(seed)
    num_classes = int(y.max()) + 1

    label_indices = defaultdict(list)
    for i, label in enumerate(y):
        label_indices[int(label)].append(i)

    client_indices = {i: [] for i in range(num_clients)}
    for cls in range(num_classes):
        idxs = np.array(label_indices[cls])
        rng.shuffle(idxs)
        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = np.cumsum(proportions)
        proportions = proportions / proportions[-1]
        splits = np.split(idxs, (proportions[:-1] * len(idxs)).astype(int))
        for cid, chunk in enumerate(splits):
            client_indices[cid].extend(chunk.tolist())

    client_data = []
    for cid in range(num_clients):
        ci = np.array(client_indices[cid], dtype=int)
        client_data.append((x[ci], y[ci]))
    return client_data


# ─────────────────── main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Prepare 15-class medical skeleton data')
    parser.add_argument('--data-dir', default='demo1_fed_skeleton/data/nturgbd')
    parser.add_argument('--output-dir', default='demo1_fed_skeleton/data/fed_medical')
    parser.add_argument('--num-clients', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clip-len', type=int, default=100)
    parser.add_argument('--skip-2d', action='store_true')
    parser.add_argument('--skip-3d', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected_classes = set(MEDICAL_CLASS_MAP.keys())

    # ── Process 2D data ──────────────────────────────────────────────────
    if not args.skip_2d:
        pkl_2d = data_dir / 'ntu60_hrnet.pkl'
        if not pkl_2d.exists():
            print(f'Downloading 2D data → {pkl_2d} ...')
            urlretrieve(URL_2D, str(pkl_2d))

        print(f'Loading 2D data from {pkl_2d} ...')
        with open(pkl_2d, 'rb') as f:
            data_2d = pickle.load(f)

        split_info = data_2d['split']
        train_dirs = set(split_info['xsub_train'])
        test_dirs = set(split_info['xsub_val'])

        medical_anns = [a for a in data_2d['annotations'] if a['label'] in selected_classes]
        print(f'2D: {len(medical_anns)} medical samples from {len(data_2d["annotations"])} total')

        train_x, train_y, test_x, test_y = [], [], [], []
        for ann in medical_anns:
            x = preprocess_2d(ann, clip_len=args.clip_len)
            new_label = MEDICAL_CLASS_MAP[ann['label']]
            if ann['frame_dir'] in train_dirs:
                train_x.append(x)
                train_y.append(new_label)
            elif ann['frame_dir'] in test_dirs:
                test_x.append(x)
                test_y.append(new_label)

        train_x = np.stack(train_x).astype(np.float32)
        train_y = np.array(train_y, dtype=np.int64)
        test_x = np.stack(test_x).astype(np.float32)
        test_y = np.array(test_y, dtype=np.int64)

        print(f'2D train: {train_x.shape}, test: {test_x.shape}')
        _save_splits(train_x, train_y, test_x, test_y,
                     args.output_dir + '_2d', args.num_clients, args.alpha, args.seed)

        # Also save combined for centralized training
        out_cent = Path(args.output_dir + '_2d') / 'centralized'
        out_cent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_cent / 'train.npz', x=train_x, y=train_y)
        np.savez_compressed(out_cent / 'test.npz', x=test_x, y=test_y)
        print(f'2D centralized data saved → {out_cent}')

        del data_2d  # free memory

    # ── Process 3D data ──────────────────────────────────────────────────
    if not args.skip_3d:
        pkl_3d = data_dir / 'ntu60_3danno.pkl'
        if not pkl_3d.exists():
            print(f'Downloading 3D data → {pkl_3d} ...')
            urlretrieve(URL_3D, str(pkl_3d))

        print(f'Loading 3D data from {pkl_3d} ...')
        with open(pkl_3d, 'rb') as f:
            data_3d = pickle.load(f)

        split_info = data_3d['split']
        train_dirs = set(split_info['xsub_train'])
        test_dirs = set(split_info['xsub_val'])

        medical_anns = [a for a in data_3d['annotations'] if a['label'] in selected_classes]
        print(f'3D: {len(medical_anns)} medical samples from {len(data_3d["annotations"])} total')

        train_x, train_y, test_x, test_y = [], [], [], []
        for ann in medical_anns:
            x = preprocess_3d(ann, clip_len=args.clip_len)
            new_label = MEDICAL_CLASS_MAP[ann['label']]
            if ann['frame_dir'] in train_dirs:
                train_x.append(x)
                train_y.append(new_label)
            elif ann['frame_dir'] in test_dirs:
                test_x.append(x)
                test_y.append(new_label)

        train_x = np.stack(train_x).astype(np.float32)
        train_y = np.array(train_y, dtype=np.int64)
        test_x = np.stack(test_x).astype(np.float32)
        test_y = np.array(test_y, dtype=np.int64)

        print(f'3D train: {train_x.shape}, test: {test_x.shape}')
        _save_splits(train_x, train_y, test_x, test_y,
                     args.output_dir + '_3d', args.num_clients, args.alpha, args.seed)

        # Centralized
        out_cent = Path(args.output_dir + '_3d') / 'centralized'
        out_cent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_cent / 'train.npz', x=train_x, y=train_y)
        np.savez_compressed(out_cent / 'test.npz', x=test_x, y=test_y)
        print(f'3D centralized data saved → {out_cent}')

    print('\nDone! Label map:')
    for i, lbl in enumerate(MEDICAL_LABELS):
        print(f'  [{i:2d}] {lbl}')


def _save_splits(train_x, train_y, test_x, test_y,
                 output_dir, num_clients, alpha, seed):
    """Save federated client shards + test set."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save test set
    np.savez_compressed(out / 'test.npz', x=test_x, y=test_y)

    # Federated split
    client_data = dirichlet_split(train_x, train_y, num_clients, alpha, seed)
    for cid, (cx, cy) in enumerate(client_data):
        np.savez_compressed(out / f'client_{cid}.npz', x=cx, y=cy)
        n_cls = len(set(cy.tolist())) if len(cy) > 0 else 0
        print(f'  Client {cid}: {len(cy):5d} samples, {n_cls}/{NUM_CLASSES} classes')

    print(f'  Test:    {len(test_y):5d} samples')
    print(f'  Saved → {out}')


if __name__ == '__main__':
    main()
