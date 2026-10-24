"""Prepare versioned NTU cross-subject tensors and matched label-skew clients.

Supply an existing trusted PYSKL annotation pickle; no downloads are implicit.
``--smoke`` makes independent synthetic full-shape fixtures, never NTU results.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from .corrected_fl import sha256
from .prepare_medical_data import MEDICAL_CLASS_MAP, MEDICAL_LABELS, preprocess_2d, preprocess_3d


def class_contract(classes):
    if classes == 15:
        return dict(MEDICAL_CLASS_MAP), list(MEDICAL_LABELS)
    if classes == 10:
        source_ids = [42, 41, 43, 44, 45, 46, 47, 8, 7, 58]
        names = [MEDICAL_LABELS[MEDICAL_CLASS_MAP[i]] for i in source_ids]
        return {source: new for new, source in enumerate(source_ids)}, names
    raise ValueError("Only the explicit historical 10-class or medical 15-class contract is supported")


def assign_clients(train_y, test_y, *, num_clients, num_classes, alpha, seed):
    """Freeze label proportions once; independently allocate train/test examples."""
    if num_clients < 2 or alpha <= 0:
        raise ValueError("Need at least two clients and positive Dirichlet alpha")
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet(np.full(num_clients, alpha), size=num_classes)
    assignments = []
    for labels in (train_y, test_y):
        clients = np.full(len(labels), -1, dtype=np.int64)
        for label in range(num_classes):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            counts = np.floor(proportions[label] * len(indices)).astype(int)
            residual = len(indices) - counts.sum()
            order = np.argsort(-(proportions[label] * len(indices) - counts), kind="stable")
            counts[order[:residual]] += 1
            for cid, chunk in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                clients[chunk] = cid
        if np.any(clients < 0) or set(clients) != set(range(num_clients)):
            raise ValueError("Empty client or undeclared label: choose more data or a prespecified new partition")
        assignments.append(clients)
    return assignments, proportions


def prepare(*, output_dir, annotation=None, graph="coco", classes=15, num_clients=5,
            alpha=.5, seed=42, smoke=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty new data directory")
    mapping, names = class_contract(classes)
    sample_shape = (2, 100, 17 if graph == "coco" else 25, 3)
    arrays = {}
    if smoke:
        if annotation is not None:
            raise ValueError("Synthetic smoke cannot also use real annotations")
        rng = np.random.default_rng(seed)
        for split, repeats, offset in [("train", 2, 0), ("test", 1, 1000)]:
            labels = np.tile(np.arange(classes, dtype=np.int64), repeats)
            x = rng.normal(0, .2, size=(len(labels), *sample_shape)).astype(np.float32)
            x[:, 1] = 0
            if graph == "coco":
                x[:, 0, :, :, 2] = 1
            arrays[split] = dict(x=x, y=labels,
                                 sequence_ids=np.array([f"synthetic-{split}-{i}" for i in range(len(labels))]),
                                 subject_ids=np.arange(len(labels), dtype=np.int64) + offset)
        source = {"kind": "generated independent random fixtures; no research accuracy claim"}
    else:
        if annotation is None:
            raise ValueError("Real preparation requires the original trusted annotation pickle")
        annotation = Path(annotation)
        with annotation.open("rb") as stream:
            data = pickle.load(stream)
        split_ids = {"train": set(data["split"]["xsub_train"]),
                     "test": set(data["split"]["xsub_val"])}
        if split_ids["train"] & split_ids["test"]:
            raise ValueError("Source cross-subject split has overlapping sequence IDs")
        records = {split: {key: [] for key in ("x", "y", "sequence_ids", "subject_ids")}
                   for split in split_ids}
        preprocess = preprocess_2d if graph == "coco" else preprocess_3d
        for ann in data["annotations"]:
            if ann["label"] not in mapping:
                continue
            seq_id = ann["frame_dir"]
            split = next((s for s in split_ids if seq_id in split_ids[s]), None)
            if split is None:
                continue
            match = re.search(r"P(\d{3})", seq_id)
            if match is None:
                raise ValueError(f"Cannot recover original performer identity: {seq_id}")
            record = records[split]
            record["x"].append(preprocess(ann, clip_len=100, num_person=2))
            record["y"].append(mapping[ann["label"]])
            record["sequence_ids"].append(seq_id)
            record["subject_ids"].append(int(match.group(1)))
        for split, record in records.items():
            arrays[split] = {"x": np.stack(record["x"]).astype(np.float32),
                             "y": np.asarray(record["y"], dtype=np.int64),
                             "sequence_ids": np.asarray(record["sequence_ids"]),
                             "subject_ids": np.asarray(record["subject_ids"], dtype=np.int64)}
        source = {"path": str(annotation.resolve()), "sha256": sha256(annotation),
                  "split": "PYSKL xsub_train versus xsub_val; original performer IDs retained"}
    if set(arrays["train"]["subject_ids"]) & set(arrays["test"]["subject_ids"]):
        raise ValueError("Source train/test performer identities overlap")
    assignments, proportions = assign_clients(arrays["train"]["y"], arrays["test"]["y"],
                                               num_clients=num_clients, num_classes=classes,
                                               alpha=alpha, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for (split, arrays_split), clients in zip(arrays.items(), assignments):
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **arrays_split, client_ids=clients)
        files[split] = {"sha256": sha256(path), "samples": len(clients),
                        "client_sizes": np.bincount(clients, minlength=num_clients).tolist(),
                        "class_counts": np.bincount(arrays_split["y"], minlength=classes).tolist()}
    protocol = {"schema_version": 1, "purpose": "synthetic_smoke" if smoke else "research",
                "graph": graph, "sample_shape": list(sample_shape), "label_names": names,
                "ntu_zero_based_ids": list(mapping), "num_clients": num_clients,
                "alpha": alpha, "partition_seed": seed, "label_client_proportions": proportions.tolist(),
                "partition_rule": "shared fixed Dirichlet label proportions; separately shuffled train/test allocation",
                "client_scope": "simulated label-skew clients; not real hospitals or inferred patient identities",
                "test_status": "historically explored benchmark; not newly collected untouched validation",
                "preprocessing": "unchanged prepare_medical_data.py; (x,y,score) for coco, (x,y,z) for ntu",
                "source": source, "files": files,
                "source_sha256": {"preparer": sha256(__file__),
                                  "preprocessing": sha256(Path(__file__).parent / "prepare_medical_data.py")}}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--graph", choices=["coco", "ntu"], default="coco")
    parser.add_argument("--classes", type=int, choices=[10, 15], default=15)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = prepare(**vars(args))
    print(json.dumps({"purpose": protocol["purpose"], "files": protocol["files"]}, indent=2))


if __name__ == "__main__":
    main()
