"""Freeze train/development performers from official training data; never prepare test tensors."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from .corrected_fl import sha256
from .prepare_corrected_fl import assign_clients, class_contract
from .prepare_medical_data import preprocess_2d


def performer(sequence_id):
    match = re.search(r"P(\d{3})", sequence_id)
    if match is None:
        raise ValueError(f"Missing performer: {sequence_id}")
    return int(match.group(1))


def freeze_split(records, *, split_seed, development_subjects):
    subjects = sorted({performer(a["frame_dir"]) for a in records})
    if not 0 < development_subjects < len(subjects):
        raise ValueError("Need both training and development performers")
    dev = set(np.random.default_rng(split_seed).choice(
        subjects, size=development_subjects, replace=False).tolist())
    return {name: sorted([a for a in records if (performer(a["frame_dir"]) in dev) == is_dev],
                         key=lambda a: a["frame_dir"])
            for name, is_dev in (("train", False), ("development", True))}


def prepare(*, annotation, output_dir, train_per_class=20, development_per_class=10,
            split_seed=20261001, subset_seed=20261002, partition_seed=42,
            development_subjects=4, num_clients=5, alpha=.5):
    annotation, output_dir = Path(annotation), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    if min(train_per_class, development_per_class) < 0:
        raise ValueError("Per-class caps must be nonnegative; zero means all clips")
    with annotation.open("rb") as stream:
        source = pickle.load(stream)
    mapping, names = class_contract(15)
    train_ids, test_ids = (set(source["split"][key]) for key in ("xsub_train", "xsub_val"))
    if train_ids & test_ids or {performer(s) for s in train_ids} & {performer(s) for s in test_ids}:
        raise ValueError("Official train/test identities overlap")
    records = [a for a in source["annotations"] if a["frame_dir"] in train_ids and a["label"] in mapping]
    del source
    if len({a["frame_dir"] for a in records}) != len(records):
        raise ValueError("Duplicate source training sequence")
    frozen = freeze_split(records, split_seed=split_seed, development_subjects=development_subjects)
    arrays, split_manifest = {}, {}
    rng = np.random.default_rng(subset_seed)
    for split, cap in (("train", train_per_class), ("development", development_per_class)):
        all_records = frozen[split]
        selected = []
        for label in mapping:
            candidates = [a for a in all_records if a["label"] == label]
            if not candidates or (cap and len(candidates) < cap):
                raise ValueError(f"Insufficient {split} clips for source label {label}")
            indices = rng.choice(len(candidates), size=cap, replace=False) if cap else range(len(candidates))
            selected.extend(candidates[i] for i in indices)
        selected.sort(key=lambda a: a["frame_dir"])
        x = np.empty((len(selected), 2, 100, 17, 3), dtype=np.float32)
        for index, annotation_record in enumerate(selected):
            x[index] = preprocess_2d(annotation_record, clip_len=100, num_person=2)
        arrays[split] = {
            "x": x,
            "y": np.array([mapping[a["label"]] for a in selected], dtype=np.int64),
            "sequence_ids": np.array([a["frame_dir"] for a in selected]),
            "subject_ids": np.array([performer(a["frame_dir"]) for a in selected], dtype=np.int64),
        }
        split_manifest[split] = {
            "all_eligible_sequence_ids": [a["frame_dir"] for a in all_records],
            "all_eligible_subjects": sorted({performer(a["frame_dir"]) for a in all_records}),
            "eligible_n": len(all_records), "selected_n": len(selected), "per_class_cap": cap,
        }
    assignments, proportions = assign_clients(
        arrays["train"]["y"], arrays["development"]["y"], num_clients=num_clients,
        num_classes=15, alpha=alpha, seed=partition_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for (split, data), clients in zip(arrays.items(), assignments):
        path = output_dir / f"{split}.npz"
        np.savez_compressed(path, **data, client_ids=clients)
        files[split] = {
            "sha256": sha256(path), "samples": len(clients),
            "client_sizes": np.bincount(clients, minlength=num_clients).tolist(),
            "class_counts": np.bincount(data["y"], minlength=15).tolist(),
            "client_class_counts": [np.bincount(data["y"][clients == c], minlength=15).tolist()
                                    for c in range(num_clients)],
        }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 2,
        "purpose": "development_pilot" if train_per_class or development_per_class else "development_training",
        "publication_accuracy_claim_allowed": False,
        "graph": "coco", "sample_shape": [2, 100, 17, 3],
        "label_names": names, "ntu_zero_based_ids": list(mapping), "num_clients": num_clients,
        "split_seed": split_seed, "development_subject_count": development_subjects,
        "subset_seed": subset_seed, "partition_seed": partition_seed, "alpha": alpha,
        "label_client_proportions": proportions.tolist(), "files": files,
        "split_manifest_sha256": sha256(manifest_path),
        "source": {"path": str(annotation.resolve()), "sha256": sha256(annotation)},
        "excluded_official_test_subjects": sorted({performer(s) for s in test_ids}),
        "test_status": "Original test excluded from preparation and training; historically explored benchmark, not a new external test",
        "client_scope": "Simulated label-skew cohorts allocated using labels, not observed institutional ownership",
        "preprocessing": "Unchanged legacy 2D x,y,confidence; uniform100 frames, two people, COCO17 joints",
        "source_sha256": {"preparer": sha256(__file__),
                          "preprocessing": sha256(Path(__file__).parent / "prepare_medical_data.py"),
                          "partitioner": sha256(Path(__file__).parent / "prepare_corrected_fl.py")},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--development-per-class", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=20261001)
    parser.add_argument("--subset-seed", type=int, default=20261002)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--development-subjects", type=int, default=4)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=.5)
    protocol = prepare(**vars(parser.parse_args()))
    print(json.dumps({"purpose": protocol["purpose"], "files": protocol["files"]}, indent=2))


if __name__ == "__main__":
    main()
