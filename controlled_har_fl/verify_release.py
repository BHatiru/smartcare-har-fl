"""Verify the public artifact subset and reproduce manuscript metrics without raw inputs."""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import statistics

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Changed released artifact: {name}")
    frozen = json.loads((ROOT / "lab_results/frozen_selection.json").read_text())
    for name, digest in frozen["run_contract"]["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Changed original source: {name}")
    # This original auditor checks checkpoint hashes and every prediction identity/metric.
    audit_path = ROOT / "lab_results/reporting/final_metric_audit.json"
    original_audit = audit_path.read_bytes()
    try:
        runpy.run_path(str(ROOT / "lab_results/reporting/audit_final_metrics.py"), run_name="__main__")
    finally:
        # The archived auditor writes a report; retain the shipped bytes across OS newline rules.
        if audit_path.read_bytes() != original_audit:
            audit_path.write_bytes(original_audit)
    results = json.loads((ROOT / "lab_results/final_test/results.json").read_text())["results"]
    with (ROOT / "paper/method_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        selected = [r for r in results if r["method"] == row["method"]]
        if {r["seed"] for r in selected} != {42, 43, 44}:
            raise ValueError("Missing or duplicate model seed")
        for prefix, values in [
            ("accuracy", [100 * r["metrics"]["pooled"]["accuracy"] for r in selected]),
            ("macro_f1", [100 * r["metrics"]["pooled"]["macro_f1_all_declared_classes"] for r in selected]),
            ("worst_client_accuracy", [100 * r["metrics"]["worst_client_accuracy"] for r in selected]),
        ]:
            for suffix, actual in [("mean", statistics.mean(values)), ("sample_sd", statistics.stdev(values))]:
                if abs(actual - float(row[f"{prefix}_{suffix}_percent"])) > 1e-10:
                    raise ValueError("Paper CSV differs from saved predictions")
    print(json.dumps(dict(passed=True, released_files=len(manifest["sha256"]),
                          original_source_files=len(frozen["run_contract"]["source_sha256"]),
                          method_seed_results=len(results), paper_method_rows=len(rows),
                          scope="Saved-artifact verification; no training or model-inference rerun")))


if __name__ == "__main__":
    main()
