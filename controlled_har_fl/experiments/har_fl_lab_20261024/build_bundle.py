"""Build a research-only transfer archive from an explicit file allowlist."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREFIX = "har_fl_lab_5080"


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = [
        "demo1_fed_skeleton/__init__.py", "demo1_fed_skeleton/models/__init__.py",
        "demo1_fed_skeleton/models/stgcn.py", "demo1_fed_skeleton/corrected_fl.py",
        "demo1_fed_skeleton/development_fl.py", "demo1_fed_skeleton/prepare_medical_data.py",
        "demo1_fed_skeleton/prepare_corrected_fl.py", "demo1_fed_skeleton/prepare_development_fl.py",
        "demo1_fed_skeleton/test_corrected_fl.py", "demo1_fed_skeleton/test_development_fl.py",
        "demo1_fed_skeleton/test_resumable_fl.py",
        "demo1_fed_skeleton/data/nturgbd/ntu60_hrnet.pkl",
        "demo1_fed_skeleton/data/fed_medical_2d/centralized/test.npz",
        "handoffs/FL_LAB_5080_20261024.md",
    ]
    old = "experiments/har_fl_repair_20261001"
    names += [f"{old}/{name}" for name in [
        "bootstrap_windows_pc.ps1", "inspect_windows_pc.ps1", "device_probe.py",
        "run_pilot.ps1", "setup_pc.ps1", "verify_pilot.py", "run_background_fl.py",
        "bind_legacy_test.py", "recovered_source.json", "legacy_test_binding.json",
        "legacy_test_identities.csv", "PILOT_RESULTS.txt", "pilot_summary.json",
    ]]
    for dataset in ["pilot_data", "full_development_data"]:
        names += [f"{old}/{dataset}/{name}" for name in ["train.npz", "development.npz", "protocol.json", "split_manifest.json"]]
    names += [f"{old}/overnight_cpu_20261001/{method}/round_003.json"
              for method in ["centralized", "local", "fedavg", "fedbn", "fedprox"]]
    names += [f"{old}/overnight_cpu_20261001/{name}" for name in ["status.json", "job.json", "code_manifest.json"]]
    names += [f"experiments/har_fl_lab_20261024/{name}" for name in [
        "verify_bundle.py", "gpu_preflight.py", "build_bundle.py",
        "START_HERE.md", "CODEX_START_PROMPT.txt", "LAB_AGENT_RULES.md",
    ]]
    mappings = {name: ROOT / name for name in names}
    mappings.update({"START_HERE.md": HERE / "START_HERE.md",
                     "CODEX_START_PROMPT.txt": HERE / "CODEX_START_PROMPT.txt",
                     "AGENTS.md": HERE / "LAB_AGENT_RULES.md"})
    virtual = {
        ".gitignore": b".venv*/\n__pycache__/\n*.pyc\nlab_results/\n*.zip\n*.pt\n*.npz\n*.pkl\nexperiments/har_fl_repair_20261001/pc_reports/\n",
        "CHECK_PC.cmd": (
            '@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0experiments\\har_fl_repair_20261001\\inspect_windows_pc.ps1"\r\n'
            'set "HAR_EXIT_CODE=%errorlevel%"\r\npause\r\nexit /b %HAR_EXIT_CODE%\r\n').encode("ascii"),
    }
    forbidden = re.compile(rb"(?:olp_|sk-proj-|ghp_)[A-Za-z0-9_-]{16,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
    files = {}
    for name, path in mappings.items():
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Invalid allowlisted file: {name}")
        if path.suffix.lower() in {".md", ".txt", ".py", ".ps1", ".json", ".csv"} and forbidden.search(path.read_bytes()):
            raise ValueError(f"Credential-like material in allowlisted text: {name}")
        files[name] = {"sha256": digest_file(path), "bytes": path.stat().st_size}
    for name, data in virtual.items():
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {
        "schema_version": 1, "snapshot": "HAR/FL RTX5080 lab handoff 2026-10-24",
        "created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "scope": "Selected current research source, full HAR annotations, prepared train/development and legacy test identities. No repository history, credentials, manuscript or vitals data.",
        "gpu_validated_on_source_machine": False,
        "final_test_evaluator_complete": False,
        "resumable_gpu_queue_complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, path in mappings.items():
            compression = zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED
            archive.write(path, f"{PREFIX}/{name}", compress_type=compression)
        for name, data in virtual.items():
            archive.writestr(f"{PREFIX}/{name}", data)
        archive.writestr(f"{PREFIX}/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        expected_names = {f"{PREFIX}/{name}" for name in files} | {f"{PREFIX}/BUNDLE_MANIFEST.json"}
        if set(archive.namelist()) != expected_names or len(archive.namelist()) != len(expected_names):
            raise AssertionError("Unexpected or duplicate ZIP entries")
        for name, entry in files.items():
            with archive.open(f"{PREFIX}/{name}") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise AssertionError(name)
    summary = {
        "archive": str(args.output.resolve()), "bytes": args.output.stat().st_size,
        "sha256": digest_file(args.output), "entry_count": len(files) + 1,
        "all_allowlisted_file_hashes_verified_in_zip": True,
        "credential_pattern_scan_passed": True,
        "raw_source_included": True, "gpu_training_tested": False,
    }
    report = args.output.with_suffix(".verification.json")
    with report.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
