"""Verify the received lab snapshot using only the Python standard library."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Invalid or empty bundle manifest")
    failures = []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            failures.append({"path": name, "reason": "outside workspace"})
        elif not path.is_file():
            failures.append({"path": name, "reason": "missing"})
        elif path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            failures.append({"path": name, "reason": "content differs"})
    result = {"passed": not failures, "files_checked": len(manifest["files"]),
              "snapshot": manifest.get("snapshot"), "failures": failures,
              "scope": "Initial transfer integrity, not scientific or GPU validation. Extra run files are allowed."}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
