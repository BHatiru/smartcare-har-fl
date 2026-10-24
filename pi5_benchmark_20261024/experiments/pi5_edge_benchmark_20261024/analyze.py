"""Summarize the Raspberry Pi 5 edge benchmark (run_pi_measurements.sh output) for the paper.

Run: python experiments/pi5_edge_benchmark_20261024/analyze.py
Writes summary.json next to this file. Compute-only timings on pre-decoded frames (bench.py);
camera capture and network delivery are not included.
"""
import json, re, statistics
from pathlib import Path

RUN = Path(__file__).parent / "pi_measure_20261024_1654"
STAGE = re.compile(r"^\s+(det|pose|recog|frame)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
FPS = re.compile(r"EFFECTIVE PIPELINE FPS.*?:\s*([\d.]+)")


def parse_runs(text):
    runs, cur = [], None
    for line in text.splitlines():
        if line.startswith("### run"):
            cur = {"t": int(re.search(r"t=(\d+)", line).group(1))}
            runs.append(cur)
            continue
        if "CONFIG:" in line and (cur is None or "fps" in cur):  # files without "### run" markers
            cur = {"t": cur["t"] if cur else 0, "config": line.split("CONFIG:")[1].split("|")[0].strip()}
            runs.append(cur)
            continue
        if "CONFIG:" in line:
            cur["config"] = line.split("CONFIG:")[1].split("|")[0].strip()
        if m := STAGE.match(line):
            cur[m.group(1)] = {"mean": float(m.group(2)), "p50": float(m.group(3)), "p95": float(m.group(4))}
        elif m := FPS.search(line):
            cur["fps"] = float(m.group(1))
    return [r for r in runs if "fps" in r]


def window(runs, lo, hi):
    sel = [r for r in runs if lo <= r["t"] < hi]
    return {"runs": len(sel), "fps_mean": round(statistics.mean(r["fps"] for r in sel), 1),
            "pose_ms_mean": round(statistics.mean(r["pose"]["mean"] for r in sel), 1),
            "recog_ms_mean": round(statistics.mean(r["recog"]["mean"] for r in sel), 1),
            "frame_p95_ms_mean": round(statistics.mean(r["frame"]["p95"] for r in sel), 1)}


thermal = []
for line in (RUN / "thermal.log").read_text().splitlines():
    m = re.match(r"(\d+) temp=([\d.]+).*?=(\d+) throttled=(0x[0-9a-f]+)", line)
    if m:
        thermal.append((int(m.group(1)), float(m.group(2)), int(m.group(3)) / 1e9, int(m.group(4), 16)))
t0 = thermal[0][0]
first_throttle = next((t - t0 for t, _, _, flag in thermal if flag & 0x4), None)
first_softcap = next((t - t0 for t, _, _, flag in thermal if flag & 0x8), None)

cold = parse_runs((RUN / "cold_run.txt").read_text())[0]
peak = float(re.search(r"peak_rss_MiB ([\d.]+)", (RUN / "cold_run.txt").read_text()).group(1))
sustained = parse_runs((RUN / "sustained.txt").read_text())
alts = parse_runs((RUN / "alternatives.txt").read_text())

summary = {
    "device": "Raspberry Pi 5 Model B (8 GB), Raspberry Pi OS (Debian 13), Python 3.13.5, onnxruntime 1.31.0, ai-edge-litert 2.3.0",
    "config": "MoveNet SinglePose Thunder int8 + 15-class graph model (ONNX), 455x256 input, 4 threads, recognition every 6 frames, 2 person slots",
    "cold": {"fps": cold["fps"], "pose": cold["pose"], "recog": cold["recog"], "frame": cold["frame"], "peak_rss_MiB": peak},
    "sustained_runs": len(sustained),
    "sustained_first_60s": window(sustained, 0, 60),
    "sustained_last_60s": window(sustained, 540, 601),
    "sustained_all": window(sustained, 0, 601),
    "thermal": {"start_C": thermal[0][1], "max_C": max(x[1] for x in thermal),
                "min_clock_GHz": round(min(x[2] for x in thermal), 2), "max_clock_GHz": round(max(x[2] for x in thermal), 2),
                "first_throttle_s": first_throttle, "first_soft_temp_limit_s": first_softcap,
                "undervoltage_seen": any(x[3] & 0x10001 for x in thermal),
                "throttled_fraction": round(sum(1 for x in thermal if x[3] & 0x4) / len(thermal), 3)},
    "alternatives_after_sustained_thermal_state": [
        {"config": a.get("config"), "fps": a["fps"], "pose_ms": a["pose"]["mean"], "recog_ms": a["recog"]["mean"],
         "det_ms": a.get("det", {}).get("mean")} for a in alts],
}
(Path(__file__).parent / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))

assert len(sustained) > 100 and summary["sustained_first_60s"]["runs"] > 0 and summary["sustained_last_60s"]["runs"] > 0
assert len(alts) == 2, "expected MoveNet Lightning and RTMPose alternative runs"
