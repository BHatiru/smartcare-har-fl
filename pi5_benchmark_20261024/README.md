# Raspberry Pi 5 compute benchmark, 24 October 2026

Recorded on a Raspberry Pi 5 Model B (8 GB), Raspberry Pi OS / Debian 13, without a detected fan; the heatsink condition was unconfirmed. The deployment used MoveNet SinglePose Thunder int8 and the older centralized 15-class ONNX checkpoint, not a newly trained federated model.

The measured configuration processes pre-decoded 455x256 frames with four threads, recognition every six frames and two person slots. Cold effective compute throughput was 38.9 frames/s; the mean of run-level values in the final minute of the ten-minute sequence was 26.9 frames/s. The device reached 88.9 C and throttled. These are compute timings on a repeated 72-frame clip, not camera-to-alarm latency, clinical performance or measured energy consumption.

## Evidence

- [Summary](experiments/pi5_edge_benchmark_20261024/summary.json).
- [Raw logs](experiments/pi5_edge_benchmark_20261024/pi_measure_20261024_1654/), including cold, sustained, alternative-front-end and thermal measurements.
- [Analyzer](experiments/pi5_edge_benchmark_20261024/analyze.py), which regenerates the summary from those logs.
- The three Python source files recorded in `env.txt` are supplied byte-for-byte under their original relative paths. Their hashes were checked against the device record.
- The measurement shell script and 15-class label map are also supplied. The shell script was not itself hashed in the device's environment record; do not extend the three-file hash assertion to it.

From this directory:

```bash
python experiments/pi5_edge_benchmark_20261024/analyze.py
```

This regenerates the recorded summary on a normal CPU. Repeating the actual hardware measurement additionally requires a Pi, the video, pose/detector models and matching runtime dependencies. Those inputs are not bundled here. The centralized recognizer is already available in the repository's top-level `checkpoints/`; `env.txt` identifies the model hashes and dependency versions used on the device. Preserve the supplied nested layout or explicitly pass model/video/label paths.

Some original harness labels use the phrase "end-to-end" for its timed compute loop. Camera capture, decoding before the loop, network delivery and alert handling are outside that loop. Use the scope above when interpreting the logs.
