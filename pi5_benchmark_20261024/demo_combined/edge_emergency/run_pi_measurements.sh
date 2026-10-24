#!/usr/bin/env bash
# On-device measurement for the paper: per-stage latency, effective FPS, sustained thermals, memory.
# Run on the Pi from the repo root (venv active):  bash demo_combined/edge_emergency/run_pi_measurements.sh "active cooler"
# The argument records the cooling setup (e.g. "active cooler", "no heatsink"). ~15 min total.
set -u
COOLING="${1:?describe cooling, e.g. \"active cooler\" or \"no heatsink\"}"
OUT="pi_measure_$(date +%Y%m%d_%H%M)"; mkdir -p "$OUT"
B="demo_combined/edge_emergency/bench.py"; M="demo/onnx_models"
# Deployed model input has M=2 person slots (paper Sec. III-C2).
PI_CFG=(--pose-backend movenet --movenet-model "$M/movenet_thunder_int8.tflite" --short-side 256 --threads 4 --recog-every 6 --num-person 2)

{ cat /proc/device-tree/model; echo; uname -a; head -1 /etc/os-release; echo "cooling: $COOLING"; python3 --version
  python3 -c "import numpy, cv2, onnxruntime; print('numpy', numpy.__version__, 'opencv', cv2.__version__, 'onnxruntime', onnxruntime.__version__)"
  python3 -c "from importlib.metadata import version; print('ai-edge-litert', version('ai-edge-litert'))"
  vcgencmd pmic_read_adc EXT5V_V; free -m; sha256sum "$B" demo_combined/edge_emergency/pose_backends.py demo/demo_onnx.py "$M"/*; } > "$OUT/env.txt" 2>&1

(while true; do echo "$(date +%s) $(vcgencmd measure_temp) $(vcgencmd measure_clock arm) $(vcgencmd get_throttled)"; sleep 2; done) > "$OUT/thermal.log" &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT

echo "1/3 cold run with peak memory"
python3 -c "import resource, subprocess, sys; subprocess.run(sys.argv[1:]); print('peak_rss_MiB', round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1))" \
  python3 "$B" "${PI_CFG[@]}" > "$OUT/cold_run.txt" 2>&1

echo "2/3 sustained 10 min (Pi launcher config)"
end=$((SECONDS + 600)); i=0
while [ $SECONDS -lt $end ]; do i=$((i + 1)); echo "### run $i t=$SECONDS s"; python3 "$B" "${PI_CFG[@]}"; done > "$OUT/sustained.txt" 2>&1

echo "3/3 alternative front ends (one run each)"
{ [ -f "$M/movenet_lightning_int8.tflite" ] && python3 "$B" --pose-backend movenet --movenet-model "$M/movenet_lightning_int8.tflite" --short-side 256 --threads 4 --recog-every 6 --num-person 2
  [ -f "$M/rtmpose_m.onnx" ] && python3 "$B" --pose-backend rtmpose --short-side 256 --threads 4 --det-every 3 --recog-every 6 --num-person 2; } > "$OUT/alternatives.txt" 2>&1

tar czf "$OUT.tar.gz" "$OUT" && echo "done: $OUT.tar.gz  (copy it back to the laptop)"
