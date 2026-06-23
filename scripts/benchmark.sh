#!/usr/bin/env bash
# scripts/benchmark.sh
# 60-second pipeline performance benchmark
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DURATION="${1:-60}"
API_BASE="http://localhost:8080/api/v1"

echo "════════════════════════════════════════════════"
echo "  Edge AI Navigation - Performance Benchmark"
echo "  Duration: ${DURATION}s"
echo "════════════════════════════════════════════════"
echo ""

# Check service is running
if ! curl -sf "${API_BASE}/status" > /dev/null; then
    echo "ERROR: Service not running at ${API_BASE}"
    exit 1
fi

echo "Collecting metrics for ${DURATION} seconds…"
echo ""

# Collect telemetry samples
python3 - <<PYEOF
import urllib.request, json, time, statistics

base = "${API_BASE}"
duration = ${DURATION}

samples = {
    "fps_cam": [], "fps_inf": [], "fps_lid": [], "fps_fus": [],
    "cpu": [], "mem_mb": [], "temp_cpu": [], "inf_lat": [],
}

start = time.monotonic()
n = 0
while time.monotonic() - start < duration:
    try:
        with urllib.request.urlopen(f"{base}/telemetry", timeout=2) as r:
            d = json.load(r)
            samples["fps_cam"].append(d.get("fps", {}).get("camera", 0))
            samples["fps_inf"].append(d.get("fps", {}).get("inference", 0))
            samples["fps_lid"].append(d.get("fps", {}).get("lidar", 0))
            samples["fps_fus"].append(d.get("fps", {}).get("fusion", 0))
            samples["cpu"].append(d.get("cpu", {}).get("percent", 0))
            samples["mem_mb"].append(d.get("memory", {}).get("used_mb", 0))
            samples["temp_cpu"].append(d.get("temperature", {}).get("cpu_c", 0))
            hailo = d.get("hailo", {})
            if hailo.get("last_latency_ms"):
                samples["inf_lat"].append(hailo["last_latency_ms"])
        n += 1
    except Exception as e:
        pass
    time.sleep(1.0)

def stat(arr, unit=""):
    if not arr: return "N/A"
    return f"{statistics.mean(arr):.1f}{unit} (min {min(arr):.1f}, max {max(arr):.1f})"

print(f"  Samples collected:         {n}")
print(f"  Camera FPS:                {stat(samples['fps_cam'], ' fps')}")
print(f"  Inference FPS:             {stat(samples['fps_inf'], ' fps')}")
print(f"  LiDAR scan rate:           {stat(samples['fps_lid'], ' Hz')}")
print(f"  Fusion rate:               {stat(samples['fps_fus'], ' Hz')}")
print(f"  CPU utilisation:           {stat(samples['cpu'], '%')}")
print(f"  Memory used:               {stat(samples['mem_mb'], ' MB')}")
print(f"  CPU temperature:           {stat(samples['temp_cpu'], '°C')}")
if samples['inf_lat']:
    print(f"  Inference latency:         {stat(samples['inf_lat'], ' ms')}")
PYEOF

echo ""
echo "════════════════════════════════════════════════"
echo "  Benchmark complete."
echo "════════════════════════════════════════════════"
