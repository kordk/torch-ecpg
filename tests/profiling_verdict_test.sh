#!/bin/bash
set -euo pipefail

PROFILING_SCRIPT="$(dirname "$0")/../profiling.sh"

awk '/^extract_metrics\(\) \{/,/^}/' "$PROFILING_SCRIPT" > extract_metrics_mock.sh

source extract_metrics_mock.sh

WORK_DIR=$(mktemp -d)

run_test() {
    local name=$1
    local log_content=$2
    local query_content=$3
    local expected_verdict=$4

    mkdir -p "$WORK_DIR/$name"

    echo "$log_content" > "$WORK_DIR/$name/tecpg.log"
    echo "[DEBUG] PROFILE chunk | reg/s=1000" >> "$WORK_DIR/$name/tecpg.log"

    echo "timestamp, name, utilization.gpu [%], clocks_throttle_reasons.active" > "$WORK_DIR/$name/nvidia-smi-query.csv"
    echo "$query_content" >> "$WORK_DIR/$name/nvidia-smi-query.csv"

    touch "$WORK_DIR/$name/pidstat.csv"

    local metrics
    metrics=$(extract_metrics "$WORK_DIR/$name/tecpg.log" "$WORK_DIR/$name/chunk_profile.tsv" "$WORK_DIR/$name/chunk_profile_summary.txt" "$WORK_DIR/$name/nvidia-smi-query.csv" "$WORK_DIR/$name/pidstat.csv")

    local actual_verdict
    actual_verdict=$(echo "$metrics" | awk -F, '{print $5}')

    if [ "$actual_verdict" != "$expected_verdict" ]; then
        echo "Test Failed: $name"
        echo "Expected: $expected_verdict"
        echo "Actual:   $actual_verdict"
        rm -rf "$WORK_DIR" extract_metrics_mock.sh
        # kill to fail test script
        kill -INT $$
    else
        echo "Test Passed: $name -> $actual_verdict"
    fi
}

echo "Running profiling_verdict tests..."

# Test 1: save/D2H bound
log1="[DEBUG] PROFILE chunk | prep=10.0 h2d=10.0 gpu=10.0 d2h=10.0 post=10.0 write=60.0 idle=0.0"
query1="2026/04/29 02:00:23.000, L4, 10 %, 0x0000000000000000"
run_test "save_bound" "$log1" "$query1" "save/D2H bound"

# Test 2: H2D bound
log2="[DEBUG] PROFILE chunk | prep=10.0 h2d=60.0 gpu=10.0 d2h=10.0 post=10.0 write=10.0 idle=0.0"
query2="2026/04/29 02:00:23.000, L4, 10 %, 0x0000000000000000"
run_test "h2d_bound" "$log2" "$query2" "H2D bound"

# Test 3: compute bound
log3="[DEBUG] PROFILE chunk | prep=10.0 h2d=10.0 gpu=100.0 d2h=10.0 post=10.0 write=10.0 idle=0.0"
query3="2026/04/29 02:00:23.000, L4, 80 %, 0x0000000000000000"
run_test "compute_bound" "$log3" "$query3" "compute bound"

# Test 4: thermal/power throttled (using 0x10 = SwThermalSlowdown mask)
log4="[DEBUG] PROFILE chunk | prep=10.0 h2d=10.0 gpu=100.0 d2h=10.0 post=10.0 write=10.0 idle=0.0"
query4="2026/04/29 02:00:23.000, L4, 50 %, 0x0000000000000010"
run_test "thermal_throttled" "$log4" "$query4" "thermal/power throttled"

# Test 5: producer/CPU bound (idle high)
log5="[DEBUG] PROFILE chunk | prep=10.0 h2d=10.0 gpu=10.0 d2h=10.0 post=10.0 write=10.0 idle=50.0"
query5="2026/04/29 02:00:23.000, L4, 10 %, 0x0000000000000000"
run_test "producer_bound" "$log5" "$query5" "producer/CPU bound"

# Test 6: not throttled (GpuIdle bit 0x1 only) should NOT be thermal throttled even if util < 70 and gpu > 60
log6="[DEBUG] PROFILE chunk | prep=20.0 h2d=10.0 gpu=100.0 d2h=10.0 post=10.0 write=10.0 idle=0.0"
query6="2026/04/29 02:00:23.000, L4, 50 %, 0x0000000000000001"
run_test "gpu_idle_not_throttled" "$log6" "$query6" "mixed/inconclusive (top components: gpu/prep)"

echo "All tests passed!"
rm -rf "$WORK_DIR" extract_metrics_mock.sh
