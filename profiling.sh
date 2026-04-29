#!/bin/bash
set -euo pipefail

# profiling.sh - Diagnostic tool for tecpg GPU underutilization
# Run ./profiling.sh --help for usage

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Default settings
DATASET="dummy"
MAPPING="all"
OUT_DIR="./profiling-runs/$(hostname)-$(date -u +'%Y%m%dT%H%M%SZ')"
DURATION=600
PREFETCH_CHUNKS=0
BLAS_THREADS=""
G_CHUNK=""
S_CHUNK=""
GPU_INDEX=0
RUN_MATRIX=0
NO_NSYS=0
KEEP_OUTPUT=0
FORCE_OVERWRITE=0
SUMMARIZE_ONLY=""

declare -a SAMPLER_PIDS=()
CURRENT_CELL_DIR=""

cleanup() {
    local exit_code=$?
    set +e
    if [ ${#SAMPLER_PIDS[@]} -gt 0 ]; then
        for pid in "${SAMPLER_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 1
        for pid in "${SAMPLER_PIDS[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
        SAMPLER_PIDS=()
    fi

    if [ -n "${TECPG_PID:-}" ]; then
        kill "$TECPG_PID" 2>/dev/null || true
    fi

    if [ $exit_code -ne 0 ]; then
        echo "Script exited with error or interruption. Partial output is in: $OUT_DIR"
    fi
    exit $exit_code
}

trap 'cleanup' EXIT INT TERM

show_help() {
    cat << 'HELP'
Usage: ./profiling.sh [OPTIONS]

Diagnostic tool to profile tecpg workloads and isolate GPU bottlenecks.

Options:
  -d, --dataset DATASET    {dummy,gtp,mesa} (default: dummy)
  -m, --mapping MAPPING    {all,promoter} (default: all)
  -o, --output-dir DIR     Directory for artifacts (default: ./profiling-runs/<hostname>-<timestamp>)
  -D, --duration SECS      Max runtime cap in seconds (default: 600).
  --prefetch-chunks N      Pass to tecpg (default: 0)
  --blas-threads N         Pass to tecpg via env and args if set
  -g N                     Gene chunk size for tecpg (default: depends on dataset)
  -s N                     Meth chunk size for tecpg (default: depends on dataset)
  --gpu-index N            Value for CUDA_VISIBLE_DEVICES (default: 0)
  --matrix                 Run a small parameter sweep instead of single run (duration capped at 90s per cell)
  --summarize-only DIR     Skip execution, just parse logs and print summary for existing run dir
  --no-nsys, --no-nvprof   Opt out of heavy profilers
  --keep-output            Keep regression output files (default: delete)
  --force                  Overwrite output directory if it exists
  -h, --help               Show this help
HELP
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--mapping)
            MAPPING="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -D|--duration)
            DURATION="$2"
            shift 2
            ;;
        --prefetch-chunks)
            PREFETCH_CHUNKS="$2"
            shift 2
            ;;
        --blas-threads)
            BLAS_THREADS="$2"
            shift 2
            ;;
        -g)
            G_CHUNK="$2"
            shift 2
            ;;
        -s)
            S_CHUNK="$2"
            shift 2
            ;;
        --gpu-index)
            GPU_INDEX="$2"
            shift 2
            ;;
        --matrix)
            RUN_MATRIX=1
            shift 1
            ;;
        --summarize-only)
            SUMMARIZE_ONLY="$2"
            shift 2
            ;;
        --no-nsys|--no-nvprof)
            NO_NSYS=1
            shift 1
            ;;
        --keep-output)
            KEEP_OUTPUT=1
            shift 1
            ;;
        --force)
            FORCE_OVERWRITE=1
            shift 1
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [ -d "$OUT_DIR" ]; then
    if [ "$FORCE_OVERWRITE" -eq 1 ]; then
        rm -rf "$OUT_DIR"
    else
        echo "Error: Output directory $OUT_DIR already exists. Use --force to overwrite."
        exit 1
    fi
fi

mkdir -p "$OUT_DIR"

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. This script requires an NVIDIA GPU and drivers."
    exit 1
fi

if ! nvidia-smi -i "$GPU_INDEX" &> /dev/null; then
    echo "Error: GPU index $GPU_INDEX is not visible or invalid according to nvidia-smi."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

for tool in pidstat iostat vmstat awk sort tar mktemp; do
    if ! command -v "$tool" &> /dev/null; then
        log "Warning: $tool not found, some metrics will be missing."
    fi
done

if [ "$DATASET" == "gtp" ] || [ "$DATASET" == "mesa" ]; then
    [ -z "$S_CHUNK" ] && S_CHUNK=20000
    [ -z "$G_CHUNK" ] && G_CHUNK=500
else
    [ -z "$S_CHUNK" ] && S_CHUNK=500
    [ -z "$G_CHUNK" ] && G_CHUNK=500
fi

capture_environment() {
    local env_file="$OUT_DIR/env.txt"
    log "Capturing environment into $env_file"
    {
        echo "=== System ==="
        uname -a || true
        lsb_release -a 2>/dev/null || true
        echo "nproc: $(nproc)"
        free -h || true
        lscpu || true
        echo -e "\n=== NVIDIA ==="
        nvidia-smi || true
        nvidia-smi -q || true
        nvidia-smi topo -m 2>/dev/null || true
        echo -e "\n=== Tools ==="
        command -v nvcc >/dev/null && nvcc --version || echo "nvcc not found"
        python3 --version || true
        pip freeze || true
        echo -e "\n=== PyTorch ==="
        python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| cuDNN:', torch.backends.cudnn.version(), '| Device:', torch.cuda.get_device_name(0), '| Capability:', torch.cuda.get_device_capability(0))" || echo "PyTorch info failed"
        echo -e "\n=== Environment Variables ==="
        env | grep -E "CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|TORCH_CUDA_ARCH_LIST|PYTORCH_CUDA_ALLOC_CONF|TECPG_" || true
        echo -e "\n=== Repo State ==="
        git rev-parse HEAD 2>/dev/null || echo "Not a git repo"
        git status --porcelain 2>/dev/null || true
        tecpg --version 2>/dev/null || echo "tecpg command not found"
        echo -e "\n=== Disk Info ==="
        df -h "$OUT_DIR" || true
        echo "Mount type: $(stat -f -c '%T' "$OUT_DIR" 2>/dev/null || echo 'unknown')"
    } > "$env_file"

    local temp_file
    temp_file=$(mktemp "$OUT_DIR/probe.XXXXXX")
    echo -e "\n=== Disk IO Probe (256MB write) ===" >> "$env_file"
    if dd if=/dev/zero of="$temp_file" bs=1M count=256 oflag=direct 2>> "$env_file"; then
        :
    else
        echo "Direct IO probe failed, trying standard IO:" >> "$env_file"
        dd if=/dev/zero of="$temp_file" bs=1M count=256 2>> "$env_file" || true
    fi
    rm -f "$temp_file"
}

start_samplers() {
    local cell_dir=$1
    # shellcheck disable=SC2034
    local duration=$2
    # shellcheck disable=SC2034
    CURRENT_CELL_DIR="$cell_dir"
    SAMPLER_PIDS=()

    nvidia-smi dmon -s pucvmet -d 1 -o DT > "$cell_dir/nvidia-smi-dmon.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)

    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate,clocks_throttle_reasons.active --format=csv -lms 500 > "$cell_dir/nvidia-smi-query.csv" 2>/dev/null &
    SAMPLER_PIDS+=($!)

    if command -v pidstat &> /dev/null; then
        pidstat -h -d -r -u -w -p ALL 1 > "$cell_dir/pidstat.csv" 2>/dev/null &
        SAMPLER_PIDS+=($!)
    fi

    if command -v iostat &> /dev/null; then
        iostat -xmt 1 > "$cell_dir/iostat.txt" 2>/dev/null &
        SAMPLER_PIDS+=($!)
    fi

    if command -v vmstat &> /dev/null; then
        vmstat 1 > "$cell_dir/vmstat.txt" 2>/dev/null &
        SAMPLER_PIDS+=($!)
    fi
}

stop_samplers() {
    if [ ${#SAMPLER_PIDS[@]} -gt 0 ]; then
        for pid in "${SAMPLER_PIDS[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        SAMPLER_PIDS=()
    fi
}

# extract_metrics computes the chunk processing profile and verdict for a run.
#
# The verdict is evaluated in top-to-bottom priority (first match wins):
# 1. save/D2H bound: m_write / T > 0.4
# 2. D2H bound: m_d2h / T > 0.4
# 3. H2D bound: m_h2d / T > 0.4
# 4. thermal/power throttled: m_gpu / T > 0.6 AND avg_util < 70 AND masked_throttle_ratio > 0.2 AND m_gpu / T >= 0.05
# 5. compute bound: m_gpu / T > 0.6 AND avg_util >= 70
# 6. producer/CPU bound: m_idle / T > 0.4
# 7. mixed / inconclusive: otherwise
#
# Note on Throttle Mask:
# Real throttle bits to count are at minimum:
#   0x2 SwPowerCap, 0x4 HwSlowdown, 0x8 SyncBoost, 0x10 SwThermalSlowdown,
#   0x20 HwThermalSlowdown, 0x40 HwPowerBrakeSlowdown, 0x80 DisplayClockSetting.
# Bit 0 (0x1) is GpuIdle, which is not a throttle and happens when waiting on writes.
# We use THROTTLE_MASK = 0xFE to isolate actual throttles.

extract_metrics() {
    local log_file=$1
    local tsv_file=$2
    local sum_file=$3
    local query_file=$4
    # shellcheck disable=SC2034
    local pidstat_file=$5

    if [ ! -f "$log_file" ]; then
        echo "Error: tecpg.log not found. Execution failed." > "$sum_file"
        echo "NA"
        return
    fi

    grep -a "PROFILE chunk" "$log_file" | sed -n 's/.*| //p' | awk '{
        prep="NA"; h2d="NA"; gpu="NA"; d2h="NA"; post="NA"; write="NA"; idle="NA"
        for(i=1; i<=NF; i++) {
            split($i, a, "=")
            val=a[2]
            gsub(/[a-zA-Z/%]+/, "", val)
            if(a[1]=="prep") prep=val
            else if(a[1]=="h2d") h2d=val
            else if(a[1]=="gpu") gpu=val
            else if(a[1]=="d2h") d2h=val
            else if(a[1]=="post") post=val
            else if(a[1]=="write" || a[1]=="write_enqueue_ms") write=val
            else if(a[1]=="idle") idle=val
        }
        print prep, h2d, gpu, d2h, post, write, idle
    }' > "$tsv_file" || true

    if [ ! -s "$tsv_file" ]; then
        echo "No PROFILE lines found in log." > "$sum_file"
        echo "NA"
        return
    fi

    local lines
    lines=$(wc -l < "$tsv_file")

    compute_percentiles() {
        local col=$1
        awk -v col="$col" '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" '
            NR == int(n*0.5)+1 {p50=$1}
            NR == int(n*0.9)+1 {p90=$1}
            NR == int(n*0.99)+1 {p99=$1}
            {sum+=$1}
            END {printf "%.1f\t%.1f\t%.1f\t%.1f\n", p50, p90, p99, sum}
        '
    }

    echo -e "Metric\tp50\tp90\tp99\tTotal" > "$sum_file"
    {
        echo -e "prep_ms\t$(compute_percentiles 1)"
        echo -e "h2d_ms\t$(compute_percentiles 2)"
        echo -e "gpu_ms\t$(compute_percentiles 3)"
        echo -e "d2h_ms\t$(compute_percentiles 4)"
        echo -e "post_ms\t$(compute_percentiles 5)"
        echo -e "write_ms\t$(compute_percentiles 6)"
        echo -e "idle_ms\t$(compute_percentiles 7)"
    } >> "$sum_file"

    local m_prep
    m_prep=$(awk -v col=1 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_h2d
    m_h2d=$(awk -v col=2 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_gpu
    m_gpu=$(awk -v col=3 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_d2h
    m_d2h=$(awk -v col=4 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_post
    m_post=$(awk -v col=5 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_write
    m_write=$(awk -v col=6 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_idle
    m_idle=$(awk -v col=7 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')

    local avg_util=0
    if [ -s "$query_file" ]; then
        avg_util=$(tail -n +2 "$query_file" | awk -F, '{sum+=$3; count++} END {if (count>0) print sum/count; else print 0}')
    fi

    local throttle_pct=0
    local raw_throttle_pct=0
    if [ -s "$query_file" ]; then
        local counts
        counts=$(tail -n +2 "$query_file" | awk -F, '
            BEGIN { THROTTLE_MASK = 0xFE }
            {
                reason=$NF;
                gsub(/^[ \t]+|[ \t]+$/, "", reason);

                if (reason != "None" && reason != "[Not Supported]" && reason != "0x0000000000000000" && reason != "GpuIdle" && reason != "Idle" && reason != "") {
                    raw_count++;
                }

                if (reason ~ /^0x/) {
                    if (and(strtonum(reason), THROTTLE_MASK) != 0) {
                        masked_count++;
                    }
                }
            }
            END { print raw_count+0, masked_count+0 }
        ')
        local raw_throttle_count
        raw_throttle_count=$(echo "$counts" | awk '{print $1}')
        local throttle_count
        throttle_count=$(echo "$counts" | awk '{print $2}')
        local total_query
        total_query=$(tail -n +2 "$query_file" | wc -l)
        if [ "$total_query" -gt 0 ]; then
            throttle_pct=$(awk -v num="$throttle_count" -v den="$total_query" 'BEGIN{print (num/den)*100}')
            raw_throttle_pct=$(awk -v num="$raw_throttle_count" -v den="$total_query" 'BEGIN{print (num/den)*100}')
        fi
    fi

    {
        echo -e "\nAvg GPU Util: ${avg_util}%"
        echo -e "Throttle events (masked, real throttles): ${throttle_pct}%"
        echo -e "Throttle events (raw, unmasked): ${raw_throttle_pct}%"
    } >> "$sum_file"

    local reg_sec=0
    local total_wall=0
    if grep -q "PROFILE chunk" "$log_file"; then
        reg_sec=$(grep -a "PROFILE chunk" "$log_file" | tail -n 1 | sed -E 's/.*reg\/s=([^ ]+).*/\1/')
        total_wall=$(awk 'BEGIN{sum=0} {sum+=$1+$2+$3+$4+$5+$6+$7} END{print sum/1000}' "$tsv_file")
    fi

    local verdict="mixed/inconclusive"
    local T
    T=$(awk -v p="$m_prep" -v h="$m_h2d" -v g="$m_gpu" -v d="$m_d2h" -v po="$m_post" -v w="$m_write" 'BEGIN{print p+h+g+d+po+w}')

    local is_save_bound
    is_save_bound=$(awk -v w="$m_write" -v t="$T" 'BEGIN{if(t>0 && w/t > 0.4) print 1; else print 0}')
    local is_d2h
    is_d2h=$(awk -v d="$m_d2h" -v t="$T" 'BEGIN{if(t>0 && d/t > 0.4) print 1; else print 0}')
    local is_h2d
    is_h2d=$(awk -v h="$m_h2d" -v t="$T" 'BEGIN{if(t>0 && h/t > 0.4) print 1; else print 0}')
    local is_thermal
    is_thermal=$(awk -v g="$m_gpu" -v t="$T" -v u="$avg_util" -v tp="$throttle_pct" 'BEGIN{if(t>0 && g/t > 0.6 && u < 70 && tp > 20 && g/t >= 0.05) print 1; else print 0}')
    local is_compute
    is_compute=$(awk -v g="$m_gpu" -v t="$T" -v u="$avg_util" 'BEGIN{if(t>0 && g/t > 0.6 && u >= 70) print 1; else print 0}')
    local is_starved
    is_starved=$(awk -v i="$m_idle" -v t="$T" 'BEGIN{if(t>0 && i/t > 0.4) print 1; else print 0}')

    if [ "$is_save_bound" -eq 1 ]; then
        verdict="save/D2H bound"
    elif [ "$is_d2h" -eq 1 ]; then
        verdict="D2H bound"
    elif [ "$is_h2d" -eq 1 ]; then
        verdict="H2D bound"
    elif [ "$is_thermal" -eq 1 ]; then
        verdict="thermal/power throttled"
    elif [ "$is_compute" -eq 1 ]; then
        verdict="compute bound"
    elif [ "$is_starved" -eq 1 ]; then
        verdict="producer/CPU bound"
    else
        local top2
        top2=$(echo -e "$m_prep prep\n$m_h2d h2d\n$m_gpu gpu\n$m_d2h d2h\n$m_post post\n$m_write write" | sort -nr | head -n 2 | awk '{print $2}' | paste -sd "/" -)
        verdict="mixed/inconclusive (top components: $top2)"
    fi

    echo -e "\nVerdict: $verdict" >> "$sum_file"
    echo -e "T: $T ms | Idle: $m_idle ms | GPU: $m_gpu ms | Util: ${avg_util}%" >> "$sum_file"

    echo "${avg_util},$(awk -v i="$m_idle" -v t="$T" 'BEGIN{if(t>0) print i/t; else print 0}'),${total_wall},${reg_sec},${verdict}"
}

run_workload() {
    local cell_name=$1
    local extra_args=$2
    local extra_env=$3
    local is_baseline=$4
    local cap_duration=$5

    local cell_dir="$OUT_DIR/$cell_name"
    mkdir -p "$cell_dir"

    log "Running cell: $cell_name"

    local tecpg_global_args="--debug -i data_${DATASET} -a annot_${DATASET} -o $cell_dir/tecpg_out"
    if [ -n "$BLAS_THREADS" ]; then tecpg_global_args+=" --blas-threads $BLAS_THREADS"; fi

    local tecpg_args="run mlr --mlr-method lstsq --$MAPPING --compute-ig"
    if [ -n "$S_CHUNK" ]; then tecpg_args+=" -m $S_CHUNK"; fi
    if [ -n "$G_CHUNK" ]; then tecpg_args+=" -g $G_CHUNK"; fi
    if [ -n "$PREFETCH_CHUNKS" ] && [ "$PREFETCH_CHUNKS" -ne 0 ]; then tecpg_args+=" --prefetch-chunks $PREFETCH_CHUNKS"; fi
    tecpg_args+=" $extra_args"

    local base_cmd="tecpg $tecpg_global_args $tecpg_args"

    local env_cmd="env TECPG_PROFILE=1 CUDA_LAUNCH_BLOCKING=${TECPG_PROFILING_BLOCKING:-0} $extra_env"

    echo "Cell: $cell_name" > "$cell_dir/cell.txt"
    echo "Env: $env_cmd" >> "$cell_dir/cell.txt"
    echo "Cmd: timeout $cap_duration $base_cmd" >> "$cell_dir/cell.txt"

    if [ "$DATASET" == "dummy" ] && [ ! -d "data_dummy" ]; then
        log "Generating temporary dummy data for profiling..."
        echo "10" | tecpg data dummy -s 100 -m 1000 -g 1000 > /dev/null || true
        mkdir -p data_dummy annot_dummy
        mv data/* data_dummy/ 2>/dev/null || true
        mv annot/* annot_dummy/ 2>/dev/null || true
        rmdir data annot 2>/dev/null || true
        mv data_dummy/C.csv data_dummy/C_orig.csv 2>/dev/null || true
        cp data_dummy/C_orig.csv data_dummy/C.csv 2>/dev/null || true
    fi

    start_samplers "$cell_dir" "$cap_duration"

    local nsys_cmd=""
    if [ "$is_baseline" -eq 1 ] && [ "$NO_NSYS" -eq 0 ]; then
        if command -v nsys &> /dev/null; then
            local nsys_dur=$(( cap_duration < 180 ? cap_duration : 180 ))
            nsys_cmd="nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o $cell_dir/nsys --force-overwrite=true --duration=$nsys_dur "
        fi
    fi

    local run_cmd="$env_cmd $nsys_cmd timeout $cap_duration $base_cmd"

    log "Executing: $run_cmd"

    set +e
    eval "$run_cmd" > "$cell_dir/tecpg.log" 2>&1 &
    TECPG_PID=$!

    local pyspy_pid=""
    if [ "$is_baseline" -eq 1 ] && [ "$NO_NSYS" -eq 0 ]; then
        if command -v py-spy &> /dev/null; then
            py-spy record -o "$cell_dir/pyspy.svg" -p $TECPG_PID -F --idle -d "$cap_duration" > /dev/null 2>&1 &
            pyspy_pid=$!
            (
                local elapsed=0
                while kill -0 $TECPG_PID 2>/dev/null && [ $elapsed -lt "$cap_duration" ]; do
                    sleep 30
                    if kill -0 $TECPG_PID 2>/dev/null; then
                        py-spy dump -p $TECPG_PID >> "$cell_dir/pyspy_dumps.txt" 2>/dev/null || true
                    fi
                    elapsed=$((elapsed+30))
                done
            ) &
        else
            log "py-spy not installed. Hint: pip install py-spy to get CPU flamegraphs."
        fi
    fi

    top -b -d 1 -H -p $TECPG_PID > "$cell_dir/top.txt" 2>/dev/null &
    local top_pid=$!

    wait $TECPG_PID
    local tecpg_exit=$?
    TECPG_PID=""
    set -e

    if [ $tecpg_exit -eq 124 ]; then
        log "tecpg timed out after $cap_duration seconds as expected."
    elif [ $tecpg_exit -ne 0 ]; then
        log "Warning: tecpg exited with code $tecpg_exit"
        log "Last 20 lines of tecpg.log:"
        tail -n 20 "$cell_dir/tecpg.log" 2>/dev/null || true
    fi

    kill $top_pid 2>/dev/null || true
    if [ -n "$pyspy_pid" ]; then kill $pyspy_pid 2>/dev/null || true; fi

    stop_samplers

    if [ "$KEEP_OUTPUT" -eq 0 ]; then
        rm -rf "$cell_dir/tecpg_out" 2>/dev/null || true
    fi
}

capture_environment

if [ -n "$SUMMARIZE_ONLY" ]; then
    OUT_DIR="$SUMMARIZE_ONLY"
    log "Summarizing existing run directory: $OUT_DIR"
    if [ ! -d "$OUT_DIR" ]; then
        log "Error: Directory $OUT_DIR does not exist."
        # shellcheck disable=SC2317
        return 1 2>/dev/null || kill -INT $$
    fi

    matrix_csv="$OUT_DIR/matrix_summary.csv"
    echo "Cell,Avg GPU Util,Idle Ratio,Total Wall (s),Reg/s,Verdict" > "$matrix_csv"

    for cell_dir in "$OUT_DIR"/*/; do
        if [ ! -d "$cell_dir" ]; then continue; fi
        name=$(basename "$cell_dir")
        log "Extracting metrics for $name..."
        metrics=$(extract_metrics "$cell_dir/tecpg.log" "$cell_dir/chunk_profile.tsv" "$cell_dir/chunk_profile_summary.txt" "$cell_dir/nvidia-smi-query.csv" "$cell_dir/pidstat.csv")
        echo "$name,$metrics" >> "$matrix_csv"
    done

    if [ -f "$OUT_DIR/baseline/chunk_profile_summary.txt" ]; then
        ln -sf "baseline/chunk_profile_summary.txt" "$OUT_DIR/chunk_profile_summary.txt" 2>/dev/null || true
        ln -sf "baseline/nvidia-smi-query.csv" "$OUT_DIR/nvidia-smi-query.csv" 2>/dev/null || true
        ln -sf "baseline/pidstat.csv" "$OUT_DIR/pidstat.csv" 2>/dev/null || true
        VERDICT=$(grep "Verdict:" "$OUT_DIR/baseline/chunk_profile_summary.txt" | sed 's/Verdict: //' || true)
    elif [ -f "$OUT_DIR/run/chunk_profile_summary.txt" ]; then
        ln -sf "run/chunk_profile_summary.txt" "$OUT_DIR/chunk_profile_summary.txt" 2>/dev/null || true
        ln -sf "run/nvidia-smi-query.csv" "$OUT_DIR/nvidia-smi-query.csv" 2>/dev/null || true
        ln -sf "run/pidstat.csv" "$OUT_DIR/pidstat.csv" 2>/dev/null || true
        VERDICT=$(grep "Verdict:" "$OUT_DIR/run/chunk_profile_summary.txt" | sed 's/Verdict: //' || true)
    fi

    RUN_MATRIX=1 # To trigger the matrix summary output block

elif [ "$RUN_MATRIX" -eq 1 ]; then
    log "Running parameter sweep matrix..."

    matrix_csv="$OUT_DIR/matrix_summary.csv"
    echo "Cell,Avg GPU Util,Idle Ratio,Total Wall (s),Reg/s,Verdict" > "$matrix_csv"

    run_and_append() {
        local name=$1
        local args=$2
        local envs=$3
        local baseline=$4
        local dur=$DURATION


        run_workload "$name" "$args" "$envs" "$baseline" "$dur"
        local cell_dir="$OUT_DIR/$name"
        local metrics
        metrics=$(extract_metrics "$cell_dir/tecpg.log" "$cell_dir/chunk_profile.tsv" "$cell_dir/chunk_profile_summary.txt" "$cell_dir/nvidia-smi-query.csv" "$cell_dir/pidstat.csv")
        echo "$name,$metrics" >> "$matrix_csv"
    }

    run_and_append "baseline" "" "" 1
    run_and_append "prefetch" "--prefetch-chunks 4" "" 0
    g_dbl=$(( G_CHUNK * 2 ))
    s_dbl=$(( S_CHUNK * 2 ))
    if [ "$g_dbl" -gt 1000 ]; then g_dbl=1000; fi
    if [ "$s_dbl" -gt 40000 ]; then s_dbl=40000; fi

    OLD_G_CHUNK="$G_CHUNK"
    OLD_S_CHUNK="$S_CHUNK"
    G_CHUNK="$g_dbl"
    S_CHUNK="$s_dbl"
    run_and_append "bigger_chunks" "" "" 0
    G_CHUNK="$OLD_G_CHUNK"
    S_CHUNK="$OLD_S_CHUNK"
    run_and_append "tf32" "" "NVIDIA_TF32_OVERRIDE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1" 0

    # Store old BLAS_THREADS to temporarily override it for this cell
    OLD_BLAS_THREADS="$BLAS_THREADS"
    BLAS_THREADS="8"
    run_and_append "blas" "" "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8" 0
    BLAS_THREADS="$OLD_BLAS_THREADS"

    run_and_append "single_thread" "" "OMP_NUM_THREADS=1" 0

    ln -s "baseline/chunk_profile_summary.txt" "$OUT_DIR/chunk_profile_summary.txt" 2>/dev/null || true
    ln -s "baseline/nvidia-smi-query.csv" "$OUT_DIR/nvidia-smi-query.csv" 2>/dev/null || true
    ln -s "baseline/pidstat.csv" "$OUT_DIR/pidstat.csv" 2>/dev/null || true
    VERDICT=""
    if [ -f "$OUT_DIR/baseline/chunk_profile_summary.txt" ]; then
        VERDICT=$(grep "Verdict:" "$OUT_DIR/baseline/chunk_profile_summary.txt" | sed 's/Verdict: //' || true)
    fi

else
    log "Running single workload..."
    run_workload "run" "" "" 1 "$DURATION"
    # shellcheck disable=SC2034
    VERDICT_ROW=$(extract_metrics "$OUT_DIR/run/tecpg.log" "$OUT_DIR/run/chunk_profile.tsv" "$OUT_DIR/run/chunk_profile_summary.txt" "$OUT_DIR/run/nvidia-smi-query.csv" "$OUT_DIR/run/pidstat.csv")
    VERDICT=""
    if [ -f "$OUT_DIR/run/chunk_profile_summary.txt" ]; then
        VERDICT=$(grep "Verdict:" "$OUT_DIR/run/chunk_profile_summary.txt" | sed 's/Verdict: //' || true)
    fi
    ln -s "run/chunk_profile_summary.txt" "$OUT_DIR/chunk_profile_summary.txt" 2>/dev/null || true
    ln -s "run/nvidia-smi-query.csv" "$OUT_DIR/nvidia-smi-query.csv" 2>/dev/null || true
    ln -s "run/pidstat.csv" "$OUT_DIR/pidstat.csv" 2>/dev/null || true
fi

if [ -z "${VERDICT:-}" ]; then
    VERDICT="Failed to extract verdict (no PROFILE lines found)"
fi

tarball="${OUT_DIR}.tar.gz"
log "Bundling artifacts to $tarball"
tar -czf "$tarball" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"

abs_tarball=$(realpath "$tarball")
sha256=$(sha256sum "$abs_tarball" | awk '{print $1}')

echo "========================================================="
echo "Profiling Complete!"
echo "Verdict: $VERDICT"
echo ""

if [ "$RUN_MATRIX" -eq 1 ] && [ -f "$OUT_DIR/matrix_summary.csv" ]; then
    echo "Cell Summary:"
    # shellcheck disable=SC2034
    tail -n +2 "$OUT_DIR/matrix_summary.csv" | while IFS=, read -r cell util idle wall reg verdict; do
        if [[ "$verdict" == *"Failed"* ]] || [[ "$verdict" == "NA" ]]; then
            echo "  - $cell: Failed"
        else
            echo "  - $cell: Completed"
        fi
    done
    echo ""
fi

echo "What to look at first:"
echo "1. chunk_profile_summary.txt (Tells you which stage dominates)"
echo "2. nvidia-smi-query.csv (Look at clocks_throttle_reasons.active for thermal/power capping; note GpuIdle is masked out in summary)"
echo "3. pidstat.csv (Look for CPU-bound producer/save threads)"
echo ""
echo "Archive: $abs_tarball"
echo "SHA256:  $sha256"
echo "========================================================="
