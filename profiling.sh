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
  -D, --duration SECS      Max runtime cap in seconds (default: 600). Capped at 90s per cell if --matrix is used.
  --prefetch-chunks N      Pass to tecpg (default: 0)
  --blas-threads N         Pass to tecpg via env and args if set
  -g N                     Gene chunk size for tecpg (default: depends on dataset)
  -s N                     Meth chunk size for tecpg (default: depends on dataset)
  --gpu-index N            Value for CUDA_VISIBLE_DEVICES (default: 0)
  --matrix                 Run a small parameter sweep instead of single run (duration capped at 90s per cell)
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
    [ -z "$S_CHUNK" ] && S_CHUNK=100000
    [ -z "$G_CHUNK" ] && G_CHUNK=1000
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
    local duration=$2
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

extract_metrics() {
    local log_file=$1
    local tsv_file=$2
    local sum_file=$3
    local query_file=$4
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
    echo -e "prep_ms\t$(compute_percentiles 1)" >> "$sum_file"
    echo -e "h2d_ms\t$(compute_percentiles 2)" >> "$sum_file"
    echo -e "gpu_ms\t$(compute_percentiles 3)" >> "$sum_file"
    echo -e "d2h_ms\t$(compute_percentiles 4)" >> "$sum_file"
    echo -e "post_ms\t$(compute_percentiles 5)" >> "$sum_file"
    echo -e "write_ms\t$(compute_percentiles 6)" >> "$sum_file"
    echo -e "idle_ms\t$(compute_percentiles 7)" >> "$sum_file"

    local m_prep=$(awk -v col=1 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_h2d=$(awk -v col=2 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_gpu=$(awk -v col=3 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_d2h=$(awk -v col=4 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_post=$(awk -v col=5 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_write=$(awk -v col=6 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')
    local m_idle=$(awk -v col=7 '{print $col}' "$tsv_file" | sort -n | awk -v n="$lines" 'NR == int(n*0.5)+1 {print $1}')

    local avg_util=0
    if [ -s "$query_file" ]; then
        avg_util=$(tail -n +2 "$query_file" | awk -F, '{sum+=$3; count++} END {if (count>0) print sum/count; else print 0}')
    fi

    local throttle_pct=0
    if [ -s "$query_file" ]; then
        local throttle_count
        throttle_count=$(tail -n +2 "$query_file" | awk -F, '
            {
                reason=$NF;
                gsub(/^[ \t]+|[ \t]+$/, "", reason);
                if (reason != "None" && reason != "[Not Supported]" && reason != "0x0000000000000000" && reason != "GpuIdle" && reason != "Idle" && reason != "") count++;
            }
            END {print count+0}
        ')
        local total_query=$(tail -n +2 "$query_file" | wc -l)
        if [ "$total_query" -gt 0 ]; then
            throttle_pct=$(awk -v num="$throttle_count" -v den="$total_query" 'BEGIN{print (num/den)*100}')
        fi
    fi

    echo -e "\nAvg GPU Util: ${avg_util}%" >> "$sum_file"
    echo -e "Throttle events (non-idle): ${throttle_pct}%" >> "$sum_file"

    local reg_sec=0
    local total_wall=0
    if grep -q "PROFILE chunk" "$log_file"; then
        reg_sec=$(grep -a "PROFILE chunk" "$log_file" | tail -n 1 | sed -E 's/.*reg\/s=([^ ]+).*/\1/')
        total_wall=$(awk 'BEGIN{sum=0} {sum+=$1+$2+$3+$4+$5+$6+$7} END{print sum/1000}' "$tsv_file")
    fi

    local verdict="mixed/inconclusive"
    local T
    T=$(awk -v p="$m_prep" -v h="$m_h2d" -v g="$m_gpu" -v d="$m_d2h" -v po="$m_post" -v w="$m_write" 'BEGIN{print p+h+g+d+po+w}')

    local is_starved=$(awk -v i="$m_idle" -v g="$m_gpu" -v u="$avg_util" 'BEGIN{if(i > 0.5*g && u < 30) print 1; else print 0}')
    local is_h2d=$(awk -v h="$m_h2d" -v t="$T" 'BEGIN{if(h > 0.4*t) print 1; else print 0}')
    local is_d2h=$(awk -v d="$m_d2h" -v w="$m_write" -v t="$T" 'BEGIN{if(d+w > 0.4*t) print 1; else print 0}')
    local is_compute=$(awk -v g="$m_gpu" -v t="$T" -v u="$avg_util" 'BEGIN{if(g > 0.6*t && u > 70) print 1; else print 0}')
    local is_launch=$(awk -v g="$m_gpu" -v t="$total_wall" -v chunks="$lines" 'BEGIN{if(t > 0 && g < 2 && chunks/t > 100) print 1; else print 0}')
    local is_thermal=$(awk -v tp="$throttle_pct" 'BEGIN{if(tp > 10) print 1; else print 0}')

    if [ "$is_thermal" -eq 1 ]; then
        verdict="thermal/power throttled"
    elif [ "$is_starved" -eq 1 ]; then
        verdict="GPU is starved (host-bound)"
    elif [ "$is_h2d" -eq 1 ]; then
        verdict="H2D bound"
    elif [ "$is_d2h" -eq 1 ]; then
        verdict="D2H/save bound"
    elif [ "$is_compute" -eq 1 ]; then
        verdict="compute bound"
    elif [ "$is_launch" -eq 1 ]; then
        verdict="kernel-launch bound"
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

    local tecpg_args="run mlr --mlr-method lstsq --$MAPPING --compute-ig"
    if [ -n "$S_CHUNK" ]; then tecpg_args+=" -m $S_CHUNK"; fi
    if [ -n "$G_CHUNK" ]; then tecpg_args+=" -g $G_CHUNK"; fi
    if [ -n "$PREFETCH_CHUNKS" ] && [ "$PREFETCH_CHUNKS" -ne 0 ]; then tecpg_args+=" --prefetch-chunks $PREFETCH_CHUNKS"; fi
    if [ -n "$BLAS_THREADS" ]; then tecpg_args+=" --blas-threads $BLAS_THREADS"; fi
    tecpg_args+=" $extra_args"

    local base_cmd="tecpg --debug -i data_${DATASET} -a annot_${DATASET} -o $cell_dir/tecpg_out $tecpg_args"

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
            py-spy record -o "$cell_dir/pyspy.svg" -p $TECPG_PID -F --idle -d $cap_duration > /dev/null 2>&1 &
            pyspy_pid=$!
            (
                local elapsed=0
                while kill -0 $TECPG_PID 2>/dev/null && [ $elapsed -lt $cap_duration ]; do
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

    local row
    row=$(extract_metrics "$cell_dir/tecpg.log" "$cell_dir/chunk_profile.tsv" "$cell_dir/chunk_profile_summary.txt" "$cell_dir/nvidia-smi-query.csv" "$cell_dir/pidstat.csv")

    echo "$row"
}

capture_environment

if [ "$RUN_MATRIX" -eq 1 ]; then
    log "Running parameter sweep matrix..."

    matrix_csv="$OUT_DIR/matrix_summary.csv"
    echo "Cell,Avg GPU Util,Idle Ratio,Total Wall (s),Reg/s,Verdict" > "$matrix_csv"

    run_and_append() {
        local name=$1
        local args=$2
        local envs=$3
        local baseline=$4
        local dur=$DURATION
        if [ "$dur" -gt 90 ]; then dur=90; fi

        local metrics
        metrics=$(run_workload "$name" "$args" "$envs" "$baseline" "$dur")
        echo "$name,$metrics" >> "$matrix_csv"
    }

    run_and_append "baseline" "" "" 1
    run_and_append "prefetch" "--prefetch-chunks 4" "" 0
    g_dbl=$(( G_CHUNK * 2 ))
    s_dbl=$(( S_CHUNK * 2 ))
    run_and_append "bigger_chunks" "-g $g_dbl -m $s_dbl" "" 0
    run_and_append "tf32" "" "NVIDIA_TF32_OVERRIDE=1 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1" 0
    run_and_append "blas" "--blas-threads 8" "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8" 0
    run_and_append "single_thread" "" "OMP_NUM_THREADS=1" 0

    ln -s "baseline/chunk_profile_summary.txt" "$OUT_DIR/chunk_profile_summary.txt" 2>/dev/null || true
    ln -s "baseline/nvidia-smi-query.csv" "$OUT_DIR/nvidia-smi-query.csv" 2>/dev/null || true
    ln -s "baseline/pidstat.csv" "$OUT_DIR/pidstat.csv" 2>/dev/null || true
    VERDICT=""
    if [ -f "$OUT_DIR/baseline/chunk_profile_summary.txt" ]; then
        VERDICT=$(tail -n 1 "$OUT_DIR/baseline/chunk_profile_summary.txt" | grep "Verdict:" | sed 's/Verdict: //' || true)
    fi

else
    log "Running single workload..."
    VERDICT_ROW=$(run_workload "run" "" "" 1 "$DURATION")
    VERDICT=""
    if [ -f "$OUT_DIR/run/chunk_profile_summary.txt" ]; then
        VERDICT=$(tail -n 1 "$OUT_DIR/run/chunk_profile_summary.txt" | grep "Verdict:" | sed 's/Verdict: //' || true)
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
echo "What to look at first:"
echo "1. chunk_profile_summary.txt (Tells you which stage dominates)"
echo "2. nvidia-smi-query.csv (Look at clocks_throttle_reasons.active for thermal/power capping)"
echo "3. pidstat.csv (Look for CPU-bound producer/save threads)"
echo ""
echo "Archive: $abs_tarball"
echo "SHA256:  $sha256"
echo "========================================================="
