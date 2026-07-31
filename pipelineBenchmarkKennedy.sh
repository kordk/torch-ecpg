#!/bin/bash
set -e

# Configurable defaults
EXPLORATORY_THRESH=1e-5
PRIMARY_THRESH=1e-11
BATCH_SIZE=500000

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "Example: $0 gtp"
    echo "         $0 mesa"
    exit 1
fi

DATASET=$1
OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
CATALOG="${OUT_DIR}/bootstrap_merged.parquet"
BENCH_DIR="${OUT_DIR}/kennedy"

DATASET_UC=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')
KENNEDY="${DATA_DIR}/eCpGs_Kennedy2018_${DATASET_UC}.txt"

log "======================================"
log "Starting Benchmark Kennedy Pipeline for DATASET: ${DATASET}"
log "======================================"

if [ ! -f "$CATALOG" ]; then
    log "Error: Expected input file $CATALOG not found!"
    log "Please ensure the tecpg pipeline has successfully generated bootstrap_merged.parquet."
    exit 1
fi

if [ ! -f "$KENNEDY" ]; then
    log "Error: Expected input file $KENNEDY not found!"
    log "The Kennedy supplementary file is downloaded as part of dataset acquisition; it is not produced by the pipeline."
    exit 1
fi

# Ensure output directory exists
mkdir -p "${BENCH_DIR}"

# Stage 1: Characterization
# Note: benchmark_kennedy.py requires BOTH -t and -k even for --characterize
log "[1/4] Characterization..."
python3 -u tools/benchmark_kennedy.py \
    -t "$CATALOG" \
    -k "$KENNEDY" \
    -o "${BENCH_DIR}/characterize/" \
    --batch-size "$BATCH_SIZE" \
    --characterize

# Stage 2: Exploratory comparison at the 1e-5 diagonal
# This run produces the larger pairs_tecpg_only.tsv used by tools/runEnrichment.py
log "[2/4] Exploratory comparison at the 1e-5 diagonal..."
python3 -u tools/benchmark_kennedy.py \
    -t "$CATALOG" \
    -k "$KENNEDY" \
    -o "${BENCH_DIR}/t1e5_k1e5/" \
    --tecpg-thresh "$EXPLORATORY_THRESH" \
    --kennedy-thresh "$EXPLORATORY_THRESH" \
    --batch-size "$BATCH_SIZE"

# Stage 3: Primary comparison at the 1e-11 diagonal
# This run is the like-for-like primary comparison, stored separately so it doesn't overwrite the exploratory run
log "[3/4] Primary comparison at the 1e-11 diagonal..."
python3 -u tools/benchmark_kennedy.py \
    -t "$CATALOG" \
    -k "$KENNEDY" \
    -o "${BENCH_DIR}/t1e11_k1e11/" \
    --tecpg-thresh "$PRIMARY_THRESH" \
    --kennedy-thresh "$PRIMARY_THRESH" \
    --batch-size "$BATCH_SIZE"

# Stage 4: Summary
log "[4/4] Summary..."

log "Artifacts in Characterization:"
if [ -d "${BENCH_DIR}/characterize/" ]; then
    ls -lh "${BENCH_DIR}/characterize/"
fi
if [ -f "${BENCH_DIR}/characterize/benchmark_report.html" ]; then
    log "HTML Report: ${BENCH_DIR}/characterize/benchmark_report.html"
fi

log "Artifacts in 1e-5 Exploratory:"
if [ -d "${BENCH_DIR}/t1e5_k1e5/" ]; then
    ls -lh "${BENCH_DIR}/t1e5_k1e5/"
fi
if [ -f "${BENCH_DIR}/t1e5_k1e5/benchmark_report.html" ]; then
    log "HTML Report: ${BENCH_DIR}/t1e5_k1e5/benchmark_report.html"
fi

log "Artifacts in 1e-11 Primary:"
if [ -d "${BENCH_DIR}/t1e11_k1e11/" ]; then
    ls -lh "${BENCH_DIR}/t1e11_k1e11/"
fi
if [ -f "${BENCH_DIR}/t1e11_k1e11/benchmark_report.html" ]; then
    log "HTML Report: ${BENCH_DIR}/t1e11_k1e11/benchmark_report.html"
fi

log "======================================"
log "Benchmark Kennedy pipeline completed successfully!"
log "======================================"
