#!/bin/bash
set -e

# Log function for timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Default settings
DATASET="dummy"
TOTAL_TESTS=1000000
M_CHUNK=500
G_CHUNK=500

# Parse arguments
if [ "$#" -ge 1 ]; then
    DATASET=$1
fi

if [ "$DATASET" == "gtp" ]; then
    TOTAL_TESTS=13744315260 # Approximate full GTP size
    M_CHUNK=5000
    G_CHUNK=100
elif [ "$DATASET" == "mesa" ]; then
    TOTAL_TESTS=10000000000 # Placeholder for MESA
    M_CHUNK=5000
    G_CHUNK=100
elif [ "$DATASET" == "dummy" ]; then
    TOTAL_TESTS=1000000 # 1000 M * 1000 G
    M_CHUNK=500
    G_CHUNK=500
else
    log "Error: Unknown dataset: $DATASET"
    log "Usage: ./pipeline.sh [gtp|mesa|dummy]"
    exit 1
fi

log "======================================"
log "Starting eQTM Pipeline for: $DATASET"
log "Dataset configurations: TOTAL_TESTS=$TOTAL_TESTS, M_CHUNK=$M_CHUNK, G_CHUNK=$G_CHUNK"
log "======================================"

# Setup directories
OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
ANNOT_DIR="annot_${DATASET}"
mkdir -p "$OUT_DIR" "$DATA_DIR" "$ANNOT_DIR"

# Stage 1: Data Preparation
log "[1/8] Preparing data..."
log "Checking if dataset files already exist in $DATA_DIR..."

if [ -s "$DATA_DIR/M.csv" ] && [ -s "$DATA_DIR/G.csv" ] && [ -s "$DATA_DIR/C.csv" ]; then
    log "Data files (M.csv, G.csv, C.csv) already exist and are not empty. Skipping download/generation."
else
    log "Data files not found or empty. Proceeding with data generation/download for $DATASET..."
    if [ "$DATASET" == "dummy" ]; then
        # Generate small synthetic data for testing
        log "Generating synthetic dummy data..."
        echo "10" | tecpg data dummy -s 100 -m 1000 -g 1000
        mv data/* "$DATA_DIR/"
        mv annot/* "$ANNOT_DIR/"
        rmdir data annot
    elif [ "$DATASET" == "gtp" ]; then
        log "Downloading GTP data..."
        echo "y" | tecpg data gtp --yes
        mv data/* "$DATA_DIR/"
        # For GTP, assuming the demo annots are used
        cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6"
        cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6"
        rmdir data
    elif [ "$DATASET" == "mesa" ]; then
        log "Downloading MESA data..."
        echo "y" | tecpg data mesa
        mv data/* "$DATA_DIR/"
        # For MESA, assuming appropriate demo annots are used if available
        # Or fall back to EPIC/HT12 for now
        cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6" 2>/dev/null || true
        cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6" 2>/dev/null || true
        rmdir data
    fi
fi

# Determine Degrees of Freedom for P-value calculation
# SAMPLES - COVARIATES - 1 (M) - 1 (Intercept)
# Using placeholder calculation if needed; assuming df=96 for dummy (100 - 2 - 2)
# Here we just parse the C.csv lines dynamically
SAMPLES=$(wc -l < "$DATA_DIR/C.csv")
SAMPLES=$((SAMPLES - 1)) # Header
COVARS=$(head -n 1 "$DATA_DIR/C.csv" | awk -F, '{print NF-1}')
DF=$((SAMPLES - COVARS - 2))

log "Calculated Degrees of Freedom (DF): $DF (SAMPLES=$SAMPLES, COVARS=$COVARS)"

# Stage 2: Mapping (lstsq + ig)
log "[2/8] Performing eQTM Mapping (lstsq + IG)..."
log "This stage runs the multiple linear regression (mlr) model and computes Integrated Gradients (IG)."
log "Using chunks: M_CHUNK=$M_CHUNK, G_CHUNK=$G_CHUNK. Input: $DATA_DIR, Annotations: $ANNOT_DIR, Output: $OUT_DIR"
tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq --all -m "$M_CHUNK" -g "$G_CHUNK" --compute-ig

# Stage 3: Merge chunked outputs
log "[3/8] Merging chunked outputs to Parquet..."
MERGED_PARQUET="$OUT_DIR/merged.parquet"
log "Converting CSV chunks in $OUT_DIR into a single Parquet file at $MERGED_PARQUET..."
python3 tools/mergeOutputs.py --format parquet "$OUT_DIR" "$MERGED_PARQUET"

# Clean up CSV chunks to save space
log "Cleaning up intermediate CSV chunks..."
rm "$OUT_DIR"/*-*.csv || true

# Stage 4: Annotate regions
log "[4/8] Annotating regions..."
ANNOTATED_PARQUET="$OUT_DIR/annotated.parquet"
log "Mapping eCpG and Gene coordinates to determine regional categories (e.g., CIS, TRANS)."
log "Input Parquet: $MERGED_PARQUET, Output Parquet: $ANNOTATED_PARQUET"
python3 tools/assignRegionToEcpg_parquet.py -d "$MERGED_PARQUET" -g "$ANNOT_DIR/G.bed6" -m "$ANNOT_DIR/M.bed6" -o "$ANNOTATED_PARQUET"

# Stage 5: Precise P-value recalculation
log "[5/8] Recalculating precise p-values..."
RECALC_PARQUET="$OUT_DIR/annotated_pcalc.parquet"
log "Calculating high-precision p-values using DF=$DF."
log "Input Parquet: $ANNOTATED_PARQUET, Output Parquet: $RECALC_PARQUET"
python3 tools/recalculate_pvalues_parquet.py "$ANNOTATED_PARQUET" --df "$DF" --output-file "$RECALC_PARQUET"

# Stage 6: Summarize & FDR
log "[6/8] Calculating FDR and summarizing..."
SUMMARIZED_PARQUET="$OUT_DIR/summarized.parquet"
log "Estimating global Benjamini-Hochberg FDR based on TOTAL_TESTS=$TOTAL_TESTS."
log "Generating diagnostic plots (QQ, Histogram, Saliency)."
log "Input Parquet: $RECALC_PARQUET, Output Parquet: $SUMMARIZED_PARQUET"
python3 tools/summarizeOutput_parquet.py --main-file "$RECALC_PARQUET" --reservoir-file "$OUT_DIR/sample_reservoir.csv" --total-tests "$TOTAL_TESTS" --df "$DF" --calculate-fdr --output-fdr-file "$SUMMARIZED_PARQUET"
# Ensure plots are created in the right folder, but for now they go to CWD based on tool
log "Moving generated plots to $OUT_DIR..."
mv p_value_histogram.png qq_plot.png saliency_profile_top50.png "$OUT_DIR/" 2>/dev/null || true

# Stage 7: Bootstrap List creation
log "[7/8] Creating Bootstrap List..."
BOOTSTRAP_LIST="$OUT_DIR/bootstrap_list.csv"
log "Identifying top hits (ranked by p-value) to be evaluated via bootstrapping."
log "Input Parquet: $SUMMARIZED_PARQUET, Output List: $BOOTSTRAP_LIST"
python3 tools/createBootstrapList.py --input "$SUMMARIZED_PARQUET" --output "$BOOTSTRAP_LIST" --rank-by p-value

# Stage 8: Bootstrap evaluation
log "[8/8] Bootstrapping top hits..."
log "Running bootstrap analysis on the top candidates to validate association robustness."
log "Pairs File: $BOOTSTRAP_LIST, Master Parquet: $SUMMARIZED_PARQUET"
tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq_bootstrap --pairs-file "$BOOTSTRAP_LIST" --master-parquet "$SUMMARIZED_PARQUET" --bootstrap-iterations 100 --bootstrap-batch-size 10 --compute-ig

log "======================================"
log "Pipeline completed successfully!"
log "Outputs are in $OUT_DIR/"
log "======================================"
