#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# Log function for timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Default settings
DATASET="dummy"
MAPPING="all"
TOTAL_TESTS=1000000
M_CHUNK=500
G_CHUNK=500
NUM_PCS=5

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: ./pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --help               Show this help message and exit"
            echo "  -d, --dataset DATASET    Specify the dataset to use. Options: dummy (default), gtp, mesa"
            echo "  -m, --mapping MAPPING    Specify the mapping method for tecpg. Options: all (default), promoter"
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
        *)
            echo "Unknown parameter passed: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

if [ "$MAPPING" != "all" ] && [ "$MAPPING" != "promoter" ]; then
    log "Error: Unknown mapping: $MAPPING"
    log "Usage: ./pipeline.sh --dataset [dummy|gtp|mesa] --mapping [all|promoter]"
    exit 1
fi

if [ "$DATASET" == "gtp" ]; then
    TOTAL_TESTS=13744315260 # Placeholder for full GTP size, will be dynamically updated
    M_CHUNK=15000
    G_CHUNK=1000
elif [ "$DATASET" == "mesa" ]; then
    TOTAL_TESTS=10000000000 # Placeholder for MESA, will be dynamically updated
    M_CHUNK=20000
    G_CHUNK=1000
elif [ "$DATASET" == "dummy" ]; then
    TOTAL_TESTS=1000000 # Placeholder for 1000 M * 1000 G, will be dynamically updated
    M_CHUNK=500
    G_CHUNK=500
else
    log "Error: Unknown dataset: $DATASET"
    log "Usage: ./pipeline.sh --dataset [dummy|gtp|mesa] --mapping [all|promoter]"
    exit 1
fi

log "======================================"
log "Starting eQTM Pipeline for: $DATASET (Mapping: $MAPPING)"
log "Dataset configurations: TOTAL_TESTS=$TOTAL_TESTS, M_CHUNK=$M_CHUNK, G_CHUNK=$G_CHUNK"
log "======================================"

# Setup directories
OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
ANNOT_DIR="annot_${DATASET}"
mkdir -p "$OUT_DIR" "$DATA_DIR" "$ANNOT_DIR"

# Stage 1: Data Preparation
log "[1/9] Preparing data..."
log "Checking if dataset files already exist in $DATA_DIR..."

if ( [ -s "$DATA_DIR/M_orig.csv" ] || [ -s "$DATA_DIR/M.csv" ] ) && [ -s "$DATA_DIR/G.csv" ] && ( [ -s "$DATA_DIR/C_orig.csv" ] || [ -s "$DATA_DIR/C.csv" ] ); then
    log "Data files (M_orig.csv/M.csv, G.csv, and C_orig.csv/C.csv) already exist and are not empty. Skipping download/generation."
    if [ -s "$DATA_DIR/C.csv" ] && [ ! -s "$DATA_DIR/C_orig.csv" ]; then
        log "Found C.csv but not C_orig.csv. Renaming C.csv to C_orig.csv for backwards compatibility."
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
    fi
    if [ -s "$DATA_DIR/M.csv" ] && [ ! -s "$DATA_DIR/M_orig.csv" ]; then
        log "Found M.csv but not M_orig.csv. Renaming M.csv to M_orig.csv for backwards compatibility."
        mv "$DATA_DIR/M.csv" "$DATA_DIR/M_orig.csv"
    fi
else
    log "Data files not found or empty. Proceeding with data generation/download for $DATASET..."
    if [ "$DATASET" == "dummy" ]; then
        # Generate small synthetic data for testing
        log "Generating synthetic dummy data..."
        echo "10" | python3 -m tecpg data dummy -s 100 -m 1000 -g 1000
        mv data/* "$DATA_DIR/"
        mv annot/* "$ANNOT_DIR/"
        rmdir data annot
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
        mv "$DATA_DIR/M.csv" "$DATA_DIR/M_orig.csv"
    elif [ "$DATASET" == "gtp" ]; then
        log "Downloading GTP data..."
        echo "y" | python3 -m tecpg data gtp --yes
        mv data/* "$DATA_DIR/"
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
        mv "$DATA_DIR/M.csv" "$DATA_DIR/M_orig.csv"
        # For GTP, assuming the demo annots are used
        cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6"
        cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6"
        rmdir data
    elif [ "$DATASET" == "mesa" ]; then
        log "Downloading MESA data..."
        echo "y" | python3 -m tecpg data mesa
        mv data/* "$DATA_DIR/"
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
        mv "$DATA_DIR/M.csv" "$DATA_DIR/M_orig.csv"
        # For MESA, assuming appropriate demo annots are used if available
        # Or fall back to EPIC/HT12 for now
        cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6" 2>/dev/null || true
        cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6" 2>/dev/null || true
        rmdir data
    fi
fi

# Apply EPIC probe blacklist filter
if [ -s "$DATA_DIR/M.csv" ]; then
    log "M.csv already exists. Skipping probe blacklist filtering."
else
    log "Generating EPIC probe blacklist..."
    ./tools/generateEpicProbeBlacklist.sh "$DATA_DIR"

    log "Applying blacklist filter to M_orig.csv..."
    python3 tools/exclude_blacklisted_probes.py "$DATA_DIR/M_orig.csv" "$DATA_DIR/epic_probes_blacklist.csv" "$DATA_DIR/M.csv"
fi

# Stage 1.5: Estimate Immune Cell Proportions
log "[1.5/9] Estimating immune cell proportions using EpiDISH..."
if [ -s "$DATA_DIR/C_post_cellTypes.csv" ]; then
    log "C_post_cellTypes.csv already exists. Skipping cell proportion estimation."
else
    log "Running EpiDISH to estimate cell proportions..."
    ./tools/estimateCellProportions.sh "$DATA_DIR/M.csv" "$DATA_DIR/C_orig.csv" "$DATA_DIR/C_post_cellTypes.csv" "$DATASET"
fi

# Stage 2: Residualization & PCA
log "[2/9] Generating Expression and Methylation PCs..."
if [ -s "$DATA_DIR/C.csv" ]; then
    log "C.csv already exists. Skipping Residualization and PCA generation."
else
    log "Running Expression Residualization & PCA..."
    ./tools/residualize_pca.sh "$DATA_DIR/G.csv" "$DATA_DIR/C_post_cellTypes.csv" "$DATA_DIR/G_PCs.csv" "Exp_PC" --log2-transform

    log "Running Methylation Residualization & PCA..."
    ./tools/residualize_pca.sh "$DATA_DIR/M.csv" "$DATA_DIR/C_post_cellTypes.csv" "$DATA_DIR/M_PCs.csv" "Meth_PC"

    log "Merging Covariates with PCs..."
    python3 -c "
import pandas as pd
C = pd.read_csv('$DATA_DIR/C_post_cellTypes.csv', dtype={0: str})
C.set_index(C.columns[0], inplace=True)
G_PCs = pd.read_csv('$DATA_DIR/G_PCs.csv', dtype={0: str})
G_PCs.set_index(G_PCs.columns[0], inplace=True)
M_PCs = pd.read_csv('$DATA_DIR/M_PCs.csv', dtype={0: str})
M_PCs.set_index(M_PCs.columns[0], inplace=True)
C_final = pd.concat([C, G_PCs, M_PCs], axis=1)
C_final.to_csv('$DATA_DIR/C.csv')
"
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

# Stage 3: Mapping (lstsq + ig)
log "[3/9] Performing eQTM Mapping (lstsq + IG)..."
log "This stage runs the multiple linear regression (mlr) model and computes Integrated Gradients (IG)."
log "Using chunks: M_CHUNK=$M_CHUNK, G_CHUNK=$G_CHUNK. Input: $DATA_DIR, Annotations: $ANNOT_DIR, Output: $OUT_DIR"

# Ensure pipefail is set so pipeline errors (like in mlr) are not masked by tee
set -o pipefail
python3 -m tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq --$MAPPING --meth-loci-per-chunk "$M_CHUNK" --gene-loci-per-chunk "$G_CHUNK" --compute-ig 2>&1 | tee "$OUT_DIR/mlr_run.log"
set +o pipefail

# Extract dynamically evaluated TOTAL_TESTS
EXTRACTED_TOTALS=$(grep -o 'TOTAL_TESTS=[0-9]*' "$OUT_DIR/mlr_run.log" | tail -n 1 | cut -d= -f2 || true)
if [ -n "$EXTRACTED_TOTALS" ]; then
    TOTAL_TESTS=$EXTRACTED_TOTALS
    log "Dynamically extracted TOTAL_TESTS=$TOTAL_TESTS from mlr output."
else
    log "Warning: Could not extract TOTAL_TESTS from mlr output. Falling back to placeholder value ($TOTAL_TESTS)."
fi

# Stage 4: Merge chunked outputs
log "[4/9] Merging chunked outputs to Parquet..."
MERGED_PARQUET="$OUT_DIR/merged.parquet"
log "Converting CSV chunks in $OUT_DIR into a single Parquet file at $MERGED_PARQUET..."
python3 tools/mergeOutputs.py --format parquet "$OUT_DIR" "$MERGED_PARQUET"

# Clean up CSV chunks to save space
log "Cleaning up intermediate CSV chunks..."
rm "$OUT_DIR"/*-*.csv || true

# Stage 5: Annotate regions
log "[5/9] Annotating regions..."
ANNOTATED_PARQUET="$OUT_DIR/annotated.parquet"
log "Mapping eCpG and Gene coordinates to determine regional categories (e.g., CIS, TRANS)."
log "Input Parquet: $MERGED_PARQUET, Output Parquet: $ANNOTATED_PARQUET"
python3 tools/assignRegionToEcpg_parquet.py -d "$MERGED_PARQUET" -g "$ANNOT_DIR/G.bed6" -m "$ANNOT_DIR/M.bed6" -o "$ANNOTATED_PARQUET"

# Stage 6: Precise P-value recalculation
log "[6/9] Recalculating precise p-values..."
RECALC_PARQUET="$OUT_DIR/annotated_pcalc.parquet"
log "Calculating high-precision p-values using DF=$DF."
log "Input Parquet: $ANNOTATED_PARQUET, Output Parquet: $RECALC_PARQUET"
python3 tools/recalculate_pvalues_parquet.py "$ANNOTATED_PARQUET" --df "$DF" --output-file "$RECALC_PARQUET"

# Stage 7: Summarize & FDR
log "[7/9] Calculating FDR and summarizing..."
SUMMARIZED_PARQUET="$OUT_DIR/summarized.parquet"
log "Estimating global Benjamini-Hochberg FDR based on TOTAL_TESTS=$TOTAL_TESTS."
log "Generating diagnostic plots (QQ, Histogram, Saliency)."
log "Input Parquet: $RECALC_PARQUET, Output Parquet: $SUMMARIZED_PARQUET"
python3 tools/summarizeOutput_parquet.py --main-file "$RECALC_PARQUET" --reservoir-file "$OUT_DIR/sample_reservoir.csv" --total-tests "$TOTAL_TESTS" --df "$DF" --calculate-fdr --output-fdr-file "$SUMMARIZED_PARQUET"
# Ensure plots are created in the right folder, but for now they go to CWD based on tool
log "Moving generated plots to $OUT_DIR..."
mv p_value_histogram.png qq_plot.png saliency_profile_top50.png "$OUT_DIR/" 2>/dev/null || true

# Stage 8: Bootstrap List creation
log "[8/9] Creating Bootstrap List..."
BOOTSTRAP_LIST="$OUT_DIR/bootstrap_list.csv"
log "Identifying top hits (ranked by p-value) to be evaluated via bootstrapping."
log "Input Parquet: $SUMMARIZED_PARQUET, Output List: $BOOTSTRAP_LIST"
python3 tools/createBootstrapList.py --input "$SUMMARIZED_PARQUET" --output "$BOOTSTRAP_LIST" --rank-by p-value

# Stage 9: Bootstrap evaluation
log "[9/9] Bootstrapping top hits..."
log "Running bootstrap analysis on the top candidates to validate association robustness."
log "Pairs File: $BOOTSTRAP_LIST, Master Parquet: $SUMMARIZED_PARQUET"
python3 -m tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq_bootstrap --pairs-file "$BOOTSTRAP_LIST" --master-parquet "$SUMMARIZED_PARQUET" --bootstrap-iterations 100 --bootstrap-batch-size 10 --compute-ig

log "======================================"
log "Pipeline completed successfully!"
log "Outputs are in $OUT_DIR/"
log "======================================"
