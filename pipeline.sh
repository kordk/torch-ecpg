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
NUM_PCS=5
START_STAGE="all"


# Per-stage Integrated Gradients (IG) Covariate Configuration
# 'none' / '': Compute only scalar methylation attribution (mt_ig)
# 'all': Compute per-feature attribution for all covariates (<covariate>_ig)
# Why two variables? Per-feature IG adds ~12 float columns per row.
# Stage 3 (genome-wide) generates 153M rows. Adding per-feature IG here would bloat
# intermediate files by >5GB each, so it defaults to 'none'.
# Stage 9 (bootstrap) runs on the top 20k candidates. Adding per-feature IG here
# costs only ~1MB, but enables full saliency fraction analysis, so it defaults to 'all'.
MLR_IG_COVARIATES="none"
BOOTSTRAP_IG_COVARIATES="all"

# Chunk sizes for `tecpg run mlr` are intentionally NOT set here. As of
# tecpg 1.21.0-dev the CLI's anchored auto-sizer (`_auto_chunk_sizes` in
# tecpg/cli.py) picks `--gene-loci-per-chunk` and `--meth-loci-per-chunk`
# from the live RAM/GPU budget on server-class hosts when the user
# supplies neither flag, and honors anchored mode when exactly one is
# supplied. Combined with the post-PR1 inner-kernel memory reductions
# (commits 24e96a6, b0f3a15) and the dropped 40k methylation ceiling
# (commit 0a35dbb), the auto-sizer chooses chunk sizes that are tuned to
# the actual host, so we no longer hardcode stale per-dataset values.
# Override by exporting TECPG_M_CHUNK / TECPG_G_CHUNK before invoking
# this script if you need to pin a specific chunking.

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: ./pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --help               Show this help message and exit"
            echo "  -d, --dataset DATASET    Specify the dataset to use. Options: dummy (default), gtp, mesa"
            echo "  -m, --mapping MAPPING    Specify the mapping method for tecpg. Options: all (default), cis"
            echo "  -s, --start-stage STAGE  Specify the starting stage. Options: all, map, merge, annotate, precise_p, summarize, boot_list, bootstrap. Default is 'all'."
            exit 0
            ;;
        -s|--start-stage)
            START_STAGE="$2"
            shift 2
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

if [ "$MAPPING" != "all" ] && [ "$MAPPING" != "cis" ]; then
    log "Error: Unknown mapping: $MAPPING"
    log "Usage: ./pipeline.sh --dataset [dummy|gtp|mesa] --mapping [all|cis] --start-stage [STAGE]"
    exit 1
fi

VALID_STAGES=("all" "map" "merge" "annotate" "precise_p" "summarize" "boot_list" "bootstrap")
IS_VALID_STAGE=0
for stage in "${VALID_STAGES[@]}"; do
    if [ "$START_STAGE" == "$stage" ]; then
        IS_VALID_STAGE=1
        break
    fi
done

if [ $IS_VALID_STAGE -eq 0 ]; then
    log "Error: Unknown start stage: $START_STAGE"
    log "Valid options for --start-stage: ${VALID_STAGES[*]}"
    exit 1
fi

if [ "$DATASET" == "gtp" ]; then
    TOTAL_TESTS=13744315260 # Placeholder for full GTP size, will be dynamically updated
elif [ "$DATASET" == "mesa" ]; then
    TOTAL_TESTS=10000000000 # Placeholder for MESA, will be dynamically updated
elif [ "$DATASET" == "dummy" ]; then
    TOTAL_TESTS=1000000 # Placeholder for 1000 M * 1000 G, will be dynamically updated
else
    log "Error: Unknown dataset: $DATASET"
    log "Usage: ./pipeline.sh --dataset [dummy|gtp|mesa] --mapping [all|cis]"
    exit 1
fi

# Optional pinning of chunk sizes via env vars; otherwise the tecpg CLI
# auto-sizer picks both from the host budget.
MLR_CHUNK_ARGS=()
if [ -n "${TECPG_M_CHUNK:-}" ]; then
    MLR_CHUNK_ARGS+=(--meth-loci-per-chunk "$TECPG_M_CHUNK")
fi
if [ -n "${TECPG_G_CHUNK:-}" ]; then
    MLR_CHUNK_ARGS+=(--gene-loci-per-chunk "$TECPG_G_CHUNK")
fi

# Determine whether to apply logit transformation based on dataset
#if [ "$DATASET" = "gtp" ]; then
#    MLR_CHUNK_ARGS+=(--logit-transform)
#fi

log "======================================"
log "Starting eQTM Pipeline for: $DATASET (Mapping: $MAPPING)"
if [ "${#MLR_CHUNK_ARGS[@]}" -gt 0 ]; then
    log "Dataset configurations: TOTAL_TESTS=$TOTAL_TESTS, chunk overrides=${MLR_CHUNK_ARGS[*]}"
else
    log "Dataset configurations: TOTAL_TESTS=$TOTAL_TESTS, chunk sizes=auto (tecpg CLI)"
fi
log "Starting from stage: $START_STAGE"
log "======================================"

# Setup directories
OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
ANNOT_DIR="annot_${DATASET}"
mkdir -p "$OUT_DIR" "$DATA_DIR" "$ANNOT_DIR"

for f in M.csv G.csv C.csv; do
    [ -s "$DATA_DIR/$f" ] || { log "Error: $DATA_DIR/$f missing or empty. Run ./pipelinePre.sh --dataset $DATASET first."; exit 1; }
done

for f in G.bed6 M.bed6; do
    [ -s "$ANNOT_DIR/$f" ] || { log "Error: $ANNOT_DIR/$f missing or empty. Run ./pipelinePre.sh --dataset $DATASET first."; exit 1; }
done

EXECUTE=0
if [ "$START_STAGE" == "all" ] || [ "$START_STAGE" == "map" ]; then EXECUTE=1; fi

# We calculate DF inside the block that needs it (or we can just calculate it here unconditionally if the file exists)
# SAMPLES - COVARIATES - 1 (M) - 1 (Intercept)
# Using placeholder calculation if needed; assuming df=96 for dummy (100 - 2 - 2)
# Here we just parse the C.csv lines dynamically
if [ -s "$DATA_DIR/C.csv" ]; then
    SAMPLES=$(wc -l < "$DATA_DIR/C.csv")
    SAMPLES=$((SAMPLES - 1)) # Header
    COVARS=$(head -n 1 "$DATA_DIR/C.csv" | awk -F, '{print NF-1}')
    DF=$((SAMPLES - COVARS - 2))
    log "Calculated Degrees of Freedom (DF): $DF (SAMPLES=$SAMPLES, COVARS=$COVARS)"
else
    # Default placeholder
    DF=96
fi

# Stage 3: Mapping (lstsq + ig)
if [ $EXECUTE -eq 1 ]; then
log "[3/9] Performing eQTM Mapping (lstsq + IG)..."
log "This stage runs the multiple linear regression (mlr) model and computes Integrated Gradients (IG)."
if [ "${#MLR_CHUNK_ARGS[@]}" -gt 0 ]; then
    log "Chunk overrides: ${MLR_CHUNK_ARGS[*]}. Input: $DATA_DIR, Annotations: $ANNOT_DIR, Output: $OUT_DIR"
else
    log "Chunk sizes auto-selected by tecpg CLI from host budget. Input: $DATA_DIR, Annotations: $ANNOT_DIR, Output: $OUT_DIR"
fi

# Ensure pipefail is set so pipeline errors (like in mlr) are not masked by tee
set -o pipefail
MLR_IG_ARGS=()
if [ "$MLR_IG_COVARIATES" = "all" ]; then
    MLR_IG_ARGS+=(--ig-covariates)
elif [ -n "$MLR_IG_COVARIATES" ] && [ "$MLR_IG_COVARIATES" != "none" ]; then
    MLR_IG_ARGS+=(--ig-covariates-list "$MLR_IG_COVARIATES")
fi
python3 -m tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq --$MAPPING "${MLR_CHUNK_ARGS[@]}" --compute-ig "${MLR_IG_ARGS[@]}" 2>&1 | tee "mlr_run_${DATASET}.log"
set +o pipefail
fi

if [ "$START_STAGE" == "merge" ]; then EXECUTE=1; fi

# Extract dynamically evaluated TOTAL_TESTS (needed if we run mapping or summarize, independent of whether Stage 3 ran just now)
if [ -f "mlr_run_${DATASET}.log" ]; then
    EXTRACTED_TOTALS=$(grep -o 'TOTAL_TESTS=[0-9]*' "mlr_run_${DATASET}.log" | tail -n 1 | cut -d= -f2 || true)
    if [ -n "$EXTRACTED_TOTALS" ]; then
        TOTAL_TESTS=$EXTRACTED_TOTALS
        log "Dynamically extracted TOTAL_TESTS=$TOTAL_TESTS from mlr output."
    else
        log "Warning: Could not extract TOTAL_TESTS from mlr output. Falling back to placeholder value ($TOTAL_TESTS)."
    fi
else
    log "Warning: mlr_run_${DATASET}.log not found. Falling back to placeholder value ($TOTAL_TESTS)."
fi

# Stage 4: Merge chunked outputs
if [ $EXECUTE -eq 1 ]; then
log "[4/9] Merging chunked outputs to Parquet..."
MERGED_PARQUET="$OUT_DIR/merged.parquet"
log "Converting CSV chunks in $OUT_DIR into a single Parquet file at $MERGED_PARQUET..."
python3 tools/mergeOutputs.py --format parquet --pattern "*.*" "$OUT_DIR" "$MERGED_PARQUET"

# Clean up CSV chunks to save space
log "Cleaning up intermediate CSV chunks..."
rm "$OUT_DIR"/*-*.csv "$OUT_DIR"/*-*.parquet 2>/dev/null || true
fi

if [ "$START_STAGE" == "annotate" ]; then EXECUTE=1; fi

# Variables that span across conditionally-executed later stages
MERGED_PARQUET="$OUT_DIR/merged.parquet"
ANNOTATED_PARQUET="$OUT_DIR/annotated.parquet"
RECALC_PARQUET="$OUT_DIR/annotated_pcalc.parquet"
SUMMARIZED_PARQUET="$OUT_DIR/summarized.parquet"
BOOTSTRAP_LIST="$OUT_DIR/bootstrap_list.csv"

# Stage 5: Annotate regions
if [ $EXECUTE -eq 1 ]; then
log "[5/9] Annotating regions..."
log "Mapping eCpG and Gene coordinates to determine regional categories (e.g., CIS, TRANS)."
log "Input Parquet: $MERGED_PARQUET, Output Parquet: $ANNOTATED_PARQUET"
python3 tools/assignRegionToEcpg_parquet.py -d "$MERGED_PARQUET" -g "$ANNOT_DIR/G.bed6" -m "$ANNOT_DIR/M.bed6" -o "$ANNOTATED_PARQUET"
fi

if [ "$START_STAGE" == "precise_p" ]; then EXECUTE=1; fi

# Stage 6: Precise P-value recalculation
if [ $EXECUTE -eq 1 ]; then
log "[6/9] Recalculating precise p-values..."
log "Calculating high-precision p-values using DF=$DF."
log "Input Parquet: $ANNOTATED_PARQUET, Output Parquet: $RECALC_PARQUET"
python3 tools/recalculate_pvalues_parquet.py "$ANNOTATED_PARQUET" --df "$DF" --output-file "$RECALC_PARQUET"
fi

if [ "$START_STAGE" == "summarize" ]; then EXECUTE=1; fi

# Stage 7: Summarize & FDR
if [ $EXECUTE -eq 1 ]; then
log "[7/9] Calculating FDR and summarizing..."
log "Estimating global Benjamini-Hochberg FDR based on TOTAL_TESTS=$TOTAL_TESTS."
log "Generating diagnostic plots (QQ, Histogram, Saliency)."
log "Input Parquet: $RECALC_PARQUET, Output Parquet: $SUMMARIZED_PARQUET"
python3 tools/summarizeOutput_parquet.py --main-file "$RECALC_PARQUET" --reservoir-file "$OUT_DIR/sample_reservoir.csv" --total-tests "$TOTAL_TESTS" --df "$DF" --calculate-fdr --output-fdr-file "$SUMMARIZED_PARQUET"
# Ensure plots are created in the right folder, but for now they go to CWD based on tool
log "Moving generated plots to $OUT_DIR..."
mv p_value_histogram.png qq_plot.png saliency_profile_top50.png "$OUT_DIR/" 2>/dev/null || true
fi

if [ "$START_STAGE" == "boot_list" ]; then EXECUTE=1; fi

# Stage 8: Bootstrap List creation
if [ $EXECUTE -eq 1 ]; then
log "[8/9] Creating Bootstrap List..."
log "Identifying top hits (ranked by p-value) to be evaluated via bootstrapping."
log "Input Parquet: $SUMMARIZED_PARQUET, Output List: $BOOTSTRAP_LIST"
python3 tools/createBootstrapList.py --input "$SUMMARIZED_PARQUET" --output "$BOOTSTRAP_LIST" --rank-by p-value --min-per-region 4500 --max-per-region 10000
fi

if [ "$START_STAGE" == "bootstrap" ]; then EXECUTE=1; fi

# Stage 9: Bootstrap evaluation
if [ $EXECUTE -eq 1 ]; then
log "[9/9] Bootstrapping top hits..."
log "Running bootstrap analysis on the top candidates to validate association robustness."
log "Pairs File: $BOOTSTRAP_LIST, Master Parquet: $SUMMARIZED_PARQUET"
BOOTSTRAP_IG_ARGS=()
if [ "$BOOTSTRAP_IG_COVARIATES" = "all" ]; then
    BOOTSTRAP_IG_ARGS+=(--ig-covariates)
elif [ -n "$BOOTSTRAP_IG_COVARIATES" ] && [ "$BOOTSTRAP_IG_COVARIATES" != "none" ]; then
    BOOTSTRAP_IG_ARGS+=(--ig-covariates-list "$BOOTSTRAP_IG_COVARIATES")
fi
python3 -m tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq_bootstrap --pairs-file "$BOOTSTRAP_LIST" --master-parquet "$SUMMARIZED_PARQUET" --bootstrap-iterations 1000 --bootstrap-batch-size 10 --compute-ig "${BOOTSTRAP_IG_ARGS[@]}"
fi

log "======================================"
log "Pipeline completed successfully!"
log "Outputs are in $OUT_DIR/"
log "======================================"
