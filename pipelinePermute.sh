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
START_STAGE="all"
MASTER_PARQUET=""
USE_RESERVOIR=0
ASSIGN_REGIONS=1

PERMUTE_ARGS=()

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: ./pipelinePermute.sh --master-parquet PATH [OPTIONS]"
            echo ""
            echo "qr_permute is a POST-MAPPING CONSUMER: it reads the observed mt_t and the"
            echo "(mt_id, gt_id) universe from an existing mapping output (the master parquet)"
            echo "and scores that universe against the permutation null. This wrapper does NOT"
            echo "run the mapping -- produce the master first (e.g. via pipeline.sh, whose merge"
            echo "stage writes output_<ds>/merged.parquet) and pass it with --master-parquet."
            echo ""
            echo "Options:"
            echo "  -h, --help                     Show this help message and exit"
            echo "      --master-parquet PATH      REQUIRED (if no --reservoir). Existing mapping output parquet."
            echo "                                 Also accepts the reservoir directly: a *.csv PATH (e.g."
            echo "                                 sample_reservoir.csv) is converted to reservoir_master.parquet,"
            echo "                                 and a not-yet-built reservoir_master.parquet is built from its"
            echo "                                 sibling sample_reservoir.csv."
            echo "      --reservoir                REQUIRED (if no --master-parquet). Convert and use sample_reservoir.csv"
            echo "                                 whose (mt_id, gt_id) universe is scored. Must have been"
            echo "                                 mapped from the SAME data_<ds> (same covariate design);"
            echo "                                 qr_permute fail-closes at runtime if the design mismatches."
            echo "  -d, --dataset DATASET          Specify the dataset to use. Options: dummy (default), gtpsub, gtp, mesa"
            echo "  -m, --mapping MAPPING          Region flag passed to qr_permute. Options: all (default)."
            echo "                                 'cis' is accepted by the parser but rejected at runtime:"
            echo "                                 qr_permute's null is trans-global and Phase 2 (cis Beta)"
            echo "                                 is not implemented."
            echo "  -s, --start-stage STAGE        Specify the starting stage. Options: all, permute, eval. Default is 'all'."
            echo "      --no-assign-regions        Skip adding a 'region' column via assignRegionToEcpg_parquet.py"
            echo "      --permutations N           Passthrough to qr_permute."
            echo "      --subsample-mt-count N     Passthrough to qr_permute. NOTE: does NOT shrink output."
            echo "      --subsample-g-count N      Passthrough to qr_permute. NOTE: does NOT shrink output."
            echo "      --seed N                   Passthrough to qr_permute."
            exit 0
            ;;
        -s|--start-stage)
            START_STAGE="$2"
            shift 2
            ;;
        --no-assign-regions)
            ASSIGN_REGIONS=0
            shift
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        --master-parquet)
            MASTER_PARQUET="$2"
            shift 2
            ;;

        --reservoir)
            USE_RESERVOIR=1
            shift
            ;;
        -m|--mapping)
            MAPPING="$2"
            shift 2
            ;;
        --permutations)
            PERMUTE_ARGS+=(--permutations "$2")
            shift 2
            ;;
        --subsample-mt-count)
            PERMUTE_ARGS+=(--subsample-mt-count "$2")
            shift 2
            ;;
        --subsample-g-count)
            PERMUTE_ARGS+=(--subsample-g-count "$2")
            shift 2
            ;;
        --seed)
            PERMUTE_ARGS+=(--seed "$2")
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# qr_permute is a post-mapping consumer: it requires an observed statistic master.
if [ -z "$MASTER_PARQUET" ] && [ $USE_RESERVOIR -eq 0 ]; then
    log "Error: Exactly one of --master-parquet or --reservoir is required."
    exit 1
fi
if [ -n "$MASTER_PARQUET" ] && [ $USE_RESERVOIR -eq 1 ]; then
    log "Error: --master-parquet and --reservoir are mutually exclusive."
    exit 1
fi

if [ "$MAPPING" != "all" ]; then
    log "Error: qr_permute supports --mapping all only."
    log "  --cis filters COVERAGE, but qr_permute's null is trans-global and Phase 2 (cis Beta)"
    log "  is not implemented. A cis-coverage run leaves eval_permute's stratify arm with no"
    log "  trans stratum, which yields a confidently wrong verdict rather than an error."
    exit 1
fi

VALID_STAGES=("all" "permute" "eval")
IS_VALID_STAGE=0
for stage in "${VALID_STAGES[@]}"; do
    if [ "$START_STAGE" == "$stage" ]; then
        IS_VALID_STAGE=1
        break
    fi
done

if [ $IS_VALID_STAGE -eq 0 ]; then
    log "Error: Unknown start stage: $START_STAGE"
    log "Usage: ./pipelinePermute.sh --dataset [dummy|gtp|mesa] --start-stage [STAGE]"
    exit 1
fi

VALID_DATASETS=("dummy" "gtpsub" "gtp" "mesa")
IS_VALID_DATASET=0
for ds in "${VALID_DATASETS[@]}"; do
    if [ "$DATASET" == "$ds" ]; then
        IS_VALID_DATASET=1
        break
    fi
done

if [ $IS_VALID_DATASET -eq 0 ]; then
    log "Error: Unknown dataset: $DATASET"
    log "Usage: ./pipelinePermute.sh --dataset [dummy|gtp|mesa]"
    exit 1
fi

log "============================================================"
log "Starting tecpg Permute Pipeline"
log "Dataset: $DATASET"
log "Master Parquet: $MASTER_PARQUET"
log "Mapping: $MAPPING"
log "Start Stage: $START_STAGE"
log "============================================================"

# Warnings
for arg in "${PERMUTE_ARGS[@]}"; do
    if [[ "$arg" == "--subsample-mt-count" || "$arg" == "--subsample-g-count" ]]; then
        log "NOTE: --subsample-mt-count/--subsample-g-count subsample the NULL population only."
        log "      The reported set is the master parquet's (mt_id, gt_id) universe; these flags"
        log "      do NOT reduce output size. To score a smaller set, narrow the master (map a"
        log "      smaller universe) -- a --pairs-file subset is not exposed by this wrapper."
        log "      Subsample LOCI, never SAMPLES -- dropping samples changes DF."
        break
    fi
done

if [ "$DATASET" == "dummy" ]; then
    log "NOTE: dummy is a WIRING SMOKE TEST ONLY. Disbelieve its numbers."
    log "      Dummy annotations are chrom=randrange(1,23) over random data, so cis and trans"
    log "      are exchangeable BY CONSTRUCTION and the stratify arm will return"
    log "      'single_global_null_adequate' trivially. It says nothing about real data."
fi


OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
ANNOT_DIR="annot_${DATASET}"
mkdir -p "$OUT_DIR" "$DATA_DIR" "$ANNOT_DIR"

if [ $USE_RESERVOIR -eq 1 ]; then
    RESERVOIR_CSV="${OUT_DIR}/sample_reservoir.csv"
    if [ ! -s "$RESERVOIR_CSV" ]; then
        log "Error: --reservoir passed but $RESERVOIR_CSV not found or empty."
        log "  Check that mapping was run with --reservoir-count."
        exit 1
    fi
    log "Converting $RESERVOIR_CSV to parquet..."
    python3 -u tools/reservoir_to_parquet.py --in "$RESERVOIR_CSV" --out "${OUT_DIR}/reservoir_master.parquet"
    MASTER_PARQUET="${OUT_DIR}/reservoir_master.parquet"
fi

# Convenience: accept the reservoir supplied via --master-parquet in either natural
# form, so it doesn't reach pd.read_parquet() as a non-parquet and crash with the
# opaque "Parquet magic bytes not found" error.
#   (a) --master-parquet points at a *.csv (e.g. the reservoir CSV itself): convert
#       it to output_<dir>/reservoir_master.parquet and use that. reservoir_to_parquet
#       validates mt_id/gt_id/mt_t, so a CSV that isn't reservoir-shaped fails loudly
#       with a clear column error instead of a cryptic parquet-reader crash.
#   (b) --master-parquet names a reservoir_master.parquet that isn't built yet but has
#       a sibling sample_reservoir.csv: build it in place. Scoped to that basename on
#       purpose -- a mistyped path to a real mapping output (e.g. merged.parquet) must
#       still fail loudly, not be silently swapped for the reservoir.
if [ $USE_RESERVOIR -eq 0 ] && [ -n "$MASTER_PARQUET" ]; then
    case "$MASTER_PARQUET" in
        *.csv)
            if [ ! -s "$MASTER_PARQUET" ]; then
                log "Error: master CSV not found or empty: $MASTER_PARQUET"
                exit 1
            fi
            CONVERTED="$(dirname "$MASTER_PARQUET")/reservoir_master.parquet"
            log "Master given as CSV: $MASTER_PARQUET"
            log "  Converting to $CONVERTED ..."
            python3 -u tools/reservoir_to_parquet.py --in "$MASTER_PARQUET" --out "$CONVERTED"
            MASTER_PARQUET="$CONVERTED"
            ;;
        *)
            if [ ! -s "$MASTER_PARQUET" ] \
               && [ "$(basename "$MASTER_PARQUET")" = "reservoir_master.parquet" ]; then
                RESERVOIR_CSV="$(dirname "$MASTER_PARQUET")/sample_reservoir.csv"
                if [ -s "$RESERVOIR_CSV" ]; then
                    log "Master parquet not present: $MASTER_PARQUET"
                    log "  Found sibling $RESERVOIR_CSV -- converting it to the reservoir master."
                    python3 -u tools/reservoir_to_parquet.py --in "$RESERVOIR_CSV" --out "$MASTER_PARQUET"
                fi
            fi
            ;;
    esac
fi

if [ ! -s "$MASTER_PARQUET" ]; then
    log "Error: master parquet not found or empty: $MASTER_PARQUET"
    log "  Check the path. It should be an existing mapping output carrying an 'mt_t' column"
    log "  (e.g. output_<ds>/merged.parquet from a prior pipeline.sh run)."
    exit 1
fi


for f in M.csv G.csv C.csv; do
    [ -s "$DATA_DIR/$f" ] || { log "Error: $DATA_DIR/$f missing or empty. Run ./pipelinePre.sh --dataset $DATASET first."; exit 1; }
done

for f in G.bed6 M.bed6; do
    [ -s "$ANNOT_DIR/$f" ] || { log "Error: $ANNOT_DIR/$f missing or empty. Run ./pipelinePre.sh --dataset $DATASET first."; exit 1; }
done

# DF Block Lifted from pipeline.sh
# We calculate DF inside the block that needs it unconditionally since the file exists
# SAMPLES - COVARIATES - 1 (M) - 1 (Intercept)
# Here we just parse the C.csv lines dynamically
SAMPLES=$(wc -l < "$DATA_DIR/C.csv")
SAMPLES=$((SAMPLES - 1)) # Header
COVARS=$(head -n 1 "$DATA_DIR/C.csv" | awk -F, '{print NF-1}')

# M7-DF: stage-boundary check. pipelinePre.sh records the exact (samples,
# covars) shape the PCA merge produced in C.shape.meta. Validate that the
# counts C.csv now carries still match before deriving DF, so a stray
# trailing blank line (shifts SAMPLES) or an extra index column (shifts
# COVARS) fails loudly instead of silently shifting DF and corrupting every
# precise_mt_p / FDR downstream.
SHAPE_META="$DATA_DIR/C.shape.meta"
if [ -f "$SHAPE_META" ]; then
    EXP_SAMPLES=$(grep -o 'samples=[0-9]*' "$SHAPE_META" | cut -d= -f2)
    EXP_COVARS=$(grep -o 'covars=[0-9]*' "$SHAPE_META" | cut -d= -f2)
    if [ -z "$EXP_SAMPLES" ] || [ -z "$EXP_COVARS" ]; then
        log "Error: malformed $SHAPE_META (could not read expected samples/covars from the PCA merge)."
        log "Re-run pipelinePre.sh to regenerate C.csv and its shape metadata."
        exit 1
    fi
    if [ "$SAMPLES" != "$EXP_SAMPLES" ]; then
        log "Error: sample count mismatch in $DATA_DIR/C.csv. Expected $EXP_SAMPLES (from PCA merge) but observed $SAMPLES."
        log "C.csv may have a stray trailing blank line or have been regenerated inconsistently. Refusing to derive DF."
        exit 1
    fi
    if [ "$COVARS" != "$EXP_COVARS" ]; then
        log "Error: covariate count mismatch in $DATA_DIR/C.csv. Expected $EXP_COVARS (from PCA merge) but observed $COVARS."
        log "C.csv may carry an extra index column or have been regenerated inconsistently. Refusing to derive DF."
        exit 1
    fi
else
    log "Error: $SHAPE_META not found. pipelinePre.sh failed to write it."
    log "Cannot cross-check C.csv against the PCA merge. Aborting."
    exit 1
fi

DF=$((SAMPLES - COVARS - 2))
log "Calculated Degrees of Freedom (DF): $DF (SAMPLES=$SAMPLES, COVARS=$COVARS)"

# Assert DF > 0 BEFORE recalculation, not only inside
# recalculate_pvalues_parquet.py, so a non-positive DF aborts the pipeline.
if [ "$DF" -le 0 ]; then
    log "Error: Non-positive degrees of freedom (DF=$DF) from SAMPLES=$SAMPLES, COVARS=$COVARS."
    log "Too few samples relative to covariates; p-value recalculation would be invalid."
    exit 1
fi

EXECUTE=0
if [ "$START_STAGE" == "all" ] || [ "$START_STAGE" == "permute" ]; then EXECUTE=1; fi

PERM_OUTPUT="$OUT_DIR/permutation_results.parquet"

# Stage 1: assign regions
if [ $EXECUTE -eq 1 ]; then
    if [ "$ASSIGN_REGIONS" -eq 1 ]; then
        # Idempotency guard: assignRegionToEcpg_parquet.py appends a 'region' field and
        # CRASHES if one already exists (KeyError: Column region does not exist in schema).
        # Skip re-annotation when the master already carries 'region' (re-runs, or a master
        # that is itself a prior *.region.parquet). Cheap schema read, no full load.
        if python3 -c "import pyarrow.parquet as pq, sys; sys.exit(0 if 'region' in pq.read_schema('$MASTER_PARQUET').names else 1)"; then
            log "[1/4] Master already carries a 'region' column; skipping annotation."
        else
            REGION_MASTER="${MASTER_PARQUET%.parquet}.region.parquet"
            log "[1/4] Assigning canonical regions ($MASTER_PARQUET -> $REGION_MASTER)..."
            python3 -u tools/assignRegionToEcpg_parquet.py \
                -d "$MASTER_PARQUET" \
                -g "$ANNOT_DIR/G.bed6" -m "$ANNOT_DIR/M.bed6" \
                -o "$REGION_MASTER"
            MASTER_PARQUET="$REGION_MASTER"
            log "      Read the 'eCpgs Counts by Region' line above: it is the coverage gate for the per-region eval."
        fi
    else
        log "[1/4] Region annotation skipped (--no-assign-regions); eval falls back to 2-way strata."
    fi
fi

# Stage 2: permute
if [ $EXECUTE -eq 1 ]; then
    log "[2/4] Running permute (consuming master: $MASTER_PARQUET)..."
    log "      qr_permute reads observed mt_t from the master and scores it against the null"
    log "      built from $DATA_DIR (M/G/C). It fail-closes if the master's covariate design"
    log "      does not match this C.csv (DF=$DF)."
    set -o pipefail
    python3 -u -m tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" \
        run mlr --mlr-method qr_permute --all \
        --master-parquet "$MASTER_PARQUET" \
        --output-format auto \
        "${PERMUTE_ARGS[@]}" 2>&1 | tee "permute_run_${DATASET}.log"
    set +o pipefail

    [ -s "$PERM_OUTPUT" ] || { log "Error: $PERM_OUTPUT missing or empty after the permute run."; log "Check permute_run_${DATASET}.log."; exit 1; }
fi

if [ "$START_STAGE" == "eval" ]; then EXECUTE=1; fi

# Stage 3: eval
if [ $EXECUTE -eq 1 ]; then
    log "[3/4] Running eval..."
    [ -s "$PERM_OUTPUT" ] || { log "Error: $PERM_OUTPUT missing or empty. Run with --start-stage permute first."; exit 1; }

    python3 -u tools/eval_permute.py \
        --perm-output "$PERM_OUTPUT" \
        --m-annot "$ANNOT_DIR/M.bed6" \
        --g-annot "$ANNOT_DIR/G.bed6" \
        --df "$DF" \
        --out-dir "$OUT_DIR"

    log "Finished eval_permute. Report at $OUT_DIR/eval_permute_report.json"
fi


if [ $EXECUTE -eq 1 ]; then
    log "[4/4] Running summary..."
    python3 -u tools/summarize_permute.py \
        --perm-output "$PERM_OUTPUT" \
        --report "${OUT_DIR}/eval_permute_report.json" \
        --df "$DF" \
        --m-annot "$ANNOT_DIR/M.bed6" \
        --g-annot "$ANNOT_DIR/G.bed6" \
        --out-dir "$OUT_DIR"
    log "Finished summarize_permute."
fi
