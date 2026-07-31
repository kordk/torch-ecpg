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
CIS_ENRICH=0
CIS_WINDOW=1000000
TOTAL_TESTS=""
ANNOTATE_MAINLINE=1
N_NULL_PAIRS=""

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
            echo "      --master-parquet PATH      Overrides --cis-enrich. Existing mapping output parquet."
            echo "                                 Also accepts the reservoir directly: a *.csv PATH (e.g."
            echo "                                 sample_reservoir.csv) is converted to reservoir_master.parquet,"
            echo "                                 and a not-yet-built reservoir_master.parquet is built from its"
            echo "                                 sibling sample_reservoir.csv."
            echo "      --reservoir                Overrides --cis-enrich. Convert and use sample_reservoir.csv"
            echo "                                 whose (mt_id, gt_id) universe is scored. Must have been"
            echo "                                 mapped from the SAME data_<ds> (same covariate design);"
            echo "                                 qr_permute fail-closes at runtime if the design mismatches."
            echo "      --cis-enrich               [DEFAULT] Instead of a flat reservoir, build a unified master from"
            echo "                                 analysis: runs a cis write-all map, assembles the near-gene pairs"
            echo "                                 with the reservoir's trans/distal pairs (build_gene_anchored_master.py),"
            echo "                                 and scores the assembled master. Needs output_<ds>/sample_reservoir.csv"
            echo "                                 (from a prior --reservoir-count map). Produces the near-gene coverage"
            echo "                                 the uniform reservoir lacks so the per-region eval can render a verdict."
            echo "      --cis-window N             Cis map half-window in bp applied up/downstream (default: 1000000)."
            echo "                                 Over-capture is intended; assignRegionToEcpg relabels canonically."
            echo "      --total-tests N            BH denominator for fdr_permute. REQUIRED when the mainline"
            echo "                                 annotation stage runs. Must be the MAPPING GRID size -- the"
            echo "                                 same TOTAL_TESTS pipeline.sh uses for fdr_est -- NOT the"
            echo "                                 permute run's tecpg_perm_n_reported. Cross-checked against"
            echo "                                 mlr_run_<ds>.log when that log is present."
            echo "      --n-null-pairs N           Fallback for permutation parquets written before tecpg_perm_n_null_pairs"
            echo "                                 was stamped. Passed directly to eval_permute."
            echo "      --no-annotate-mainline     Skip stage [5/5] (mainline p_permute/fdr_permute annotation)."
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
        --cis-enrich)
            CIS_ENRICH=1
            shift
            ;;
        --cis-window)
            CIS_WINDOW="$2"
            shift 2
            ;;
        --total-tests)
            TOTAL_TESTS="$2"
            shift 2
            ;;
        --n-null-pairs)
            N_NULL_PAIRS="$2"
            shift 2
            ;;
        --no-annotate-mainline)
            ANNOTATE_MAINLINE=0
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
# Exactly one master source: an existing --master-parquet, the --reservoir, or
# --cis-enrich (which BUILDS the master from a cis map + the reservoir).

# Default master source. Applied only when no source was given explicitly,
# so the mutual-exclusion guard below still sees exactly one source and
# stays argument-order-independent. Do NOT move this into the parse cases:
# MODE_COUNT is computed after the parse loop, so clearing CIS_ENRICH there
# makes `--cis-enrich --master-parquet X` and `--master-parquet X
# --cis-enrich` behave differently.
if [ -z "$MASTER_PARQUET" ] && [ $USE_RESERVOIR -eq 0 ] && [ $CIS_ENRICH -eq 0 ]; then
    CIS_ENRICH=1
    CIS_ENRICH_DEFAULTED=1
else
    CIS_ENRICH_DEFAULTED=0
fi

MODE_COUNT=0
[ -n "$MASTER_PARQUET" ] && MODE_COUNT=$((MODE_COUNT + 1))
[ $USE_RESERVOIR -eq 1 ] && MODE_COUNT=$((MODE_COUNT + 1))
[ $CIS_ENRICH -eq 1 ] && MODE_COUNT=$((MODE_COUNT + 1))
if [ $MODE_COUNT -ne 1 ]; then
    log "Error: Exactly one of --master-parquet, --reservoir, or --cis-enrich is required."
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
if [ $CIS_ENRICH -eq 1 ]; then log "Mode: cis-enrich (cis map + assemble; cis-window=+/-${CIS_WINDOW} bp)"; fi
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

# --cis-enrich builds the master internally (cis write-all map -> assemble with the
# reservoir). Point MASTER_PARQUET at the assembled output now so the downstream
# stages reference it; require the reservoir CSV the assemble step combines with.
if [ $CIS_ENRICH -eq 1 ]; then
    ENRICH_RESERVOIR_CSV="${OUT_DIR}/sample_reservoir.csv"
    if [ ! -s "$ENRICH_RESERVOIR_CSV" ]; then
        log "Error: --cis-enrich needs $ENRICH_RESERVOIR_CSV (the reservoir the cis map is assembled with)."
        log "  Produce it with a prior --reservoir-count map (e.g. via pipeline.sh)."
        exit 1
    fi
    MASTER_PARQUET="${OUT_DIR}/gene_anchored_master.parquet"
fi

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

# In --cis-enrich mode the master does not exist yet; the enrichment stage builds it below.
if [ $CIS_ENRICH -eq 0 ] && [ ! -s "$MASTER_PARQUET" ]; then
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

# Stage 0 (--cis-enrich only): cis write-all map -> assemble gene-anchored master.
# Runs only on a fresh run (--start-stage all); --start-stage permute/eval reuses an
# existing gene_anchored_master.parquet. Mirrors pipeline.sh's map->mergeOutputs flow:
# the map auto-chunks, so per-chunk outputs land in a clean subdir and are merged to
# one parquet before assembly.
if [ $CIS_ENRICH -eq 1 ] && [ "$START_STAGE" == "all" ]; then
    CIS_MAP_DIR="$OUT_DIR/cis_map"
    CIS_MAP_PARQUET="$OUT_DIR/cis_map_write_all.parquet"

    log "[cis-map] Cis write-all map (region=cis, +/-${CIS_WINDOW} bp, p-thresh 1.0)..."
    log "          Per-chunk outputs -> $CIS_MAP_DIR, merged -> $CIS_MAP_PARQUET."
    log "          Write-all (-p 1.0) is required: it keeps the mostly-null near-gene"
    log "          bulk the calibration needs. Over-capture is intended; assignRegion"
    log "          relabels canonically downstream."
    rm -rf "$CIS_MAP_DIR"
    mkdir -p "$CIS_MAP_DIR"
    set -o pipefail
    python3 -u -m tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$CIS_MAP_DIR" \
        run mlr --mlr-method qr --cis \
        -w 0 -u "$CIS_WINDOW" -d "$CIS_WINDOW" \
        -p 1.0 --output-format parquet 2>&1 | tee "cis_map_run_${DATASET}.log"
    set +o pipefail

    python3 -u tools/mergeOutputs.py --format parquet --pattern "*.*" \
        "$CIS_MAP_DIR" "$CIS_MAP_PARQUET"
    [ -s "$CIS_MAP_PARQUET" ] || { log "Error: cis map produced no merged output at $CIS_MAP_PARQUET. Check cis_map_run_${DATASET}.log."; exit 1; }

    log "[assemble] Assembling gene-anchored master (cis near-gene + reservoir trans/distal)..."
    python3 -u tools/build_gene_anchored_master.py \
        --cis-map "$CIS_MAP_PARQUET" \
        --reservoir "$ENRICH_RESERVOIR_CSV" \
        --out "$MASTER_PARQUET"
    [ -s "$MASTER_PARQUET" ] || { log "Error: assembly produced no master at $MASTER_PARQUET."; exit 1; }
    log "          Assembled master: $MASTER_PARQUET"
fi

# Stage 1: assign regions
if [ $EXECUTE -eq 1 ]; then
    if [ "$ASSIGN_REGIONS" -eq 1 ]; then
        # Idempotency guard: assignRegionToEcpg_parquet.py appends a 'region' field and
        # CRASHES if one already exists (KeyError: Column region does not exist in schema).
        # Skip re-annotation when the master already carries 'region' (re-runs, or a master
        # that is itself a prior *.region.parquet). Cheap schema read, no full load.
        if python3 -c "import pyarrow.parquet as pq, sys; sys.exit(0 if 'region' in pq.read_schema('$MASTER_PARQUET').names else 1)"; then
            log "[1/5] Master already carries a 'region' column; skipping annotation."
        else
            REGION_MASTER="${MASTER_PARQUET%.parquet}.region.parquet"
            log "[1/5] Assigning canonical regions ($MASTER_PARQUET -> $REGION_MASTER)..."
            python3 -u tools/assignRegionToEcpg_parquet.py \
                -d "$MASTER_PARQUET" \
                -g "$ANNOT_DIR/G.bed6" -m "$ANNOT_DIR/M.bed6" \
                -o "$REGION_MASTER"
            MASTER_PARQUET="$REGION_MASTER"
            log "      Read the 'eCpgs Counts by Region' line above: it is the coverage gate for the per-region eval."
        fi
    else
        log "[1/5] Region annotation skipped (--no-assign-regions); eval falls back to 2-way strata."
    fi
fi

# Stage 2: permute
if [ $EXECUTE -eq 1 ]; then
    log "[2/5] Running permute (consuming master: $MASTER_PARQUET)..."
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
    log "[3/5] Running eval..."
    [ -s "$PERM_OUTPUT" ] || { log "Error: $PERM_OUTPUT missing or empty. Run with --start-stage permute first."; exit 1; }

    if [ -s "$OUT_DIR/eval_permute_report.json" ]; then
        BACKUP_PATH="${OUT_DIR}/eval_permute_report.$(date +%Y%m%d-%H%M%S).json"
        if [ ! -f "$BACKUP_PATH" ]; then
            cp "$OUT_DIR/eval_permute_report.json" "$BACKUP_PATH" || { log "Error: Failed to backup eval report to $BACKUP_PATH"; exit 1; }
            log "Backed up existing eval report to $BACKUP_PATH"
        fi
    fi

    EVAL_ARGS=(
        --perm-output "$PERM_OUTPUT"
        --m-annot "$ANNOT_DIR/M.bed6"
        --g-annot "$ANNOT_DIR/G.bed6"
        --df "$DF"
        --out-dir "$OUT_DIR"
    )

    if [ -n "$N_NULL_PAIRS" ]; then
        EVAL_ARGS+=( --n-null-pairs "$N_NULL_PAIRS" )
    fi

    python3 -u tools/eval_permute.py "${EVAL_ARGS[@]}"

    log "Finished eval_permute. Report at $OUT_DIR/eval_permute_report.json"
fi


if [ $EXECUTE -eq 1 ]; then
    log "[4/5] Running summary..."
    python3 -u tools/summarize_permute.py \
        --perm-output "$PERM_OUTPUT" \
        --report "${OUT_DIR}/eval_permute_report.json" \
        --df "$DF" \
        --m-annot "$ANNOT_DIR/M.bed6" \
        --g-annot "$ANNOT_DIR/G.bed6" \
        --out-dir "$OUT_DIR"
    log "Finished summarize_permute."
fi

# Stage 5: mainline annotation (p_permute + fdr_permute)
#
# Annotates the MAINLINE catalogs -- not the permute output. The permute run's
# eval verdict licenses which strata may carry a permutation-named p; the values
# themselves are the mainline float64 analytic p (precise_mt_p), which is the
# quantity eval_permute.py actually validated.
#
# DENOMINATOR: --total-tests is the mapping grid size (the same TOTAL_TESTS
# pipeline.sh passes at stage 7 of 9), NOT the permute run's tecpg_perm_n_reported.
# n_reported is the permute universe (~1.8e7); the grid is ~1.3e10. Passing
# n_reported would make every fdr_permute ~770x anti-conservative, and
# summarizeOutput's total_tests >= total_rows guard would NOT catch it.
if [ $EXECUTE -eq 1 ]; then
    log "[5/5] Annotating mainline catalogs with p_permute / fdr_permute..."

    EVAL_REPORT="${OUT_DIR}/eval_permute_report.json"
    PERMUTE_PLOTS_DIR="${OUT_DIR}/permute_plots"
    MAINLINE_STATUS=()

    if [ $ANNOTATE_MAINLINE -eq 0 ]; then
        log "      Skipped by request (--no-annotate-mainline)."
    elif [ ! -s "$EVAL_REPORT" ]; then
        log "Error: $EVAL_REPORT missing or empty; the licensing verdict is unavailable."
        log "  p_permute is keyed on the per-region calibration verdict. Run the eval stage first"
        log "  (--start-stage eval), or pass --no-annotate-mainline to skip this stage."
        exit 1
    else
        mkdir -p "$PERMUTE_PLOTS_DIR"

        # Resolve TOTAL_TESTS for fdr_permute
        MLR_LOG="mlr_run_${DATASET}.log"
        TOTAL_TESTS_SOURCE=""
        if [ -n "$TOTAL_TESTS" ]; then
            if [ -f "$MLR_LOG" ]; then
                LOG_TOTAL=$(grep -o 'TOTAL_TESTS=[0-9]*' "$MLR_LOG" | tail -n 1 | cut -d= -f2 || true)
                if [ -n "$LOG_TOTAL" ] && [ "$LOG_TOTAL" != "$TOTAL_TESTS" ]; then
                    log "Error: --total-tests ($TOTAL_TESTS) disagrees with TOTAL_TESTS=$LOG_TOTAL in $MLR_LOG."
                    log "  fdr_permute must share fdr_est's denominator. Refusing to proceed."
                    exit 1
                fi
                TOTAL_TESTS_SOURCE="--total-tests (cross-checked against mlr_run_${DATASET}.log)"
            else
                TOTAL_TESTS_SOURCE="--total-tests (no mapping log present)"
            fi
        else
            if [ -f "$MLR_LOG" ]; then
                LOG_TOTAL=$(grep -o 'TOTAL_TESTS=[0-9]*' "$MLR_LOG" | tail -n 1 | cut -d= -f2 || true)
                if [ -n "$LOG_TOTAL" ]; then
                    TOTAL_TESTS="$LOG_TOTAL"
                    TOTAL_TESTS_SOURCE="derived from mlr_run_${DATASET}.log"
                else
                    log "Error: Could not resolve TOTAL_TESTS. --total-tests was absent, and $MLR_LOG yielded no value."
                    log "  --total-tests may be supplied explicitly."
                    exit 1
                fi
            else
                log "Error: Could not resolve TOTAL_TESTS. --total-tests was absent, and $MLR_LOG is missing."
                log "  --total-tests may be supplied explicitly."
                exit 1
            fi
        fi

        # Extract GENES_EVAL and LOCI_EVAL
        if [ -f "$MLR_LOG" ]; then
            GENES_EVAL=$(grep -o 'Genes evaluated: [0-9]*' "$MLR_LOG" | tail -n 1 | grep -o '[0-9]*' || true)
            LOCI_EVAL=$(grep -o 'Methylation loci evaluated: [0-9]*' "$MLR_LOG" | tail -n 1 | grep -o '[0-9]*' || true)
            LOG_MTIME=$(date -r "$MLR_LOG" || echo "n/a")
        else
            GENES_EVAL=""
            LOCI_EVAL=""
            LOG_MTIME="n/a"
        fi

        if [ -z "$GENES_EVAL" ] || [ -z "$LOCI_EVAL" ]; then
            GRID_DIMS="unavailable"
            log "      (Dimension validation will be skipped; GENES_EVAL or LOCI_EVAL could not be extracted)"
        else
            GRID_DIMS="${GENES_EVAL} genes x ${LOCI_EVAL} loci"
        fi

        log "[5/5] BH denominator resolution:"
        log "        TOTAL_TESTS = $TOTAL_TESTS"
        log "        source      = $TOTAL_TESTS_SOURCE"
        log "        log mtime   = $LOG_MTIME"
        log "        grid dims   = $GRID_DIMS"
        log "        cis default = $CIS_ENRICH_DEFAULTED"

        for TARGET in "${OUT_DIR}/summarized.parquet" "${OUT_DIR}/bootstrap_merged.parquet"; do
            TARGET_NAME="$(basename "$TARGET")"

            if [ ! -s "$TARGET" ]; then
                log "      $TARGET_NAME: SKIPPED -- not found (mainline pipeline.sh has not produced it)."
                MAINLINE_STATUS+=("${TARGET_NAME}=SKIPPED(absent)")
                continue
            fi

            # Idempotency guard, mirroring the [1/5] region guard: both downstream
            # tools refuse to overwrite an existing column and would abort under
            # set -e. Detect and skip instead so a re-run (e.g. --start-stage eval)
            # does not kill the pipeline.
            if python3 -c "import pyarrow.parquet as pq, sys; n=pq.read_schema('$TARGET').names; sys.exit(0 if ('p_permute' in n or 'fdr_permute' in n) else 1)"; then
                log "      $TARGET_NAME: SKIPPED -- already carries p_permute/fdr_permute."
                MAINLINE_STATUS+=("${TARGET_NAME}=SKIPPED(already annotated)")
                continue
            fi

            # Required inputs for this target.
            if ! python3 -c "import pyarrow.parquet as pq, sys; n=pq.read_schema('$TARGET').names; sys.exit(0 if ('region' in n and 'precise_mt_p' in n) else 1)"; then
                log "Error: $TARGET_NAME lacks 'region' and/or 'precise_mt_p'."
                log "  Both are mainline pipeline.sh products (stages 5 and 6 of 9). Refusing to annotate."
                exit 1
            fi

            # Check catalog grid (validation)
            GRID_ARGS=(--catalog "$TARGET")
            if [ -n "$GENES_EVAL" ]; then GRID_ARGS+=(--max-genes "$GENES_EVAL"); fi
            if [ -n "$LOCI_EVAL" ]; then GRID_ARGS+=(--max-loci "$LOCI_EVAL"); fi
            python3 tools/check_catalog_grid.py "${GRID_ARGS[@]}"

            ANNOT_TMP="${TARGET%.parquet}.p_permute.tmp.parquet"
            FINAL_OUT="${TARGET%.parquet}.permute.parquet"
            rm -f "$ANNOT_TMP"

            IN_ROWS=$(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$TARGET').metadata.num_rows)")

            log "      $TARGET_NAME: [A] annotating p_permute from the verdict ($IN_ROWS rows)..."
            python3 -u tools/annotate_permute_p.py \
                --input "$TARGET" \
                --output "$ANNOT_TMP" \
                --eval-report "$EVAL_REPORT" \
                --p-source precise_mt_p \
                --p-column p_permute

            A_ROWS=$(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$ANNOT_TMP').metadata.num_rows)")
            if [ "$A_ROWS" != "$IN_ROWS" ]; then
                log "Error: annotate wrote $A_ROWS rows from $IN_ROWS input rows for $TARGET_NAME."
                rm -f "$ANNOT_TMP"
                exit 1
            fi

            # summarizeOutput_parquet.py REQUIRES --reservoir-file and exits 1 if the
            # path is missing, which under set -e would abort the whole pipeline at its
            # last stage. The reservoir is guaranteed present under --reservoir and
            # --cis-enrich but NOT under a bare --master-parquet run. It feeds only the
            # advisory lambda_GC and QQ plot, neither of which gates anything here, so
            # fall back to an empty reservoir and say so rather than dying.
            PERM_RESERVOIR="${OUT_DIR}/sample_reservoir.csv"
            if [ ! -s "$PERM_RESERVOIR" ]; then
                PERM_RESERVOIR="${PERMUTE_PLOTS_DIR}/empty_reservoir.csv"
                printf 'mt_id,gt_id,mt_t\n' > "$PERM_RESERVOIR"
                log "      No sample_reservoir.csv in $OUT_DIR; using an empty reservoir."
                log "      lambda_GC and the QQ plot are SKIPPED for this pass. Both are"
                log "      advisory and neither affects p_permute or fdr_permute."
            fi

            log "      $TARGET_NAME: [B] BH over p_permute (denominator TOTAL_TESTS=$TOTAL_TESTS)..."
            python3 -u tools/summarizeOutput_parquet.py \
                --main-file "$ANNOT_TMP" \
                --reservoir-file "$PERM_RESERVOIR" \
                --total-tests "$TOTAL_TESTS" \
                --df "$DF" \
                --p-column p_permute \
                --fdr-column fdr_permute \
                --compare-fdr-column fdr_est \
                --calculate-fdr \
                --output-fdr-file "$FINAL_OUT"

            # Truncation guard. summarizeOutput_parquet.py catches exceptions in its
            # output-write loop, prints, and does NOT exit; its finally-block closes
            # the writer, finalizing a VALID, READABLE, SILENTLY TRUNCATED parquet at
            # exit 0. Neither $? nor [ -s ] can detect this. Row count can.
            if [ ! -s "$FINAL_OUT" ]; then
                log "Error: BH stage produced no output at $FINAL_OUT for $TARGET_NAME."
                rm -f "$ANNOT_TMP"
                exit 1
            fi
            F_ROWS=$(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$FINAL_OUT').metadata.num_rows)")
            if [ "$F_ROWS" != "$IN_ROWS" ]; then
                log "Error: TRUNCATED OUTPUT. $FINAL_OUT has $F_ROWS rows; expected $IN_ROWS."
                log "  summarizeOutput_parquet.py exits 0 on a mid-write failure. Check the log above"
                log "  for 'Error writing output FDR file'. The partial output is being removed."
                rm -f "$FINAL_OUT" "$ANNOT_TMP"
                exit 1
            fi

            # summarizeOutput writes its plots to the CWD; move them aside so they do
            # not collide with the mainline QQ/histogram (pipeline.sh:308 moves those
            # into OUT_DIR from the same filenames).
            for png in p_value_histogram.png qq_plot.png saliency_profile_top50.png; do
                [ -f "$png" ] && mv "$png" "${PERMUTE_PLOTS_DIR}/${TARGET_NAME%.parquet}.${png}"
            done

            rm -f "$ANNOT_TMP"
            log "      $TARGET_NAME: annotated -> $FINAL_OUT ($F_ROWS rows)"
            MAINLINE_STATUS+=("${TARGET_NAME}=ANNOTATED(${FINAL_OUT})")
        done
    fi

    log "------------------------------------------------------------"
    log "Mainline annotation summary:"
    if [ ${#MAINLINE_STATUS[@]} -eq 0 ]; then
        log "  (stage did not run)"
    else
        for st in "${MAINLINE_STATUS[@]}"; do log "  $st"; done
    fi
    log "  NOTE: pipelinePost.sh still reads the UN-annotated bootstrap_merged.parquet."
    log "        Repointing it is a separate, deliberate change."
    log "------------------------------------------------------------"
fi
