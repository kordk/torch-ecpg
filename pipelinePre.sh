#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# Log function for timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Move a staging directory's contents into place, then remove it.
# 'mv dir/*' skips dotfiles, and a bare 'rmdir' aborts the whole run under
# 'set -e' -- letting a cleanup step destroy a completed download. Move
# dotfiles too, and never let the cleanup itself be fatal.
drain_staging_dir() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    shopt -s dotglob nullglob
    local entries=("$src"/*)
    shopt -u dotglob nullglob
    if [ ${#entries[@]} -gt 0 ]; then
        mv "${entries[@]}" "$dst/"
    fi
    if ! rmdir "$src" 2>/dev/null; then
        log "Warning: '$src' not empty after moving its contents; leaving it in place."
        ls -A "$src" | sed 's/^/           leftover: /'
    fi
}

# Default settings
DATASET="dummy"

# gtpsub-only: locus subsample targets (rows kept from full GTP M/G; samples untouched)
GTPSUB_M_LOCI=10000
GTPSUB_G_LOCI=5000
GTPSUB_SEED=42
START_STAGE="all"

# Methylation array the probe blacklist is scoped to: 450k, epic, or both.
# GTP and MESA are both HumanMethylation450.
METH_ARRAY="450k"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: ./pipelinePre.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --help               Show this help message and exit"
            echo "  -d, --dataset DATASET    Specify the dataset to use. Options: dummy (default), gtp, gtpsub, mesa"
            echo "  -s, --start-stage STAGE  Specify the starting stage. Options: all, prep, ancestry, cell_prop, pca. Default is 'all'."
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
        *)
            echo "Unknown parameter passed: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

VALID_STAGES=("all" "prep" "ancestry" "cell_prop" "pca")
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

if [ "$DATASET" != "gtp" ] && [ "$DATASET" != "gtpsub" ] && [ "$DATASET" != "mesa" ] && [ "$DATASET" != "dummy" ]; then
    log "Error: Unknown dataset: $DATASET"
    log "Usage: ./pipelinePre.sh --dataset [dummy|gtp|gtpsub|mesa]"
    exit 1
fi

log "======================================"
log "Starting Preprocessing Pipeline for: $DATASET"
log "Starting from stage: $START_STAGE"
log "======================================"

# Setup directories
OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
ANNOT_DIR="annot_${DATASET}"
mkdir -p "$OUT_DIR" "$DATA_DIR" "$ANNOT_DIR"

# Staging directories for `tecpg data <ds>`, which writes to --input-dir /
# --annot-dir and CLEARS them first (helper.initialize_dir rmtree's the target).
# These must be per-dataset: the tecpg default is a shared ./data, so a GTP and
# a MESA run in the same working directory overwrite each other's staging area
# and one run's cleanup deletes or trips over the other's files. They are
# scratch -- never point these at DATA_DIR, which initialize_dir would wipe.
STAGE_DIR="data_stage_${DATASET}"
STAGE_ANNOT_DIR="annot_stage_${DATASET}"

# Covariate file threaded through preprocessing. Each stage that modifies
# covariates writes a NEW file rather than overwriting its input, and this
# resolver picks the most advanced one present. Resolving by file existence
# (instead of assigning inside the stage blocks) keeps --start-stage correct:
# starting at 'pca' still picks up covariates produced by earlier stages.
resolve_cov_file() {
    COV_FILE="$DATA_DIR/C_orig.csv"
    if [ -s "$DATA_DIR/C_orig.anc.csv" ]; then
        COV_FILE="$DATA_DIR/C_orig.anc.csv"
    fi
    if [ -s "$DATA_DIR/C_post_cellTypes.csv" ]; then
        COV_FILE="$DATA_DIR/C_post_cellTypes.csv"
    fi
    if [ -s "$DATA_DIR/C_post_cellTypes.encoded.csv" ]; then
        COV_FILE="$DATA_DIR/C_post_cellTypes.encoded.csv"
    fi
}

# Per-dataset covariate configuration. These are dataset judgements, not
# framework rules, so they live here rather than in the tools.
#   ANCESTRY_MERGE_COLUMNS: which ancestry components become covariates.
#     MESA: the 65 rs control probes are the only clean genotype instrument in
#     this matrix (mean_clusters 3.0); the data-driven gap probes came back
#     predominantly bimodal and are not used. Two components clear the 65-probe
#     noise floor.
#   ENCODE_COLUMNS / ENCODE_MIN_CELL_SIZE: integer-coded categoricals that need
#     indicator encoding, and the smallest level retained.
ANCESTRY_MERGE_COLUMNS=""
ANCESTRY_MERGE_NAMES=""
ENCODE_COLUMNS=""
ENCODE_MIN_CELL_SIZE=3
if [ "$DATASET" == "mesa" ]; then
    ANCESTRY_MERGE_COLUMNS="rs_PC1,rs_PC2"
    ANCESTRY_MERGE_NAMES="Anc_PC1,Anc_PC2"
    ENCODE_COLUMNS="racegendersite"
fi

EXECUTE=0
if [ "$START_STAGE" == "all" ] || [ "$START_STAGE" == "prep" ]; then EXECUTE=1; fi

# Stage 1: Data Preparation
if [ $EXECUTE -eq 1 ]; then
log "[1/9] Preparing data..."
log "Checking if dataset files already exist in $DATA_DIR..."

if ( [ -s "$DATA_DIR/M_orig.csv" ] || [ -s "$DATA_DIR/M.csv" ] ) && [ -s "$DATA_DIR/G.csv" ] && ( [ -s "$DATA_DIR/C_orig.csv" ] || [ -s "$DATA_DIR/C.csv" ] ); then
    log "Data files (M_orig.csv/M.csv, G.csv, and C_orig.csv/C.csv) already exist and are not empty. Skipping download/generation."
    if [ -s "$DATA_DIR/C.csv" ] && [ ! -s "$DATA_DIR/C_orig.csv" ]; then
        log "Found C.csv but not C_orig.csv. Renaming C.csv to C_orig.csv for backwards compatibility."
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
    fi
else
    log "Data files not found or empty. Proceeding with data generation/download for $DATASET..."
    # A marker from a previous build would suppress the blacklist filter on the
    # freshly downloaded M.csv.
    rm -f "$DATA_DIR/M.csv.blacklist.meta"
    if [ "$DATASET" == "dummy" ]; then
        # Generate small synthetic data for testing
        log "Generating synthetic dummy data..."
        echo "10" | python3 -m tecpg -i "$STAGE_DIR" -a "$STAGE_ANNOT_DIR" data dummy -s 100 -m 20 -g 20
        drain_staging_dir "$STAGE_DIR" "$DATA_DIR"
        drain_staging_dir "$STAGE_ANNOT_DIR" "$ANNOT_DIR"
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
    elif [ "$DATASET" == "gtp" ] || [ "$DATASET" == "gtpsub" ]; then
        log "Downloading GTP data..."
        echo "y" | python3 -m tecpg -i "$STAGE_DIR" data gtp --yes
        drain_staging_dir "$STAGE_DIR" "$DATA_DIR"

        if [ "$DATASET" == "gtpsub" ]; then
            log "Subsampling gtpsub loci..."
            python3 tools/subsample_loci.py "$DATA_DIR/M.csv" "$DATA_DIR/M.csv" "$GTPSUB_M_LOCI" --seed "$GTPSUB_SEED"
            python3 tools/subsample_loci.py "$DATA_DIR/G.csv" "$DATA_DIR/G.csv" "$GTPSUB_G_LOCI" --seed "$GTPSUB_SEED"
            python3 tools/subsample_loci.py "$DATA_DIR/M_orig.csv" "$DATA_DIR/M_orig.csv" "$GTPSUB_M_LOCI" --seed "$GTPSUB_SEED"
            python3 tools/subsample_loci.py "$DATA_DIR/G_orig.csv" "$DATA_DIR/G_orig.csv" "$GTPSUB_G_LOCI" --seed "$GTPSUB_SEED"
        fi
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
        # For GTP, assuming the demo annots are used
        if [ -f "demo/annoEPIC_comprehensive.hg19.bed6" ]; then
            cp demo/annoEPIC_comprehensive.hg19.bed6 "$ANNOT_DIR/M.bed6"
        else
            cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6"
        fi

        if [ -f "demo/annoHT12_comprehensive.hg19.bed6" ]; then
            cp demo/annoHT12_comprehensive.hg19.bed6 "$ANNOT_DIR/G.bed6"
        else
            cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6"
        fi
    elif [ "$DATASET" == "mesa" ]; then
        log "Downloading MESA data..."
        echo "y" | python3 -m tecpg -i "$STAGE_DIR" data mesa
        drain_staging_dir "$STAGE_DIR" "$DATA_DIR"
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
        # For MESA, assuming appropriate demo annots are used if available
        # Or fall back to EPIC/HT12 for now
        if [ -f "demo/annoEPIC_comprehensive.hg19.bed6" ]; then
            cp demo/annoEPIC_comprehensive.hg19.bed6 "$ANNOT_DIR/M.bed6" 2>/dev/null || true
        else
            cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6" 2>/dev/null || true
        fi

        if [ -f "demo/annoHT12_comprehensive.hg19.bed6" ]; then
            cp demo/annoHT12_comprehensive.hg19.bed6 "$ANNOT_DIR/G.bed6" 2>/dev/null || true
        else
            cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6" 2>/dev/null || true
        fi
    fi
fi

# Apply probe blacklist filter (array scope: METH_ARRAY, set in Default settings)
#
# The guard is a stage-owned marker, not the existence of M.csv. Stage 1 writes
# M.csv itself (tecpg data <ds> saves M.csv alongside M_orig.csv), so guarding
# on M.csv meant the filter never executed on a fresh build: the guard was
# satisfied by the very file this stage is supposed to produce. The marker is
# written only on successful completion of the filter.
#
# The filter reads M.csv, not M_orig.csv. M_orig is the pre-dropna matrix
# (tecpg/gtp.py takes M_orig = M.copy() before the missing-data drop), so
# filtering M_orig into M.csv would silently reintroduce every locus stage 1
# dropped for missing data -- 104,132 of them for GTP. Reading M.csv composes
# the two filters instead of letting one overwrite the other.
BLACKLIST_MARKER="$DATA_DIR/M.csv.blacklist.meta"
if [ -s "$BLACKLIST_MARKER" ]; then
    log "Blacklist filter already applied (see $BLACKLIST_MARKER). Skipping."
elif [ ! -s "$DATA_DIR/M.csv" ]; then
    log "Error: $DATA_DIR/M.csv not found; cannot apply blacklist filter."
    exit 1
else
    log "Generating probe blacklist (array: $METH_ARRAY)..."
    ./tools/generateProbeBlacklist.sh "$DATA_DIR" "$METH_ARRAY"

    log "Applying blacklist filter to M.csv..."
    BLACKLIST_TMP="$DATA_DIR/M.csv.blacklist.tmp"
    rm -f "$BLACKLIST_TMP"
    python3 tools/exclude_blacklisted_probes.py "$DATA_DIR/M.csv" "$DATA_DIR/probes_blacklist.csv" "$BLACKLIST_TMP"

    # Move into place only after the filter succeeds, so an interrupted run
    # leaves M.csv intact rather than truncated.
    mv "$BLACKLIST_TMP" "$DATA_DIR/M.csv"

    {
        echo "applied=$(date +'%Y-%m-%dT%H:%M:%S')"
        echo "array=$METH_ARRAY"
        echo "blacklist_rows=$(( $(wc -l < "$DATA_DIR/probes_blacklist.csv") - 1 ))"
        echo "m_rows_after=$(( $(wc -l < "$DATA_DIR/M.csv") - 1 ))"
    } > "$BLACKLIST_MARKER"
    log "Blacklist filter applied; wrote $BLACKLIST_MARKER"
fi
fi

if [ "$START_STAGE" == "ancestry" ]; then EXECUTE=1; fi

# Stage 1.4: Evaluate methylation-derived ancestry instruments
#
# Characterisation only at this stage: the report is written for every real
# cohort, and whether any component is admitted as a covariate is a separate,
# per-cohort decision made downstream. Reads M_orig.csv deliberately -- the
# genotype-like probes this needs (rs control probes, SNP-affected CpGs) are
# the ones the blacklist stage removes, so it wants the pre-filter matrix.
#
# The guard checks this stage's OWN output. That distinction matters: guarding
# on a file an earlier stage also writes is what let the blacklist filter be
# skipped on a fresh build.
if [ $EXECUTE -eq 1 ]; then
log "[1.4/9] Evaluating methylation-derived ancestry instruments..."
if [ "$DATASET" == "dummy" ] || [ "$DATASET" == "gtpsub" ]; then
    log "Skipping ancestry evaluation for $DATASET (no population structure to recover)."
elif [ -s "$OUT_DIR/ancestry_probes.json" ]; then
    log "ancestry_probes.json already exists. Skipping ancestry evaluation."
else
    # Grouping used only for the separation module, never as an input to the
    # instruments themselves. MESA carries the composite; GTP drops
    # race/ethnicity upstream, so Sex is the only categorical available.
    ANCESTRY_GROUP=""
    if [ "$DATASET" == "mesa" ]; then
        ANCESTRY_GROUP="racegendersite"
    elif [ "$DATASET" == "gtp" ]; then
        ANCESTRY_GROUP="Sex"
    fi

    ANCESTRY_ARGS=()
    if [ -n "$ANCESTRY_GROUP" ]; then
        ANCESTRY_ARGS+=(--group-column "$ANCESTRY_GROUP")
    fi
    if [ -s "$DATA_DIR/probes_blacklist.csv" ]; then
        ANCESTRY_ARGS+=(--blacklist "$DATA_DIR/probes_blacklist.csv")
    else
        log "No probes_blacklist.csv found; ancestry method C will be skipped."
    fi

    log "Running ancestry instrument evaluation (grouping: ${ANCESTRY_GROUP:-none})..."
    python3 tools/ancestry_probes_report.py \
        --dataset "$DATASET" \
        --methylation "$DATA_DIR/M_orig.csv" \
        --covariates "$DATA_DIR/C_orig.csv" \
        "${ANCESTRY_ARGS[@]}" \
        --out "$OUT_DIR/ancestry_probes_report.html" \
        --json "$OUT_DIR/ancestry_probes.json" \
        --scores-out "$DATA_DIR/ancestry_scores.csv" \
        --probes-out "$OUT_DIR/ancestry_probes.tsv"
fi
fi

# Stage 1.45: Merge selected ancestry components into the covariates
#
# Runs before cell-type estimation so the ancestry columns are present in the
# covariate file that EpiDISH merges onto (estimateCellProportions.R merges by
# row names and passes unrecognised columns through). Characterisation and use
# are kept separate: stage 1.4 evaluates every instrument for both cohorts,
# this stage admits components as covariates only where configured.
if [ $EXECUTE -eq 1 ] && [ -n "$ANCESTRY_MERGE_COLUMNS" ]; then
log "[1.45/9] Merging ancestry components into covariates..."
if [ -s "$DATA_DIR/C_orig.anc.csv" ]; then
    log "C_orig.anc.csv already exists. Skipping ancestry merge."
elif [ ! -s "$DATA_DIR/ancestry_scores.csv" ]; then
    log "Error: $DATA_DIR/ancestry_scores.csv not found; run the ancestry stage first."
    exit 1
else
    log "Merging $ANCESTRY_MERGE_COLUMNS as $ANCESTRY_MERGE_NAMES..."
    python3 tools/mergeCovariateColumns.py \
        --covariates "$DATA_DIR/C_orig.csv" \
        --sidecar "$DATA_DIR/ancestry_scores.csv" \
        --columns "$ANCESTRY_MERGE_COLUMNS" \
        --rename "$ANCESTRY_MERGE_NAMES" \
        --output "$DATA_DIR/C_orig.anc.csv" \
        --report "$DATA_DIR/ancestry_merge.json"
fi
fi

if [ "$START_STAGE" == "cell_prop" ]; then EXECUTE=1; fi

# Stage 1.5: Estimate Immune Cell Proportions
if [ $EXECUTE -eq 1 ]; then
log "[1.5/9] Estimating immune cell proportions using EpiDISH..."
resolve_cov_file
log "Using covariate file: $COV_FILE"
if [ -s "$DATA_DIR/C_post_cellTypes.csv" ]; then
    log "C_post_cellTypes.csv already exists. Skipping cell proportion estimation."
else
    log "Running EpiDISH to estimate cell proportions..."
    if [ "$DATASET" == "dummy" ] || [ "$DATASET" == "gtpsub" ]; then
        log "Skipping EpiDISH for dummy data (random noise causes singular fits)."
        cp "$COV_FILE" "$DATA_DIR/C_post_cellTypes.csv"
    else
        ./tools/estimateCellProportions.sh "$DATA_DIR/M.csv" "$COV_FILE" "$DATA_DIR/C_post_cellTypes.csv" "$DATASET"
    fi
fi
fi

# Stage 1.6: Encode integer-coded categorical covariates
#
# Runs after cell-type estimation and before residualization, so the PCs are
# computed on M and G residualized against indicator columns rather than
# against a single slope along an arbitrary code ordering. Placing it after the
# PC stage would leave the categorical structure in the residuals for PCA to
# rediscover, which is the behaviour this stage exists to remove.
if [ $EXECUTE -eq 1 ] && [ -n "$ENCODE_COLUMNS" ]; then
log "[1.6/9] Encoding categorical covariates ($ENCODE_COLUMNS)..."
if [ -s "$DATA_DIR/C_post_cellTypes.encoded.csv" ]; then
    log "C_post_cellTypes.encoded.csv already exists. Skipping categorical encoding."
elif [ ! -s "$DATA_DIR/C_post_cellTypes.csv" ]; then
    log "Error: $DATA_DIR/C_post_cellTypes.csv not found; run the cell_prop stage first."
    exit 1
else
    ENCODE_ARGS=()
    IFS=',' read -ra _enc_cols <<< "$ENCODE_COLUMNS"
    for _c in "${_enc_cols[@]}"; do
        ENCODE_ARGS+=(--column "$_c")
    done
    python3 tools/encodeCategorical.py \
        --input "$DATA_DIR/C_post_cellTypes.csv" \
        --output "$DATA_DIR/C_post_cellTypes.encoded.csv" \
        "${ENCODE_ARGS[@]}" \
        --min-cell-size "$ENCODE_MIN_CELL_SIZE" \
        --report "$DATA_DIR/encodeCategorical.json"
fi
fi

if [ "$START_STAGE" == "pca" ]; then EXECUTE=1; fi

# Stage 2: Residualization & PCA
if [ $EXECUTE -eq 1 ]; then
log "[2/9] Generating Expression and Methylation PCs..."
resolve_cov_file
log "Using covariate file: $COV_FILE"
if [ -s "$DATA_DIR/C.csv" ]; then
    log "C.csv already exists. Skipping Residualization and PCA generation."
else
    log "Running Expression Residualization & PCA..."
    ./tools/residualize_pca.sh "$DATA_DIR/G.csv" "$COV_FILE" "$DATA_DIR/G_PCs.csv" "Exp_PC"

    log "Running Methylation Residualization & PCA..."
    ./tools/residualize_pca.sh "$DATA_DIR/M.csv" "$COV_FILE" "$DATA_DIR/M_PCs.csv" "Meth_PC"

    log "Merging Covariates with PCs..."
    python3 -c "
import pandas as pd
C = pd.read_csv('$COV_FILE', dtype={0: str}, float_precision='round_trip')
C.set_index(C.columns[0], inplace=True)
G_PCs = pd.read_csv('$DATA_DIR/G_PCs.csv', dtype={0: str})
G_PCs.set_index(G_PCs.columns[0], inplace=True)
M_PCs = pd.read_csv('$DATA_DIR/M_PCs.csv', dtype={0: str})
M_PCs.set_index(M_PCs.columns[0], inplace=True)
C_final = pd.concat([C, G_PCs, M_PCs], axis=1)
C_final.to_csv('$DATA_DIR/C.csv')
# Record the shape the PCA merge actually produced so the next stage
# (pipeline.sh) can validate that C.csv still carries exactly this many
# samples and covariates before deriving DF = SAMPLES - COVARS - 2. A stray
# trailing blank line or extra index column would otherwise silently shift DF.
with open('$DATA_DIR/C.shape.meta', 'w') as fh:
    fh.write('samples=%d\ncovars=%d\n' % (C_final.shape[0], C_final.shape[1]))
"
fi
fi

if [ "$DATASET" == "gtp" ] || [ "$DATASET" == "gtpsub" ]; then
    log "GTP - Diagnosing expression PCs..."
    python3 -u tools/diagnoseExpressionPCs.py     --expression "$DATA_DIR/G.csv"     --covariates "$DATA_DIR/C_post_cellTypes.csv"
fi

# Data Exploration (moved to end to compare original to final processed state)
log "Exploring Omics data..."
python3 tools/exploreOmics.py \
    --input-processed-methylation "$DATA_DIR/M.csv" \
    --input-orig-methylation "$DATA_DIR/M_orig.csv" \
    --input-processed-expression "$DATA_DIR/G.csv" \
    --input-orig-expression "$DATA_DIR/G_orig.csv" \
    --output-dir "$DATA_DIR/qc"

log "======================================"
log "Preprocessing completed successfully!"
log "Data directory $DATA_DIR now contains M.csv, G.csv, and C.csv"
log "Dataset is ready for pipeline.sh"
log "======================================"
