#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# Log function for timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Default settings
DATASET="dummy"

# gtpsub-only: locus subsample targets (rows kept from full GTP M/G; samples untouched)
GTPSUB_M_LOCI=10000
GTPSUB_G_LOCI=5000
GTPSUB_SEED=42
START_STAGE="all"

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
    if [ "$DATASET" == "dummy" ]; then
        # Generate small synthetic data for testing
        log "Generating synthetic dummy data..."
        echo "10" | python3 -m tecpg data dummy -s 100 -m 20 -g 20
        mv data/* "$DATA_DIR/"
        mv annot/* "$ANNOT_DIR/"
        rmdir data annot
        mv "$DATA_DIR/C.csv" "$DATA_DIR/C_orig.csv"
    elif [ "$DATASET" == "gtp" ] || [ "$DATASET" == "gtpsub" ]; then
        log "Downloading GTP data..."
        echo "y" | python3 -m tecpg data gtp --yes
        mv data/* "$DATA_DIR/"

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
        rmdir data
    elif [ "$DATASET" == "mesa" ]; then
        log "Downloading MESA data..."
        echo "y" | python3 -m tecpg data mesa
        mv data/* "$DATA_DIR/"
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
        rmdir data
    fi
fi

# Apply probe blacklist filter (METH_ARRAY: 450k default, or epic/both)
if [ -s "$DATA_DIR/M.csv" ]; then
    log "M.csv already exists. Skipping probe blacklist filtering."
else
    log "Generating probe blacklist (array: ${METH_ARRAY:-450k})..."
    ./tools/generateProbeBlacklist.sh "$DATA_DIR" "${METH_ARRAY:-450k}"

    log "Applying blacklist filter to M_orig.csv..."
    python3 tools/exclude_blacklisted_probes.py "$DATA_DIR/M_orig.csv" "$DATA_DIR/probes_blacklist.csv" "$DATA_DIR/M.csv"
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

if [ "$START_STAGE" == "cell_prop" ]; then EXECUTE=1; fi

# Stage 1.5: Estimate Immune Cell Proportions
if [ $EXECUTE -eq 1 ]; then
log "[1.5/9] Estimating immune cell proportions using EpiDISH..."
if [ -s "$DATA_DIR/C_post_cellTypes.csv" ]; then
    log "C_post_cellTypes.csv already exists. Skipping cell proportion estimation."
else
    log "Running EpiDISH to estimate cell proportions..."
    if [ "$DATASET" == "dummy" ] || [ "$DATASET" == "gtpsub" ]; then
        log "Skipping EpiDISH for dummy data (random noise causes singular fits)."
        cp "$DATA_DIR/C_orig.csv" "$DATA_DIR/C_post_cellTypes.csv"
    else
        ./tools/estimateCellProportions.sh "$DATA_DIR/M.csv" "$DATA_DIR/C_orig.csv" "$DATA_DIR/C_post_cellTypes.csv" "$DATASET"
    fi
fi
fi

if [ "$START_STAGE" == "pca" ]; then EXECUTE=1; fi

# Stage 2: Residualization & PCA
if [ $EXECUTE -eq 1 ]; then
log "[2/9] Generating Expression and Methylation PCs..."
if [ -s "$DATA_DIR/C.csv" ]; then
    log "C.csv already exists. Skipping Residualization and PCA generation."
else
    log "Running Expression Residualization & PCA..."
    ./tools/residualize_pca.sh "$DATA_DIR/G.csv" "$DATA_DIR/C_post_cellTypes.csv" "$DATA_DIR/G_PCs.csv" "Exp_PC"

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
