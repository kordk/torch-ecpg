#!/bin/bash
set -e

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
PLOTS_DIR="${OUT_DIR}/plots"
NETWORK_DIR="${OUT_DIR}/network"
ENRICHMENT_DIR="${OUT_DIR}/enrichment"
PARQUET_FILE="${OUT_DIR}/bootstrap_merged.parquet"
SUMMARIZED_PARQUET="${OUT_DIR}/summarized.parquet"
CONCORDANCE_PARQUET="${OUT_DIR}/bootstrap_concordance.parquet"
CONCORDANCE_SUMMARY="${OUT_DIR}/bootstrap_concordance_summary.json"

# Network export filtering defaults
# Universe: FDR-significant catalog (fdr_est <= NETWORK_MAX_FDR). NETWORK_TOP_K
# is a non-binding safety cap sized above the expected significant-pair count;
# it does not define the universe.
NETWORK_TOP_K=100000
NETWORK_MAX_FDR=0.05
# Stage-5 edge-weight threshold. The tool default (0.5 on mt_ig) re-truncates the
# FDR-significant universe; 0 renders the full exported edge set.
NETWORK_EDGE_THRESHOLD="${NETWORK_EDGE_THRESHOLD:-0}"

# Influence filter (INFLUENCE_MODE=exclude|ignore, default exclude): drop rows whose
# CpG carries mt_influence_flag == True from BOTH catalogs before any figure or
# enrichment stage, so every panel agrees on one retained universe. Null flags
# (CpGs with null mt_h_max) are retained. exclude requires the flag column
# (pipeline.sh stage 7b) and fails closed without it; ignore = pre-influence behavior.
INFLUENCE_MODE="${INFLUENCE_MODE:-exclude}"
RETAINED_DIR="${OUT_DIR}/retained"

log "======================================"
log "Starting Pipeline Post-Processing for DATASET: ${DATASET}"
log "======================================"

if [ ! -f "$PARQUET_FILE" ]; then
    log "Error: Expected input file $PARQUET_FILE not found!"
    log "Please ensure the tecpg pipeline has successfully generated bootstrap_merged.parquet."
    exit 1
fi

# Ensure output directories exist
mkdir -p "$PLOTS_DIR" "$NETWORK_DIR" "$ENRICHMENT_DIR"

# Stage 1a: Influence calibration bridge + dose-response figure (QC; report-only).
# MUST run on the UNFILTERED catalogs: it measures bootstrap fragility among
# flagged pairs, which the retained copies (Stage 1b) deliberately exclude.
# set -e is active, so a hard tool failure aborts the pipeline; catalogs that
# legitimately lack the required columns (pre-influence outputs, or bootstrap
# not yet run) are detected up front and the stage is skipped with a warning.
# INFLUENCE_BRIDGE=off disables the stage.
INFLUENCE_BRIDGE="${INFLUENCE_BRIDGE:-on}"
BRIDGE_DIR="${OUT_DIR}/calibration_bridge"
if [ "$INFLUENCE_BRIDGE" != "off" ]; then
    # Prefer the stage-7b flagged master when present (same rule as Stage 1b);
    # both carry mt_h_max, but this keeps the bridge and the filter on one source.
    BRIDGE_MASTER="$SUMMARIZED_PARQUET"
    if [ -f "$OUT_DIR/summarized.influence.parquet" ]; then
        BRIDGE_MASTER="$OUT_DIR/summarized.influence.parquet"
    fi
    BRIDGE_READY=$(python3 - "$BRIDGE_MASTER" "$PARQUET_FILE" <<'PYEOF'
import sys
import pyarrow.parquet as pq
try:
    m = set(pq.ParquetFile(sys.argv[1]).schema_arrow.names)
    b = set(pq.ParquetFile(sys.argv[2]).schema_arrow.names)
except Exception as e:
    print(f'no ({e})'); sys.exit(0)
if 'mt_h_max' not in m:
    print('no (master lacks mt_h_max)')
elif not {'p_boot', 'ci_low', 'ci_high'} <= b:
    print('no (bootstrap catalog lacks p_boot/ci columns)')
else:
    print('yes')
PYEOF
)
    if [ "$BRIDGE_READY" == "yes" ]; then
        log "[1a/9] Influence calibration bridge (mt_h_max x bootstrap fragility, unfiltered catalogs)..."
        mkdir -p "$BRIDGE_DIR"
        python3 -u tools/calibration_bridge.py \
            --master "$BRIDGE_MASTER" \
            --boot "$PARQUET_FILE" \
            --covariates "$DATA_DIR/C.csv" \
            --out-dir "$BRIDGE_DIR"
        log "[1a/9] Rendering influence figures (dose-response, SE-ratio)..."
        python3 -u tools/fig_influence_dose_response.py \
            --json "$BRIDGE_DIR/calibration_bridge.json" \
            --out-dir "$BRIDGE_DIR"
        log "Bridge report and figure written to $BRIDGE_DIR/."
    else
        log "[1a/9] SKIPPING calibration bridge: $BRIDGE_READY"
    fi
else
    log "[1a/9] Influence calibration bridge disabled (INFLUENCE_BRIDGE=off)."
fi

# Stage 1b: Influence filter (build retained catalogs; repoint all consumers)
if [ "$INFLUENCE_MODE" == "exclude" ]; then
    log "[1b/9] Applying influence filter (dropping mt_influence_flag CpGs)..."
    mkdir -p "$RETAINED_DIR"
    # The flag column lives on the stage-7b output when present; fall back to
    # summarized.parquet (which fails closed below unless it carries the column).
    INFLUENCE_SRC="$OUT_DIR/summarized.influence.parquet"
    if [ -f "$INFLUENCE_SRC" ]; then
        log "Flag source: $INFLUENCE_SRC"
        SUMMARIZED_PARQUET="$INFLUENCE_SRC"
    else
        log "Flag source: $SUMMARIZED_PARQUET (no summarized.influence.parquet found)"
    fi
    PARQUET_IN="$PARQUET_FILE" SUMMARIZED_IN="$SUMMARIZED_PARQUET" RETAINED_DIR="$RETAINED_DIR" \
    python3 - <<'PYEOF'
import os, sys
import pandas as pd

def load(path):
    df = pd.read_parquet(path)
    if df.index.names != [None]:
        df = df.reset_index()
    return df

srcs = {'bootstrap_merged.parquet': os.environ['PARQUET_IN'],
        'summarized.parquet': os.environ['SUMMARIZED_IN']}
out_dir = os.environ['RETAINED_DIR']
flag_src = load(srcs['summarized.parquet'])
if 'mt_influence_flag' not in flag_src.columns:
    sys.exit('INFLUENCE_MODE=exclude but summarized catalog lacks mt_influence_flag. '
             'Run pipeline.sh stage 7b (influence_flag) first, point this script at '
             'summarized.influence.parquet, or set INFLUENCE_MODE=ignore.')
flagged = set(flag_src.loc[flag_src['mt_influence_flag'] == True, 'mt_id'].astype(str))
print(f'flagged CpGs: {len(flagged)}')
for name, path in srcs.items():
    df = load(path)
    keep = df[~df['mt_id'].astype(str).isin(flagged)]
    print(f'{name}: {len(df):,} -> {len(keep):,} rows retained')
    keep.to_parquet(os.path.join(out_dir, name), index=False)
PYEOF
    PARQUET_FILE="$RETAINED_DIR/bootstrap_merged.parquet"
    SUMMARIZED_PARQUET="$RETAINED_DIR/summarized.parquet"
    log "Influence filter applied. Consumers repointed to $RETAINED_DIR/."
else
    log "[1b/9] Influence filter disabled (INFLUENCE_MODE=$INFLUENCE_MODE)."
fi

# Stage 1: Obtain cytoBand.txt if missing
log "[1/9] Checking for cytoBand.txt..."
if [ ! -f "cytoBand.txt" ]; then
    log "cytoBand.txt not found. Downloading from UCSC..."
    curl -O http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/cytoBand.txt.gz
    gunzip -f cytoBand.txt.gz
    log "cytoBand.txt downloaded and extracted."
else
    log "cytoBand.txt already exists. Skipping download."
fi

# Stage 2: Run plotCircos.py
log "[2/9] Running plotCircos.py..."
python3 -u tools/plotCircos.py -i "$PARQUET_FILE" --cytoband cytoBand.txt --out-dir "$PLOTS_DIR"

# Stage 3: Run visualizeFindings.py
log "[3/9] Running visualizeFindings.py..."
python3 -u tools/visualizeFindings.py --all -m "$DATA_DIR/M.csv" -g "$DATA_DIR/G.csv" -c "$DATA_DIR/C.csv" "$PARQUET_FILE" --out-dir "$PLOTS_DIR"

# Stage 4: Run exportBipartiteNetwork.py to generate cytoscape nodes/edges
log "[4/9] Generating Cytoscape network files..."
python3 -u tools/exportBipartiteNetwork.py \
    -i "$PARQUET_FILE" \
    -o cytoscape \
    --out-dir "$NETWORK_DIR" \
    --top-k "$NETWORK_TOP_K" \
    --max-fdr "$NETWORK_MAX_FDR"

# Stage 5: Run visualizeBipartiteNetwork.py
log "[5/9] Running visualizeBipartiteNetwork.py..."
python3 -u tools/visualizeBipartiteNetwork.py --edges "$NETWORK_DIR/cytoscape_edges.csv" --nodes "$NETWORK_DIR/cytoscape_nodes.csv" --out-dir "$NETWORK_DIR" --per-region --threshold "$NETWORK_EDGE_THRESHOLD"
# Stage 6: Run evaluateSaliency.py
log "[6/9] Running evaluateSaliency.py..."
python3 -u tools/evaluateSaliency.py -i "$PARQUET_FILE" -o "$PLOTS_DIR"

# Stage 7: Score bootstrap/analytic concordance. This annotates raw scores and
# reports their observed distribution; it sets no thresholds and filters nothing.
# The output is a separate parquet, so bootstrap_merged.parquet is unchanged and
# every existing consumer is unaffected.
log "[7/9] Scoring bootstrap/analytic concordance..."
python3 -u tools/annotate_bootstrap_concordance.py \
    -i "$PARQUET_FILE" \
    -o "$CONCORDANCE_PARQUET" \
    -s "$CONCORDANCE_SUMMARY"

# Stage 8: Run runEnrichment.py for functional (and optional ENCODE) enrichment.
# This analysis was previously bundled into tools/summarizeOutput_parquet.py and is
# now a standalone tool. It draws significant genes from the FDR summary
# (summarized.parquet) and from the bootstrap IG ranking (bootstrap_merged.parquet).
log "[8/9] Running functional enrichment analysis..."
python3 -u tools/runEnrichment.py \
    --fdr-input "$SUMMARIZED_PARQUET" \
    --ig-input "$PARQUET_FILE" \
    --out-dir "$ENRICHMENT_DIR" \
    --rank-by fdr ig

# Stage 8 (cont.): Render a self-contained HTML summary of the enrichment
# outputs (top-25 tables and figures per analysis, plus an overview).
log "[8/9] Rendering enrichment summary HTML..."
python3 -u tools/summarizeEnrichment.py \
    --enrichment-dir "$ENRICHMENT_DIR" \
    --out "$ENRICHMENT_DIR/enrichment_summary.html" \
    --top-n 25

# Stage 9: Consolidated influence QC report. Pure consumer of JSON produced
# earlier (flag QC from pipeline.sh stage 7b, bridge from stage 1a here, and the
# Kennedy influence stratification from pipelineBenchmarkKennedy.sh when it has
# been run). Every input is optional; missing sections render as "not available",
# so this never blocks the pipeline. INFLUENCE_REPORT=off disables it.
INFLUENCE_REPORT="${INFLUENCE_REPORT:-on}"
INFLUENCE_QC_JSON="${OUT_DIR}/influence_qc/influence_qc.json"
BRIDGE_JSON="${BRIDGE_DIR}/calibration_bridge.json"
KENNEDY_INFLUENCE_JSON="${OUT_DIR}/kennedy/influence_stratified.json"
FLAGGED_PARQUET="${OUT_DIR}/summarized.influence.parquet"
if [ "$INFLUENCE_REPORT" != "off" ]; then
    REPORT_ARGS=()
    [ -f "$INFLUENCE_QC_JSON" ] && REPORT_ARGS+=(--influence-qc "$INFLUENCE_QC_JSON")
    [ -f "$BRIDGE_JSON" ] && REPORT_ARGS+=(--bridge "$BRIDGE_JSON")
    [ -f "$KENNEDY_INFLUENCE_JSON" ] && REPORT_ARGS+=(--kennedy-influence "$KENNEDY_INFLUENCE_JSON")
    [ -f "$FLAGGED_PARQUET" ] && REPORT_ARGS+=(--flagged-parquet "$FLAGGED_PARQUET")
    if [ ${#REPORT_ARGS[@]} -eq 0 ]; then
        log "[9/9] SKIPPING influence QC report: no influence artifacts found."
    else
        log "[9/9] Rendering consolidated influence QC report..."
        python3 -u tools/influence_qc_report.py \
            --dataset "$DATASET" \
            "${REPORT_ARGS[@]}" \
            --out "${OUT_DIR}/influence_qc_report.html"
        log "Influence QC report: ${OUT_DIR}/influence_qc_report.html"
    fi
else
    log "[9/9] Influence QC report disabled (INFLUENCE_REPORT=off)."
fi

log "======================================"
log "Post-processing pipeline completed successfully!"
log "Outputs saved to ${OUT_DIR}/plots/, ${OUT_DIR}/network/, and ${OUT_DIR}/enrichment/"
log "======================================"
