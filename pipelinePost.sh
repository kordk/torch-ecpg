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
PARQUET_FILE="${OUT_DIR}/bootstrap_merged.parquet"

# Network export filtering defaults
NETWORK_TOP_K=5000
NETWORK_MAX_BOOT_P=0.05

log "======================================"
log "Starting Pipeline Post-Processing for DATASET: ${DATASET}"
log "======================================"

if [ ! -f "$PARQUET_FILE" ]; then
    log "Error: Expected input file $PARQUET_FILE not found!"
    log "Please ensure the tecpg pipeline has successfully generated bootstrap_merged.parquet."
    exit 1
fi

# Ensure output directories exist
mkdir -p "$PLOTS_DIR" "$NETWORK_DIR"

# Stage 1: Obtain cytoBand.txt if missing
log "[1/6] Checking for cytoBand.txt..."
if [ ! -f "cytoBand.txt" ]; then
    log "cytoBand.txt not found. Downloading from UCSC..."
    curl -O http://hgdownload.cse.ucsc.edu/goldenPath/hg19/database/cytoBand.txt.gz
    gunzip -f cytoBand.txt.gz
    log "cytoBand.txt downloaded and extracted."
else
    log "cytoBand.txt already exists. Skipping download."
fi

# Stage 2: Run plotCircos.py
log "[2/6] Running plotCircos.py..."
python3 -u tools/plotCircos.py -i "$PARQUET_FILE" --cytoband cytoBand.txt --out-dir "$PLOTS_DIR"

# Stage 3: Run visualizeFindings.py
log "[3/6] Running visualizeFindings.py..."
python3 -u tools/visualizeFindings.py --all -m "$DATA_DIR/M.csv" -g "$DATA_DIR/G.csv" -c "$DATA_DIR/C.csv" "$PARQUET_FILE" --out-dir "$PLOTS_DIR"

# Stage 4: Run exportBipartiteNetwork.py to generate cytoscape nodes/edges
log "[4/6] Generating Cytoscape network files..."
python3 -u tools/exportBipartiteNetwork.py \
    -i "$PARQUET_FILE" \
    -o cytoscape \
    --out-dir "$NETWORK_DIR" \
    --top-k "$NETWORK_TOP_K" \
    --max-boot-p "$NETWORK_MAX_BOOT_P"

# Stage 5: Run visualizeBipartiteNetwork.py
log "[5/6] Running visualizeBipartiteNetwork.py..."
python3 -u tools/visualizeBipartiteNetwork.py --edges "$NETWORK_DIR/cytoscape_edges.csv" --nodes "$NETWORK_DIR/cytoscape_nodes.csv" --out-dir "$NETWORK_DIR"
# Stage 6: Run evaluateSaliency.py
log "[6/6] Running evaluateSaliency.py..."
python3 -u tools/evaluateSaliency.py -i "$PARQUET_FILE" -o "$PLOTS_DIR"

log "======================================"
log "Post-processing pipeline completed successfully!"
log "Outputs saved to ${OUT_DIR}/plots/ and ${OUT_DIR}/network/"
log "======================================"
