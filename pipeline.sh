#!/bin/bash
set -e

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
    M_CHUNK=10000
    G_CHUNK=10000
elif [ "$DATASET" == "mesa" ]; then
    TOTAL_TESTS=10000000000 # Placeholder for MESA
    M_CHUNK=10000
    G_CHUNK=10000
elif [ "$DATASET" == "dummy" ]; then
    TOTAL_TESTS=1000000 # 1000 M * 1000 G
    M_CHUNK=500
    G_CHUNK=500
else
    echo "Unknown dataset: $DATASET"
    echo "Usage: ./pipeline.sh [gtp|mesa|dummy]"
    exit 1
fi

echo "======================================"
echo "Starting eQTM Pipeline for: $DATASET"
echo "======================================"

# Setup directories
OUT_DIR="output_${DATASET}"
DATA_DIR="data_${DATASET}"
ANNOT_DIR="annot_${DATASET}"
mkdir -p "$OUT_DIR" "$DATA_DIR" "$ANNOT_DIR"

# Stage 1: Data Preparation
echo "[1/8] Preparing data..."
if [ "$DATASET" == "dummy" ]; then
    # Generate small synthetic data for testing
    echo "10" | tecpg data dummy -s 100 -m 1000 -g 1000
    mv data/* "$DATA_DIR/"
    mv annot/* "$ANNOT_DIR/"
    rmdir data annot
elif [ "$DATASET" == "gtp" ]; then
    echo "y" | tecpg data gtp
    mv data/* "$DATA_DIR/"
    # For GTP, assuming the demo annots are used
    cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6"
    cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6"
    rmdir data
elif [ "$DATASET" == "mesa" ]; then
    echo "y" | tecpg data mesa
    mv data/* "$DATA_DIR/"
    # For MESA, assuming appropriate demo annots are used if available
    # Or fall back to EPIC/HT12 for now
    cp demo/annoEPIC.hg19.bed6 "$ANNOT_DIR/M.bed6" 2>/dev/null || true
    cp demo/annoHT12.hg19.bed6 "$ANNOT_DIR/G.bed6" 2>/dev/null || true
    rmdir data
fi

# Determine Degrees of Freedom for P-value calculation
# SAMPLES - COVARIATES - 1 (M) - 1 (Intercept)
# Using placeholder calculation if needed; assuming df=96 for dummy (100 - 2 - 2)
# Here we just parse the C.csv lines dynamically
SAMPLES=$(wc -l < "$DATA_DIR/C.csv")
SAMPLES=$((SAMPLES - 1)) # Header
COVARS=$(head -n 1 "$DATA_DIR/C.csv" | awk -F, '{print NF-1}')
DF=$((SAMPLES - COVARS - 2))

echo "Calculated Degrees of Freedom: $DF"

# Stage 2: Mapping (lstsq + ig)
echo "[2/8] Performing eQTM Mapping (lstsq + IG)..."
tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq --all -m "$M_CHUNK" -g "$G_CHUNK" --compute-ig

# Stage 3: Merge chunked outputs
echo "[3/8] Merging chunked outputs to Parquet..."
MERGED_PARQUET="$OUT_DIR/merged.parquet"
python3 tools/mergeOutputs.py --format parquet "$OUT_DIR" "$MERGED_PARQUET"

# Clean up CSV chunks to save space
rm "$OUT_DIR"/*-*.csv || true

# Stage 4: Annotate regions
echo "[4/8] Annotating regions..."
ANNOTATED_PARQUET="$OUT_DIR/annotated.parquet"
python3 tools/assignRegionToEcpg_parquet.py -d "$MERGED_PARQUET" -g "$ANNOT_DIR/G.bed6" -m "$ANNOT_DIR/M.bed6" -o "$ANNOTATED_PARQUET"

# Stage 5: Precise P-value recalculation
echo "[5/8] Recalculating precise p-values..."
RECALC_PARQUET="$OUT_DIR/annotated_pcalc.parquet"
python3 tools/recalculate_pvalues_parquet.py "$ANNOTATED_PARQUET" --df "$DF" --output-file "$RECALC_PARQUET"

# Stage 6: Summarize & FDR
echo "[6/8] Calculating FDR and summarizing..."
SUMMARIZED_PARQUET="$OUT_DIR/summarized.parquet"
python3 tools/summarizeOutput_parquet.py --main-file "$RECALC_PARQUET" --reservoir-file "$OUT_DIR/sample_reservoir.csv" --total-tests "$TOTAL_TESTS" --df "$DF" --calculate-fdr --output-fdr-file "$SUMMARIZED_PARQUET"
# Ensure plots are created in the right folder, but for now they go to CWD based on tool
mv p_value_histogram.png qq_plot.png saliency_profile_top50.png "$OUT_DIR/" || true

# Stage 7: Bootstrap List creation
echo "[7/8] Creating Bootstrap List..."
BOOTSTRAP_LIST="$OUT_DIR/bootstrap_list.csv"
python3 tools/createBootstrapList.py --input "$SUMMARIZED_PARQUET" --output "$BOOTSTRAP_LIST" --rank-by p-value

# Stage 8: Bootstrap evaluation
echo "[8/8] Bootstrapping top hits..."
tecpg -i "$DATA_DIR" -a "$ANNOT_DIR" -o "$OUT_DIR" run mlr --mlr-method lstsq_bootstrap --pairs-file "$BOOTSTRAP_LIST" --master-parquet "$SUMMARIZED_PARQUET" --bootstrap-iterations 100 --bootstrap-batch-size 10 --compute-ig

echo "======================================"
echo "Pipeline completed successfully!"
echo "Outputs are in $OUT_DIR/"
echo "======================================"
