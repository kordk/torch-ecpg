#!/bin/bash

# Wrapper to generate the Illumina probe blacklist.
# Usage: ./generateProbeBlacklist.sh <output_dir> [array] [out_filename]
#   array        450k (default) | epic | both
#   out_filename defaults to probes_blacklist.csv

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: ./generateProbeBlacklist.sh <output_dir> [450k|epic|both] [out_filename]"
    exit 1
fi

OUT_DIR="$1"
ARRAY="${2:-450k}"
OUT_FILE="${3:-probes_blacklist.csv}"

case "$ARRAY" in
    450k|epic|both) ;;
    *) echo "Error: array must be one of: 450k, epic, both (got '$ARRAY')"; exit 1 ;;
esac

if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript is not installed or not in PATH."
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RSCRIPT_PATH="$DIR/generateProbeBlacklist.R"

# The R script writes to its working directory, so the output dir must exist.
# Create it rather than failing with a bare `cd` error.
mkdir -p "$OUT_DIR" || { echo "Error: cannot create output dir '$OUT_DIR'"; exit 1; }

# Run in the output directory so the file is written there.
EXTRA=""
[ "${SELFTEST:-0}" == "1" ] && EXTRA="--selftest"

(cd "$OUT_DIR" && Rscript "$RSCRIPT_PATH" --array="$ARRAY" --out="$OUT_FILE" $EXTRA)
