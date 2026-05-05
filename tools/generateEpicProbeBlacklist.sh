#!/bin/bash

# Wrapper to run the R script for generating the EPIC probe blacklist

if [ "$#" -ne 1 ]; then
    echo "Usage: ./generateEpicProbeBlacklist.sh <output_dir>"
    exit 1
fi

OUT_DIR="$1"

# Check if Rscript is available
if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript is not installed or not in PATH."
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RSCRIPT_PATH="$DIR/generateEpicProbeBlacklist_v2.R"

# Run the script in the specified output directory to save the file there
(cd "$OUT_DIR" && Rscript "$RSCRIPT_PATH")
