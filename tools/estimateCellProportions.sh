#!/bin/bash

# estimateCellProportions.sh
# Wrapper to run the EpiDISH R script for cell proportion estimation.

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: ./estimateCellProportions.sh <methylation_file.csv> <covariates_file.csv> <output_file.csv> [cohort_name]"
    exit 1
fi

METH_FILE="$1"
COV_FILE="$2"
OUT_FILE="$3"
COHORT_NAME="${4:-}"

# Check if Rscript is available
if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript is not installed or not in PATH."
    echo "Please ensure R is installed in your environment to run EpiDISH."
    exit 1
fi

# Run the R script, assuming it's in the same directory as this wrapper
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RSCRIPT_PATH="$DIR/estimateCellProportions.R"

if [ -n "$COHORT_NAME" ]; then
    Rscript "$RSCRIPT_PATH" "$METH_FILE" "$COV_FILE" "$OUT_FILE" "$COHORT_NAME"
else
    Rscript "$RSCRIPT_PATH" "$METH_FILE" "$COV_FILE" "$OUT_FILE"
fi
