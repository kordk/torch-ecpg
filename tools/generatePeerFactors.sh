#!/bin/bash

# generatePeerFactors.sh
# Wrapper to run the PEER factors R script.

if [ "$#" -ne 3 ]; then
    echo "Usage: ./generatePeerFactors.sh <gene_expression_file.csv> <covariates_file.csv> <output_file.csv>"
    exit 1
fi

EXPR_FILE="$1"
COV_FILE="$2"
OUT_FILE="$3"

# Check if Rscript is available
if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript is not installed or not in PATH."
    echo "Please ensure R is installed in your environment."
    exit 1
fi

# Run the R script, assuming it's in the same directory as this wrapper
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RSCRIPT_PATH="$DIR/generatePeerFactors.R"

Rscript "$RSCRIPT_PATH" "$EXPR_FILE" "$COV_FILE" "$OUT_FILE"
