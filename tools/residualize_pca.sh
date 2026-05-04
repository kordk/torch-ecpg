#!/bin/bash

# residualize_pca.sh
# Wrapper to run the Python residualization and PCA script.

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <input_matrix.csv> <covariates_file.csv> <output_file.csv> <prefix>"
    exit 1
fi

INPUT_FILE="$1"
COV_FILE="$2"
OUT_FILE="$3"
PREFIX="$4"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$DIR/residualize_pca.py"

python3 "$PYTHON_SCRIPT" -i "$INPUT_FILE" -c "$COV_FILE" -o "$OUT_FILE" -p "$PREFIX" -n 5