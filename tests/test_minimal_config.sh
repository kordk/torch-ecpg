#!/usr/bin/env bash
#
# tests/test_minimal_config.sh
#
# End-to-end smoke test: verify that the tecpg pipeline still runs to
# completion under a simulated minimum-config host (8-core / 16 GB RAM,
# no GPU). This guards the small-host path against regressions from
# the host-profile auto-tuning work (3a/3b).
#
# What this script does:
#
#   1. Generates a tiny synthetic dataset (M, G, C, M.bed6, G.bed6).
#   2. Runs `tecpg run mlr --mlr-method qr --all` with chunking
#      forced on, --host-profile minimum, --cpu-threads 1 (CPU-only
#      execution path).
#   3. Asserts the regression output files were written and contain
#      data.
#   4. Repeats with explicit --output-format csv to confirm the
#      historical CSV writer still works on the minimum profile.
#
# Run from the repo root:
#
#     bash tests/test_minimal_config.sh
#
# Exit code 0 on success, non-zero on any failure.

set -euo pipefail

# Repo root = parent of this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

WORK_DIR="$(mktemp -d -t tecpg-min-config-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "[test_minimal_config] work dir: $WORK_DIR"
echo "[test_minimal_config] repo root: $REPO_ROOT"

# 1. Generate a tiny dataset with the bundled test_data helper. We use
#    a small enough size to fit comfortably on a 16 GB host but big
#    enough to exercise both the chunked save path and the index
#    construction code.
INPUT_DIR="$WORK_DIR/data"
ANNOT_DIR="$WORK_DIR/annot"
OUTPUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$INPUT_DIR" "$ANNOT_DIR" "$OUTPUT_DIR" "$LOG_DIR"
export WORK_DIR

python - <<'PYGEN'
import os
from tecpg.test_data import generate_data

work = os.environ['WORK_DIR']
M, G, C, M_annot, G_annot = generate_data(
    sample_size=20, m_rows=80, g_rows=40, annotation=True
)
M.to_csv(os.path.join(work, 'data', 'M.csv'))
G.to_csv(os.path.join(work, 'data', 'G.csv'))
C.to_csv(os.path.join(work, 'data', 'C.csv'))
# annotation files are .bed6 by default in cli config, tab-separated
M_annot.to_csv(os.path.join(work, 'annot', 'M.bed6'), sep='\t', index=False)
G_annot.to_csv(os.path.join(work, 'annot', 'G.bed6'), sep='\t', index=False)
print(f"generated M={M.shape}, G={G.shape}, C={C.shape}")
PYGEN

run_mlr() {
    local label="$1"
    shift
    local out="$OUTPUT_DIR/$label"
    rm -rf "$out"
    echo
    echo "[test_minimal_config] === case: $label ==="
    # --cpu-threads 1 forces the CPU code path (no CUDA dependency).
    # --host-profile minimum forces the conservative defaults regardless
    # of the actual hardware running CI.
    # --gene-loci-per-chunk 20 --meth-loci-per-chunk 40 forces the chunked save path.
    tecpg \
        -r "$WORK_DIR" \
        -i data \
        -a annot \
        -o "output/$label" \
        -l "logs" \
        --host-profile minimum \
        --cpu-threads 1 \
        run mlr \
            --mlr-method qr \
            --all \
            --gene-loci-per-chunk 20 \
            --meth-loci-per-chunk 40 \
            "$@"
}

assert_files_present() {
    local label="$1"
    local pattern="$2"
    local out="$OUTPUT_DIR/$label"
    local count
    count=$(find "$out" -type f -name "$pattern" | wc -l)
    if [ "$count" -lt 1 ]; then
        echo "[test_minimal_config] FAIL: no '$pattern' files in $out"
        find "$out" -maxdepth 2 -type f | sed 's/^/    /'
        exit 1
    fi
    # At least one of those files must be non-empty.
    local nonempty
    nonempty=$(find "$out" -type f -name "$pattern" -size +0c | wc -l)
    if [ "$nonempty" -lt 1 ]; then
        echo "[test_minimal_config] FAIL: every '$pattern' file in $out is empty"
        exit 1
    fi
    echo "[test_minimal_config] OK: $count '$pattern' file(s) written, >=1 non-empty"
}

# 2. Auto output-format. It must resolve to parquet.
run_mlr auto_format
assert_files_present auto_format '*.parquet'

# 3. Explicit --output-format csv (historical default behavior).
run_mlr explicit_csv --output-format csv
assert_files_present explicit_csv '*.csv'

# 4. Explicit --output-format parquet on minimum profile (user override
#    must still work).
run_mlr explicit_parquet --output-format parquet
assert_files_present explicit_parquet '*.parquet'

echo
echo "[test_minimal_config] ALL CASES PASSED"
