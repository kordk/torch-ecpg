#!/bin/bash
# Regression harness: pipelinePre.sh must support concurrent runs for different
# datasets in one working directory.
#
# 2026-08-26: a GTP and a MESA run collided. `tecpg data <ds>` stages into a
# shared ./data (config input_dir), so run B's staging repopulated ./data while
# run A was cleaning up; A's `rmdir data` failed and set -e killed A after a
# successful 7-minute download.
cd "$( dirname "${BASH_SOURCE[0]}" )/.." || exit 1
P=0; F=0
ok(){ [ "$2" == "$3" ] && { echo "  PASS $1"; P=$((P+1)); } || { echo "  FAIL $1: want '$3' got '$2'"; F=$((F+1)); }; }

echo "T1: staging dirs are per-dataset (the fix)"
# Assert against the SCRIPT's actual definition -- reconstructing the pattern
# here would pass even if the script used one shared staging dir.
ok "STAGE_DIR is dataset-scoped" \
   "$(grep -c 'STAGE_DIR="data_stage_${DATASET}"' pipelinePre.sh || true)" "1"
ok "STAGE_ANNOT_DIR is dataset-scoped" \
   "$(grep -c 'STAGE_ANNOT_DIR="annot_stage_${DATASET}"' pipelinePre.sh || true)" "1"
# and no un-scoped staging assignment sneaks in
ok "no unscoped STAGE_DIR" \
   "$(grep -cE '^STAGE_DIR="[^$]*"$' pipelinePre.sh || true)" "0"

echo "T2: script no longer references the shared ./data staging area"
ok "no 'mv data/*'"  "$(grep -c 'mv data/\*' pipelinePre.sh || true)" "0"
ok "no 'rmdir data'" "$(grep -c '^\s*rmdir data' pipelinePre.sh || true)" "0"

echo "T3: tecpg is told where to stage (-i), for every dataset branch"
ok "gtp passes -i"   "$(grep -c 'tecpg -i "\$STAGE_DIR" data gtp' pipelinePre.sh || true)" "1"
ok "mesa passes -i"  "$(grep -c 'tecpg -i "\$STAGE_DIR" data mesa' pipelinePre.sh || true)" "1"
ok "dummy passes -i and -a" "$(grep -c 'tecpg -i "\$STAGE_DIR" -a "\$STAGE_ANNOT_DIR" data dummy' pipelinePre.sh || true)" "1"

echo "T4: staging is never pointed at DATA_DIR (initialize_dir would rmtree it)"
ok "no -i \$DATA_DIR" "$(grep -c '\-i "\$DATA_DIR"' pipelinePre.sh || true)" "0"

echo "T5: simulated concurrent drain -- two datasets do not interfere"
source <(sed -n '/^drain_staging_dir()/,/^}/p' pipelinePre.sh)
log(){ :; }
W=$(mktemp -d); mkdir -p "$W"/{data_stage_gtp,data_stage_mesa,data_gtp,data_mesa}
touch "$W"/data_stage_gtp/{M.csv,G.csv}; touch "$W"/data_stage_mesa/{M.csv,G.csv}
( cd "$W" && drain_staging_dir data_stage_gtp data_gtp )
ok "gtp drained" "$(ls "$W"/data_gtp | wc -l)" "2"
ok "mesa staging untouched" "$(ls "$W"/data_stage_mesa | wc -l)" "2"
( cd "$W" && drain_staging_dir data_stage_mesa data_mesa )
ok "mesa drained" "$(ls "$W"/data_mesa | wc -l)" "2"
ok "gtp data intact" "$(ls "$W"/data_gtp | wc -l)" "2"
rm -rf "$W"

echo; echo "passed=$P failed=$F"; [ $F -eq 0 ]
