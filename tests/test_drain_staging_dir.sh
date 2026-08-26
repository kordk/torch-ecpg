#!/bin/bash
# Regression harness for drain_staging_dir() in pipelinePre.sh.
# The GTP run of 2026-08-26 aborted at "rmdir: failed to remove 'data'" AFTER a
# successful 7-minute download: mv dir/* skips dotfiles, so the staging dir was
# left non-empty and a cleanup step killed the run under set -e.
cd "$( dirname "${BASH_SOURCE[0]}" )/.." || exit 1
set -e
source <(sed -n '/^drain_staging_dir()/,/^}/p' pipelinePre.sh)
log(){ echo "[log] $1"; }
P=0;F=0; ok(){ [ "$2" == "$3" ] && { echo "  PASS $1"; P=$((P+1)); } || { echo "  FAIL $1: want '$3' got '$2'"; F=$((F+1)); }; }

echo "T1: normal case -- all files moved, dir removed"
D=$(mktemp -d); mkdir -p $D/src $D/dst; touch $D/src/{M.csv,G.csv,C.csv}
drain_staging_dir $D/src $D/dst >/dev/null
ok "files moved" "$(ls $D/dst | wc -l)" "3"
ok "src removed" "$([ -d $D/src ] && echo yes || echo no)" "no"; rm -rf $D

echo "T2: dotfiles are moved too (mv dir/* missed these)"
D=$(mktemp -d); mkdir -p $D/src $D/dst; touch $D/src/M.csv $D/src/.hidden
drain_staging_dir $D/src $D/dst >/dev/null
ok "dotfile moved" "$([ -e $D/dst/.hidden ] && echo yes || echo no)" "yes"
ok "src removed" "$([ -d $D/src ] && echo yes || echo no)" "no"; rm -rf $D

echo 'T3: subdirectories are moved too (so only DOTFILES survived the old mv dir/*)'
D=$(mktemp -d); mkdir -p $D/src/sub $D/dst; touch $D/src/M.csv $D/src/sub/x
drain_staging_dir $D/src $D/dst >/dev/null
ok "subdir moved" "$([ -e $D/dst/sub/x ] && echo yes || echo no)" "yes"
ok "src removed" "$([ -d $D/src ] && echo yes || echo no)" "no"; rm -rf $D

echo "T3b: if rmdir fails, warn and CONTINUE (never abort the run)"
D=$(mktemp -d); mkdir -p $D/src $D/dst; touch $D/src/M.csv
rmdir(){ return 1; }          # force the failure branch deterministically
set +e; OUT=$(drain_staging_dir $D/src $D/dst 2>&1); RC=$?; set -e
unset -f rmdir
ok "returns success" "$RC" "0"
ok "warns" "$(echo "$OUT" | grep -c 'not empty')" "1"
ok "files still moved" "$([ -e $D/dst/M.csv ] && echo yes || echo no)" "yes"; rm -rf $D

echo "T4: empty staging dir"
D=$(mktemp -d); mkdir -p $D/src $D/dst
drain_staging_dir $D/src $D/dst >/dev/null
ok "src removed" "$([ -d $D/src ] && echo yes || echo no)" "no"; rm -rf $D

echo "T5: missing staging dir is a no-op"
D=$(mktemp -d); mkdir -p $D/dst
set +e; drain_staging_dir $D/nope $D/dst >/dev/null; RC=$?; set -e
ok "returns success" "$RC" "0"; rm -rf $D
echo; echo "passed=$P failed=$F"; [ $F -eq 0 ]
