#!/bin/bash
# Regression harness for the pipelinePre.sh blacklist stage (F1).
# Extracts the stage's logic and runs it against fixtures, so the two original
# defects cannot come back unnoticed:
#   F1a  guard keyed on M.csv existence -> stage never ran on a fresh build
#   F1b  filter read M_orig.csv -> resurrected loci that drop_na removed
set -uo pipefail
REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PASS=0; FAIL=0
ok(){ if [ "$2" == "$3" ]; then echo "  PASS $1"; PASS=$((PASS+1)); else echo "  FAIL $1: expected '$3' got '$2'"; FAIL=$((FAIL+1)); fi; }

setup() {
    D=$(mktemp -d)
    # M.csv as stage 1 leaves it: post-dropna, still containing blacklisted probes
    printf 'probe,s1,s2\ncg01,0.1,0.2\ncg02,0.3,0.4\nbad1,0.5,0.6\nbad2,0.7,0.8\n' > "$D/M.csv"
    # M_orig.csv: pre-dropna snapshot, carries an extra NaN locus
    printf 'probe,s1,s2\ncg01,0.1,0.2\ncg02,0.3,0.4\nbad1,0.5,0.6\nbad2,0.7,0.8\ncgNaN,,0.9\n' > "$D/M_orig.csv"
    printf 'Probe_ID,Reason\nbad1,SNP\nbad2,SEXCHROM\n' > "$D/probes_blacklist.csv"
}

# Mirrors pipelinePre.sh's stage: marker guard, filter M.csv via tmp, then stamp.
run_stage() {
    local DATA_DIR="$1"
    local BLACKLIST_MARKER="$DATA_DIR/M.csv.blacklist.meta"
    if [ -s "$BLACKLIST_MARKER" ]; then echo "skipped"; return 0; fi
    if [ ! -s "$DATA_DIR/M.csv" ]; then echo "error"; return 1; fi
    local TMP="$DATA_DIR/M.csv.blacklist.tmp"
    rm -f "$TMP"
    python3 "$REPO/tools/exclude_blacklisted_probes.py" \
        "$DATA_DIR/M.csv" "$DATA_DIR/probes_blacklist.csv" "$TMP" >/dev/null 2>&1 || return 1
    mv "$TMP" "$DATA_DIR/M.csv"
    {
        echo "applied=$(date +'%Y-%m-%dT%H:%M:%S')"
        echo "blacklist_rows=$(( $(wc -l < "$DATA_DIR/probes_blacklist.csv") - 1 ))"
        echo "m_rows_after=$(( $(wc -l < "$DATA_DIR/M.csv") - 1 ))"
    } > "$BLACKLIST_MARKER"
    echo "applied"
}

echo "T1: fresh build -- the stage actually runs (F1a)"
setup
ok "stage ran" "$(run_stage "$D")" "applied"
ok "blacklisted probe removed" "$(grep -c '^bad1,' "$D/M.csv" || true)" "0"
ok "clean probe retained" "$(grep -c '^cg01,' "$D/M.csv" || true)" "1"
ok "rows after filter" "$(( $(wc -l < "$D/M.csv") - 1 ))" "2"

echo "T2: dropped loci are NOT resurrected (F1b)"
ok "cgNaN absent from M.csv" "$(grep -c '^cgNaN,' "$D/M.csv" || true)" "0"
ok "M_orig.csv untouched" "$(( $(wc -l < "$D/M_orig.csv") - 1 ))" "5"

echo "T3: marker records the outcome"
ok "marker written" "$([ -s "$D/M.csv.blacklist.meta" ] && echo yes || echo no)" "yes"
ok "records rows after" "$(grep -c '^m_rows_after=2$' "$D/M.csv.blacklist.meta" || true)" "1"

echo "T4: idempotent -- second run is a no-op"
BEFORE=$(sha256sum "$D/M.csv" | cut -d' ' -f1)
ok "second run skips" "$(run_stage "$D")" "skipped"
ok "M.csv unchanged" "$(sha256sum "$D/M.csv" | cut -d' ' -f1)" "$BEFORE"

echo "T5: no temp file left behind"
ok "tmp cleaned up" "$([ -e "$D/M.csv.blacklist.tmp" ] && echo present || echo absent)" "absent"
rm -rf "$D"

echo "T6: missing M.csv is an error, not a silent skip"
setup; rm "$D/M.csv"
ok "returns error" "$(run_stage "$D")" "error"
rm -rf "$D"

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]
