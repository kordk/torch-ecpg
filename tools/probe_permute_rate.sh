#!/usr/bin/env bash
# probe_permute_rate.sh — measure the per-permutation cost, then size the real run.
#
#   Usage:  CUDA_VISIBLE_DEVICES=0 ./probe_permute_rate.sh gtp  [probe_perms] [budget_hours]
#
#   Defaults: probe_perms=10, budget_hours=14
#
# WHY THIS IS NOT WASTED WORK:
#   pipelinePermute.sh has no --out-dir override (OUT_DIR is hardcoded to
#   output_$DATASET), so a probe cannot be sandboxed. Instead this IS the setup
#   run: --start-stage all builds cis_map_write_all.parquet and the
#   gene_anchored_master.parquet, both of which the real run reuses via
#   --start-stage permute. The only throwaway output is a 10-permutation
#   permutation_results.parquet, which the real run overwrites.
#
# It also exercises the two things that landed today -- the null sidecar and the
# per-region delta CIs -- end to end, at hour 1 instead of hour 18.
#
# Set CUDA_VISIBLE_DEVICES yourself. This script does not set it.

set -uo pipefail

D="${1:?usage: CUDA_VISIBLE_DEVICES=N $0 <gtp|mesa> [probe_perms] [budget_hours]}"
PROBE_PERMS="${2:-10}"
BUDGET_H="${3:-14}"
OUT_DIR="output_${D}"
LOG="probe_permute.${D}.out"

echo "=================================================================="
echo " Permute timing probe"
echo "   dataset          : $D"
echo "   probe permutations: $PROBE_PERMS"
echo "   budget for real run: ${BUDGET_H}h"
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<UNSET -- will use cuda:0>}"
echo "   repo SHA         : $(git rev-parse --short HEAD)"
echo "=================================================================="

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo
    echo "WARNING: CUDA_VISIBLE_DEVICES is unset. tecpg/config.py uses a bare"
    echo "         torch.device('cuda'), so this will run on index 0 (the A2)."
    echo "         Ctrl-C now if that is not what you want."
    sleep 10
fi

# --- preconditions ---------------------------------------------------------
FAIL=0
[ -s "$OUT_DIR/sample_reservoir.csv" ] \
  || { echo "  [FAIL] $OUT_DIR/sample_reservoir.csv missing -- the permute stage hard-fails without it."; FAIL=1; }
[ -d "$OUT_DIR" ] || { echo "  [FAIL] $OUT_DIR does not exist."; FAIL=1; }
[ "$FAIL" -eq 0 ] || { echo "Preconditions failed. Not starting."; exit 1; }
echo "  [ok] reservoir present: $(ls -la "$OUT_DIR/sample_reservoir.csv" | awk '{print $5" bytes, "$6" "$7" "$8}')"
echo

# --- run -------------------------------------------------------------------
START_EPOCH=$(date +%s)
echo "Started $(date +'%F %T'). Streaming to $LOG"
/usr/bin/time -v ./pipelinePermute.sh -d "$D" --permutations "$PROBE_PERMS" \
    >"$LOG" 2>&1
RC=$?
TOTAL=$(( $(date +%s) - START_EPOCH ))
echo "Finished $(date +'%F %T'). rc=$RC total=$((TOTAL/60))m$((TOTAL%60))s"
echo

if [ "$RC" -ne 0 ]; then
    echo "  [FAIL] pipeline exited $RC. Last 30 lines:"
    tail -30 "$LOG"
    exit "$RC"
fi

# --- extract the permute-stage duration ------------------------------------
# Anchors are wrapper log() lines, which carry [YYYY-MM-DD HH:MM:SS].
python3 - "$LOG" "$PROBE_PERMS" "$BUDGET_H" <<'PY'
import re, sys, datetime

log_path, probe_perms, budget_h = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
txt = open(log_path, errors='replace').read()

def stamp(pattern):
    m = re.search(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*' + pattern, txt, re.M)
    return datetime.datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S') if m else None

t_perm  = stamp(r'\[2/5\] Running permute')
t_eval  = stamp(r'\[3/5\] Running eval')
t_cis   = stamp(r'\[cis-map\]')
t_asm   = stamp(r'\[assemble\]')

print("=== Stage timings ===")
if t_cis and t_asm:
    print("  cis map      : %5.1f min" % ((t_asm - t_cis).total_seconds() / 60))
if t_asm and t_perm:
    print("  assemble+ann : %5.1f min" % ((t_perm - t_asm).total_seconds() / 60))

if not (t_perm and t_eval):
    print("\n  [WARN] Could not locate both permute/eval banners; timing unavailable.")
    print("         Size the real run by hand from the wall-clock above.")
    sys.exit(0)

perm_s = (t_eval - t_perm).total_seconds()
per_perm = perm_s / probe_perms
print("  permute      : %5.1f min  (%d permutations)" % (perm_s / 60, probe_perms))
print("  post-permute : %5.1f min  (eval + summary + QC + annotate)"
      % ((datetime.datetime.strptime(
            re.findall(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]', txt, re.M)[-1],
            '%Y-%m-%d %H:%M:%S') - t_eval).total_seconds() / 60))
print()
print("  per-permutation: %.1f s" % per_perm)

# --- size the real run -----------------------------------------------------
# Fixed cost = everything after the permute loop. cis map and assemble are NOT
# repeated, because the real run uses --start-stage permute.
last = datetime.datetime.strptime(
    re.findall(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]', txt, re.M)[-1],
    '%Y-%m-%d %H:%M:%S')
fixed_s = (last - t_eval).total_seconds()

budget_s = budget_h * 3600
usable = budget_s - fixed_s - 0.10 * budget_s   # 10% margin
n_rec = int(usable / per_perm)
n_rec = max(100, (n_rec // 100) * 100)          # round down to a round number

print()
print("=== Recommendation for the real run ===")
print("  budget            : %.1f h" % budget_h)
print("  fixed post-permute: %.1f min" % (fixed_s / 60))
print("  10%% safety margin : %.1f min" % (0.10 * budget_s / 60))
print("  --permutations    : %d" % n_rec)
print("    -> projected permute time %.1f h, total %.1f h"
      % (n_rec * per_perm / 3600, (n_rec * per_perm + fixed_s) / 3600))
print()
print("  Sanity: %d perms is %.0fx the probe. Default is 100." % (n_rec, n_rec / probe_perms))
if n_rec < 100:
    print("  [WARN] Budget does not even cover the default of 100. Reduce scope,")
    print("         or accept a longer run than the stated budget.")
PY

# --- verify today's two new artifacts --------------------------------------
echo
echo "=== Sidecar written? ==="
SC="$OUT_DIR/permutation_results.perm_null.npz"
if [ -s "$SC" ]; then
    python3 - "$SC" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
topk, total = z['topk_values'], int(z['total_count'])
print("  [ok] %s" % sys.argv[1])
print("       total null draws  : %d" % total)
print("       empirical floor   : %.3e" % (1.0 / (total + 1)))
print("       null max |t|      : %.4f" % float(topk.max()))
print("       observed max |t|  : %.4f" % float(z['observed_max_abs_t']))
print("       extrapolation gap : %.4f" % (float(z['observed_max_abs_t']) - float(topk.max())))
print("       overflow fraction : %.3e" % (int(z['overflow_count']) / total))
print("       gpd status        : %s" % str(z['gpd_status']))
PY
else
    echo "  [FAIL] $SC missing. The sidecar write path is not firing --"
    echo "         investigate now, before committing ${BUDGET_H}h to a run."
fi

echo
echo "=== Eval consumed it, and are the delta CIs present? ==="
RPT="$OUT_DIR/eval_permute_report.json"
if [ -s "$RPT" ]; then
    python3 - "$RPT" <<'PY'
import sys, json
r = json.load(open(sys.argv[1]))
s = r.get('arms', {}).get('sidecar', {})
print("  sidecar arm : %s  (%d sweep rungs)" % (s.get('status'), len(s.get('xi_sweep', []))))
if s.get('xi_spread') is not None:
    print("  xi_spread   : %.5f" % s['xi_spread'])

pr = r.get('arms', {}).get('stratify_decision', {}).get('per_region', {})
if not pr:
    print("  [WARN] no per_region block found.")
else:
    print("  %-10s %10s %12s %10s %14s" % ("region", "n_bulk", "delta", "margin", "verdict"))
    for R, d in pr.items():
        if d.get('status') != 'ok':
            print("  %-10s %10s %12s %10s %14s"
                  % (R, d.get('n_bulk', '-'), '-', '-', d.get('status')))
            continue
        m = d.get('delta_ci_margin')
        print("  %-10s %10d %12.4f %10s %14s"
              % (R, d['n_bulk'], d['delta_vs_trans'],
                 ("%.4f" % m) if m is not None else "-",
                 d.get('delta_ci_verdict', '-')))
    if all('delta_ci_verdict' not in d for d in pr.values()):
        print("  [WARN] delta CI fields absent -- is the merged Task A on this checkout?")
PY
else
    echo "  [FAIL] $RPT missing."
fi

echo
echo "=================================================================="
echo " Probe complete. If the numbers above look right, launch the real"
echo " run reusing the master this probe just built:"
echo
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N} \\"
echo "     /usr/bin/time -v ./pipelinePermute.sh -d $D \\"
echo "       --start-stage permute --permutations <N from above> \\"
echo "       >pipelinePermute.sh.$D.out 2>&1 &"
echo
echo " --start-stage permute skips the cis map and assemble, which this"
echo " probe already completed."
echo "=================================================================="
