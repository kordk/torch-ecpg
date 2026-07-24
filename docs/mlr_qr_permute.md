# `docs/mlr_qr_permute.md` — update blocks for the enrichment-verdict / QC-report cycle

Apply against `origin/dev` @ `e3e00d3` (v2.0.0b2.dev52). Each block below is a verbatim
FIND from the current file and its REPLACE. Blocks 1–5 are replacements; block 6 is an
addition. Verify each FIND appears exactly once before editing.

---

## Block 1 — Header status note (line 3)

**FIND** — the entire status blockquote beginning `> **Status: under active development.**` and ending `See [§5 Implementation status](#5-implementation-status).`

**REPLACE WITH:**

```markdown
> **Status: under active development — trans and cis both now calibrated on GTP.** The **Phase-1 trans pipeline is complete** and `qr_permute` produces real per-pair p-values on a finalized, seeded, provenance-stamped output schema. The Phase-3a diagnostics run on real data through the **canonical 7-way region taxonomy** (`assignRegionToEcpg_parquet.py`: `TRANS`, `DISTAL5`, `CIS5`, `PROMOTER`, `GENEBODY`, `CIS3`, `DISTAL3`). **The cis-window enrichment run is complete on full GTP and returned `single_global_null_adequate` with `divergent_regions = []`** (§5): the corrected `--cis-enrich` map produced 16,526,333 cis pairs, the assembled master carried 17,525,137 pairs, and the near-gene family cleared the coverage floor by ~11,700×. Every one of the seven regions calibrates against the analytic reference; the largest near-gene departure from `TRANS` is **3.16e-05** against a tolerance of 0.5, and the verdict is invariant from 0.5 down to 1e-4. **Consequence: on GTP a single global null is adequate and the Phase-2 per-gene cis Beta is not required.** Two qualifications matter. First, **the verdict is per-dataset** — MESA and the oncology cohort each carry their own confounding structure and must be run and read independently; nothing here is inherited. Second, **this validates the bulk, not the tail**: at B=10 the permutation p cannot resolve below ~2.6e-08, so the extreme tail is *unmeasured* rather than confirmed, in either direction (§8). Separately, the bulk comparison shows the analytic p is systematically **anti-conservative by about 0.6%** (permutation p ≥ analytic p for 99.56% of bulk pairs, median ratio 1.0063) — small, but directional rather than noise. `perm_mt_p` is still **not wired into the downstream FDR pipeline** (Phase 4). Do **not** use `perm_mt_p` for inference until this notice is updated. See [§5 Implementation status](#5-implementation-status).
```

---

## Block 2 — §5, Phase 2 bullet

**FIND:**

```markdown
- **Phase 2 — cis Beta-approximation** *(pending)*. Per-gene min-p Beta null for the cis stratum (§3.4), fit in the t-domain, reusing Phase 1's residualize/refit primitives and the shared cis mask as each gene's local test set. Until it lands, cis pairs are scored against the trans-global null. **Gated on the cis-window enrichment verdict** (§5, §8): the reservoir was too sparse near genes to decide, so the near-gene calibration is read from the enrichment run, and the Beta machinery is built only if that run shows the near-gene null diverges from trans (the per-chromosome-structural alternative is already ruled out by the distal strata calibrating to `TRANS`).
```

**REPLACE WITH:**

```markdown
- **Phase 2 — cis Beta-approximation** *(not required for GTP; conditional per dataset)*. Per-gene min-p Beta null for the cis stratum (§3.4), fit in the t-domain, reusing Phase 1's residualize/refit primitives and the shared cis mask as each gene's local test set. It was gated on the cis-window enrichment verdict, and **that gate has now resolved for GTP in the negative**: the enrichment run returned `single_global_null_adequate`, so the near-gene regions are adequately served by the trans-global null and the Beta machinery is unnecessary for this cohort. It remains conditionally pending for **MESA** and the **oncology cohort**, each of which gets its own enrichment run and its own verdict; the machinery is built only if one of them shows a near-gene stratum diverging from `TRANS`.
```

---

## Block 3 — §5, enrichment-run paragraph

**FIND:**

```markdown
  **Enrichment run (full GTP) — first attempt and correction.**
```

...through the end of that paragraph (`...the resulting 7-way verdict is the decision point for Phase 2.`)

**REPLACE WITH:**

```markdown
  **Enrichment run (full GTP) — complete.** The first `pipelinePermute.sh --cis-enrich` attempt validated the wrapper end-to-end but produced only **2,741** cis pairs from a requested ±1 Mb window and failed at the assemble step. Three independent defects, all since fixed: the windowed-map predicate's **`int8` overflow** and **negative-strand bound reversal** (§4), which together meant the "±1 Mb" window was really ~±64 bp on positive-strand genes and *empty* on negative-strand ones; and the assembly tool failing to promote `mt_id`/`gt_id` from the map parquet's **named index** (the map writes them as an index, and `mergeOutputs`' parquet→parquet path is a raw Arrow passthrough that preserves it).

  With those corrected the run completed cleanly. The map returned **16,526,333** cis pairs — a 6,029× recovery, and an independent confirmation of both window fixes on real data. `DISTAL5` and `DISTAL3` came out within **0.68%** of each other (7,674,941 vs 7,727,379), which is the direct signature of the negative-strand fix: a symmetric window applied to both strands produces a symmetric split, and pre-fix the `−` strand contributed nothing. Assembly produced a **17,525,137**-pair master (16,526,333 cis + 1,000,000 reservoir − 1,196 overlapping, with no `mt_t` disagreement across the overlap). Region counts: `trans` 931,682 · `distal5` 7,674,941 · `cis5` 549,615 · `promoter` 104,252 · `genebody` 5,177 · `cis3` 514,447 · `distal3` 7,727,379.

  **Verdict: `single_global_null_adequate`, `divergent_regions = []`.** The near-gene family (`CIS5`+`PROMOTER`+`GENEBODY`+`CIS3`) totalled **1,173,491** bulk pairs against `MIN_REGION_BULK_N` = 100 — roughly 11,700× headroom, so `insufficient_near_gene_coverage` is comprehensively off the table and every one of the seven regions is `ok`. All seven calibrate to the analytic reference at a bulk median `|log10(p_perm/p_ana)|` of ~0.0027, and the spread of `Δ vs TRANS` across strata spanning 4,884 to 7.3M bulk pairs is **3.2e-05**. The largest near-gene departure is **3.162e-05** (`PROMOTER`) against a tolerance of 0.5 — a margin of ~15,800×, and the verdict is unchanged for any tolerance down to 1e-4. Two caveats are recorded rather than buried. A margin that wide can equally indicate an *insensitive test* as a strong result, which is why the tolerance remains provisional and is reported alongside a sweep (§8). And the agreement is **directional**: permutation p exceeds analytic p for 99.56% of bulk pairs at a median ratio of 1.0063, so the analytic p is systematically anti-conservative by ~0.6% — negligible for the verdict, but a property of the null model rather than sampling noise.
```

---

## Block 4 — §8, "Partially validated" bullet

**FIND:**

```markdown
- **Partially validated (trans bulk).** On the GTP subset (reservoir calibration master, §5) the trans analytic null is well-calibrated in the bulk (permutation vs analytic agree to <1%, `lambda_trans` ≈ 1). Cis-window calibration and the full stratify-or-not decision are not yet settled (gene-anchored run pending, below), and `perm_mt_p` is not wired to FDR (Phase 4). Do not use `perm_mt_p` for inference yet.
```

**REPLACE WITH:**

```markdown
- **Validated in the bulk, across all seven regions, on GTP only.** The enrichment run (§5) validates the analytic null in the mostly-null bulk band for every region including near-gene, on one cohort. Three limits bound that claim. It is **per-dataset** — MESA and the oncology cohort are unvalidated until run. It is **bulk-only** — see the tail-resolution item below. And `perm_mt_p` is still not wired to FDR (Phase 4). Do not use `perm_mt_p` for inference yet.
- **Open — the tail is unmeasured at B = 10.** The empirical permutation p cannot resolve below roughly `1 / (n_perm × n_null_pairs)`, which for the GTP run is `1 / (10 × 3,793,007) ≈ 2.6e-08`. Any pair whose true p sits far past that limit receives a floored permutation p, so the ratio `p_perm / p_ana` there measures the distance from the analytic p to the floor and says nothing about whether the analytic p is right. The observed per-region tail ratios order exactly as signal density would predict (`TRANS` 1.4× median rising to `PROMOTER` 7.2×, with a 90th percentile of 6.5e8), which is consistent with a resolution artefact **and** with genuine signal, and the two are not separable at this permutation count. **No conclusion about the analytic tail should be drawn from the current run in either direction.** The open decision is whether to fund a substantially higher-B run to support a tail claim, or to scope the claim to the bulk and say so explicitly. Reported without a status badge in the QC report for this reason.
- **Open — an empty decade in the stored analytic p.** On the assembled GTP master, `P < 1e-8` and `P < 1e-9` both return 2,092 pairs, i.e. **zero pairs in `[1e-9, 1e-8)`**. A p-value distribution carrying real signal thins out smoothly as it deepens; an empty decade sitting above a populated one is not a shape a continuous distribution produces, and points at a storage floor, a clipping step, or an upstream filter. It is **not** a 32-bit float representation limit — float32 reaches ~1.18e-38, far below where the gap appears. The `analytic-p-precision` QC module measures it (smallest non-zero stored p, pile-up count, stored-versus-recomputed maximum ratio, per-decade counts). The cause is to be **traced, not assumed**.
- **Open — gene spans and the reachability of the `GENEBODY` label.** The region taxonomy can only assign a pair to `GENEBODY` when the annotated gene admits an interval beyond the promoter window: `+` strand requires `gt_start + 2500 < mt < gt_end` and `−` strand requires `gt_start < mt < gt_end − 2500`, so both need an annotated span above 2,500 bp. In the GTP gene annotation only **1,876 of 57,490 entries (3.3%)** exceed that, which is consistent with the observed `genebody` count of 5,177 (~0.135 CpG/gene, roughly 20× below `promoter` and backwards for a 450k array). Pairs a transcript-span annotation would have labelled `GENEBODY` are currently labelled `PROMOTER` or `CIS3`. **This cannot affect the calibration verdict** — the receiving regions are themselves calibrated, and moving pairs between calibrated strata cannot create divergence — but it does affect how the catalog should be described. The open question is whether the annotation source is the right one.
- **Open — `ks_trans_bulk_vs_uniform` is biased by construction.** The null-sanity arm compares the trans bulk analytic p against an untruncated uniform, but the bulk sample is truncated to `[bulk_lo, 1.0]`. A perfectly calibrated null simulated through the same code path yields `stat ≈ 0.0500, p = 0`, essentially identical to the observed value; rescaling the same sample to `[0, 1]` yields `p ≈ 0.56`. It is advisory and gates nothing, so no verdict is affected, and it is deliberately **not rendered** in the QC report. Fix: rescale to `(p − bulk_lo) / (1 − bulk_lo)` before the test.
```

---

## Block 5 — §8, "Partly resolved — stratify-or-not, cis window" bullet

**FIND:** the bullet beginning `- **Partly resolved — stratify-or-not, cis window.**`

**REPLACE WITH:**

```markdown
- **Resolved (GTP) — stratify-or-not.** The enrichment run settled it: `single_global_null_adequate`, `divergent_regions = []`, near-gene bulk 1,173,491 pairs against a floor of 100, largest near-gene `Δ vs TRANS` = 3.16e-05 against a tolerance of 0.5. A single global null carries for GTP and Phase 2 is unnecessary there. The reservoir-first run had already ruled out the *per-chromosome structural* branch (both distal strata calibrating to `TRANS`), leaving the near-gene window as the only open question; the enrichment run answered it. **The remaining work is per-dataset**: MESA and the oncology cohort each need their own reservoir + enrichment run and their own verdict, never inherited from GTP. Eval thresholds (`BULK_LO`, `TOLERANCE_MEDIAN_LOG10_RATIO_DIFF`, `MIN_REGION_BULK_N`) remain provisional; GTP now supplies evidence for re-setting the tolerance (the verdict is invariant from 0.5 to 1e-4), and MESA should inform the final choice.
```

---

## Block 6 — §7, new subsection (ADD after the existing usage code block)

```markdown
### 7.1 QC report

`tools/permute_qc_report.py` renders a single self-contained HTML QC report from the
evaluation output, modelled on FastQC: a navigation sidebar of modules, each with a
status badge, a stated **Purpose**, a figure and/or table, and an **Interpretation**
block. Figures are base64 data URIs and CSS is inline, so the file opens offline with no
sibling files and no script element.

```bash
python3 tools/permute_qc_report.py \
    --report          output_gtp/eval_permute_report.json \
    --perm-output     output_gtp/permutation_results.parquet --df 321 \
    --gene-annotation annot_gtp/G.bed6 \
    --meth-annotation annot_gtp/M.bed6 \
    --dataset gtp --out output_gtp/permute_qc_report.html
```

Only `--report` is required. Modules whose inputs are absent render an `INFO` panel
naming every missing flag, and the report always generates.

Twelve modules, in render order: `run-provenance`, `region-composition`,
`bulk-calibration`, `calibration-direction`, `stratification`, `verdict-robustness`,
`permutation-resolution`, `tail-behaviour`, `analytic-p-precision`, `cis-pair-density`,
`gene-span-distribution`, `tss-distance-by-region`. **`permutation-resolution` precedes
`tail-behaviour` by design** — tail ratios are bounded below by the resolution floor and
are uninterpretable without it.

The report is a **renderer, not an authority**: every calibration statistic and the
verdict come from `eval_permute.py`, and the only quantities computed here are those
absent from the report JSON (the resolution floor, the tolerance sweep, and the
correctness diagnostics that read the parquet or the annotations directly). Region
labels are read from the canonical `region` column and never re-derived; annotation
parsing and every taxonomy boundary are imported from `assignRegionToEcpg_parquet.py`.

Two modules are deliberately unbadged. **Genomic inflation** appears only as a column in
the stratification table: λ measures inflation against a mostly-null expectation that
near-gene eQTM pairs violate by construction, so its regional gradient largely restates
where signal lives rather than measuring calibration quality (§5, 3a). **Tail behaviour**
carries an `INFO` badge under all conditions because a large tail ratio is confounded
with both the permutation resolution floor and signal density, and no calibrated
threshold separates those causes at present permutation counts (§8).
```
