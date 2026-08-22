#!/usr/bin/env python3
import importlib.metadata
import subprocess
import datetime
import json
import hashlib
from collections import Counter
import argparse
import base64


import io

import logging
import os

import sys


import matplotlib.pyplot as plt
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq

import upsetplot
from matplotlib_venn import venn2
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from permute_qc_report import QCModule, fig_to_base64, render_table, render_html  # noqa: E402


def resolve_kennedy_columns(columns):
    required = {'cpg': 'CpG.probe', 'probe': 'exp.Probe', 'pval': 'p.val'}
    missing = [name for name in required.values() if name not in columns]
    if missing:
        raise ValueError(f"Missing required Kennedy columns: {missing}. Observed columns: {list(columns)}")

    resolved = {
        'cpg': 'CpG.probe',
        'probe': 'exp.Probe',
        'pval': 'p.val',
        'status': 'status' if 'status' in columns else None,
        'distance': columns[4] if len(columns) > 4 else None,
        'in_dist': 'in_dist' if 'in_dist' in columns else None,
        'tstat': 'T.stat' if 'T.stat' in columns else None,
        'beta': 'beta' if 'beta' in columns else None,
        'beta_sd': 'beta.sd' if 'beta.sd' in columns else None,
        'gene': 'annot.gene' if 'annot.gene' in columns else None,
        'probe_chrom': 'exp.probe.chrm' if 'exp.probe.chrm' in columns else None,
        'probe_start': 'exp.probe.start' if 'exp.probe.start' in columns else None,
        'probe_stop': 'exp.probe.stop' if 'exp.probe.stop' in columns else None,
        'probe_strand': 'exp.probe.strand' if 'exp.probe.strand' in columns else None,
        'other_gene': 'other_gene' if 'other_gene' in columns else None,
        'other_gene_distance': 'distance.other.gene' if 'distance.other.gene' in columns else None
    }
    return resolved


def resolve_tecpg_pvalue_column(columns, override=None):
    if override is not None:
        if override not in columns:
            raise ValueError(f"Override precise_mt_p column '{override}' not found in tecpg catalog. "
                             f"Observed columns: {list(columns)}")
        if override != 'precise_mt_p':
            logging.warning(
                f"WARNING: Using non-precise p-value column '{override}'. "
                f"Non-precise p columns are float32-saturated below roughly 1e-7. "
                f"Results below that are invalid."
            )
        return override

    if 'precise_mt_p' not in columns:
        raise ValueError(f"Missing precise_mt_p in tecpg catalog. Observed columns: {list(columns)}")
    return 'precise_mt_p'


# --- C4: region-composition crosswalk to Kennedy's four categories -----------
# docs/annotation.md sec 6.2: tecpg's seven UPPERCASE region labels roll up to
# Kennedy's cis / gene body / distal / trans without remainder. Labels are
# UPPERCASE as emitted by assignRegionToEcpg_parquet.py; a stale lowercase
# label (the pre-C4 defect) or a NULL region must fall to 'unlabeled', never be
# silently dropped and never be miscounted as a real category.
KENNEDY_ROLLUP = {
    'PROMOTER': 'cis', 'CIS5': 'cis', 'CIS3': 'cis',
    'GENEBODY': 'gene body',
    'DISTAL5': 'distal', 'DISTAL3': 'distal',
    'TRANS': 'trans',
}
KENNEDY_CATEGORIES = ('cis', 'gene body', 'distal', 'trans', 'unlabeled')


def rollup_region_to_kennedy(regions):
    """Roll tecpg region labels up to Kennedy's four categories (annotation.md
    sec 6.2). Returns a dict over KENNEDY_CATEGORIES. NULL/NaN and any label
    outside the seven-label UPPERCASE vocabulary fall to 'unlabeled'. The count
    is conserved: sum(result.values()) == len(regions) for any input."""
    import math
    counts = {c: 0 for c in KENNEDY_CATEGORIES}
    for r in regions:
        if r is None or (isinstance(r, float) and math.isnan(r)):
            counts['unlabeled'] += 1
        else:
            counts[KENNEDY_ROLLUP.get(r, 'unlabeled')] += 1
    return counts


def resolve_thresholds(p_thresh, kennedy_thresh, tecpg_thresh):
    if p_thresh is not None and kennedy_thresh is not None:
        raise ValueError("Cannot provide both --p-thresh and --kennedy-thresh.")

    if p_thresh is not None:
        logging.warning("DeprecationWarning: --p-thresh is deprecated. Please use --kennedy-thresh instead.")
        final_kennedy = p_thresh
    elif kennedy_thresh is not None:
        final_kennedy = kennedy_thresh
    else:
        final_kennedy = 1e-5

    final_tecpg = tecpg_thresh if tecpg_thresh is not None else 1e-5

    return final_kennedy, final_tecpg


def load_kennedy(path, sep):
    df = pd.read_csv(path, sep=sep)
    return df


def stream_catalog_and_match(args, schema, tecpg_p_col, kennedy_pairs,
                             thresholds, kennedy_cpgs, kennedy_probes):
    """
    Stream catalog in batches.
    Return (df_matched, tecpg_threshold_counts, distinct_mt, distinct_gt, region_counter, cat_profile_metrics)
    """
    import pyarrow.parquet as pq

    desired_cols = [
        'mt_id', 'gt_id', tecpg_p_col, 'mt_est', 'mt_t',
        'region', 'fdr_est', 'mt_chrom', 'gt_chrom', 'mt_h_max'
    ]
    seen = set()
    cols_to_load = []
    for c in desired_cols:
        if c not in seen and c in schema:
            seen.add(c)
            cols_to_load.append(c)

    parquet_file = pq.ParquetFile(args.tecpg)

    distinct_mt = set()
    distinct_gt = set()
    region_counter = Counter()
    tecpg_threshold_counts = {t: 0 for t in thresholds}

    matched_chunks = []

    cat_profile_metrics = {
        'row_count': 0,
        'row_group_count': parquet_file.num_row_groups,
        'precise_mt_p_decades': {1e-5: 0, 1e-6: 0, 1e-7: 0, 1e-8: 0, 1e-9: 0, 1e-10: 0, 1e-11: 0},
        'precise_mt_p_min': float('inf'),
        'precise_mt_p_max': float('-inf'),
        'mt_chroms': set(),
        'gt_chroms': set(),
        'chrom_pairs': set()
    }

    # For streaming hash-join we need kennedy_pairs to be a fast lookup of (mt_id, gt_id)
    # wait, kennedy_pairs is passed in

    for batch in parquet_file.iter_batches(batch_size=args.batch_size, columns=cols_to_load):
        df_chunk = batch.to_pandas()
        if df_chunk.index.names != [None]:
            df_chunk = df_chunk.reset_index()

        cat_profile_metrics['row_count'] += int(len(df_chunk))

        # Profile specific to catalog
        if tecpg_p_col in df_chunk.columns:
            p_vals = df_chunk[tecpg_p_col].dropna()
            if len(p_vals) > 0:
                cat_profile_metrics['precise_mt_p_min'] = min(cat_profile_metrics['precise_mt_p_min'], p_vals.min())
                cat_profile_metrics['precise_mt_p_max'] = max(cat_profile_metrics['precise_mt_p_max'], p_vals.max())
                for t in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]:
                    cat_profile_metrics['precise_mt_p_decades'][t] += int((p_vals < t).sum())
                    if t in tecpg_threshold_counts:
                        tecpg_threshold_counts[t] += int((p_vals < t).sum())

        if 'region' in df_chunk.columns:
            region_counter.update(df_chunk['region'].fillna('None').tolist())

        distinct_mt.update(df_chunk['mt_id'].dropna().unique())
        distinct_gt.update(df_chunk['gt_id'].dropna().unique())

        if 'mt_chrom' in df_chunk.columns and 'gt_chrom' in df_chunk.columns:
            # Drop rows missing chroms
            chrom_df = df_chunk[['mt_chrom', 'gt_chrom']].dropna()
            if not chrom_df.empty:
                cat_profile_metrics['mt_chroms'].update(chrom_df['mt_chrom'].unique())
                cat_profile_metrics['gt_chroms'].update(chrom_df['gt_chrom'].unique())
                cat_profile_metrics['chrom_pairs'].update(zip(chrom_df['mt_chrom'], chrom_df['gt_chrom']))

        # Match
        # To avoid DataFrame apply overhead, use a MultiIndex or zip
        mask = pd.Series(zip(df_chunk['mt_id'], df_chunk['gt_id'])).isin(kennedy_pairs).values
        if mask.any():
            matched_chunks.append(df_chunk[mask])

    if matched_chunks:
        df_matched = pd.concat(matched_chunks, ignore_index=True)
    else:
        df_matched = pd.DataFrame(columns=cols_to_load)

    # Convert sets to exact count lists (or just keep sets since distinct mt/gt size is requested)
    return df_matched, tecpg_threshold_counts, distinct_mt, distinct_gt, region_counter, cat_profile_metrics


def compute_eligibility(distinct_mt, distinct_gt, kennedy_df, cols):
    df = kennedy_df.copy()
    df['cpg_in_tecpg_universe'] = df[cols['cpg']].isin(set(distinct_mt))
    df['probe_in_tecpg_universe'] = df[cols['probe']].isin(set(distinct_gt))
    df['eligible'] = df['cpg_in_tecpg_universe'] & df['probe_in_tecpg_universe']
    return df


def compute_overlap_rates(df_matched, kennedy_df, cols, kennedy_thresh, tecpg_thresh, tecpg_p_col,
                          return_sets=False, kennedy_cpgs=None, kennedy_probes=None, tecpg_threshold_counts=None):
    K_tk_mask = kennedy_df[cols['pval']] < kennedy_thresh
    K_tk_df = kennedy_df[K_tk_mask]
    K_tk = set(zip(K_tk_df[cols['cpg']], K_tk_df[cols['probe']]))

    K_tk_E_df = kennedy_df[K_tk_mask & kennedy_df['eligible']]
    K_tk_E = set(zip(K_tk_E_df[cols['cpg']], K_tk_E_df[cols['probe']]))

    T_tt_df = df_matched[df_matched[tecpg_p_col] < tecpg_thresh]
    T_tt_matched = set(zip(T_tt_df['mt_id'], T_tt_df['gt_id']))

    recovery_num = len(K_tk_E.intersection(T_tt_matched))
    recovery_denom = len(K_tk_E)
    recovery = recovery_num / recovery_denom if recovery_denom > 0 else 0.0

    confirmation_num = len(T_tt_matched.intersection(K_tk))
    # confirmation_denom MUST be from the total streamed catalog count, not just matched
    if tecpg_threshold_counts and tecpg_thresh in tecpg_threshold_counts:
        confirmation_denom = tecpg_threshold_counts[tecpg_thresh]
    else:
        # Fallback for tests passing full catalog_df in T_tt_df
        confirmation_denom = len(T_tt_matched)

    confirmation_raw = confirmation_num / confirmation_denom if confirmation_denom > 0 else 0.0

    if kennedy_cpgs is None:
        kennedy_cpgs = set(kennedy_df[cols['cpg']].dropna())
    if kennedy_probes is None:
        kennedy_probes = set(kennedy_df[cols['probe']].dropna())

    # T_tt_kennedy_testable is exactly T_tt_matched because df_matched is filtered by kennedy_pairs
    T_tt_kennedy_testable = {(c, p) for c, p in T_tt_matched if c in kennedy_cpgs and p in kennedy_probes}
    confirmation_testable_num = len(T_tt_kennedy_testable.intersection(K_tk))
    confirmation_testable_denom = len(T_tt_kennedy_testable)
    testable_val = confirmation_testable_num / confirmation_testable_denom if confirmation_testable_denom > 0 else 0.0
    confirmation_kennedy_testable = testable_val

    # Jaccard denominator: |A U B| = |A| + |B| - |A int B|
    union_len = confirmation_denom + len(K_tk) - confirmation_num
    jaccard = confirmation_num / union_len if union_len > 0 else 0.0

    res = {
        'recovery': recovery,
        'confirmation_raw': confirmation_raw,
        'confirmation_kennedy_testable': confirmation_kennedy_testable,
        'jaccard': jaccard,
        'counts': {
            'recovery_num': int(recovery_num),
            'recovery_denom': int(recovery_denom),
            'confirmation_num': int(confirmation_num),
            'confirmation_denom': int(confirmation_denom),
            'confirmation_testable_num': int(confirmation_testable_num),
            'confirmation_testable_denom': int(confirmation_testable_denom),
            'k_tk': int(len(K_tk)),
            'union': int(union_len),
        },
    }
    if return_sets:
        res.update({
            'T_tt': T_tt_matched,  # only returns matched sets now!
            'K_tk': K_tk,
            'K_tk_E': K_tk_E,
            'K_tk_E_df': K_tk_E_df,
            'T_tt_df': T_tt_df,
            'kennedy_cpgs': kennedy_cpgs,
            'kennedy_probes': kennedy_probes,
            'confirmation_denom': confirmation_denom
        })
    return res


def export_pair_lists(outdir, catalog_df, kennedy_df, cols, diag_results, tecpg_p_col):
    T_tt = diag_results['T_tt']
    K_tk = diag_results['K_tk']
    kennedy_cpgs = diag_results['kennedy_cpgs']
    kennedy_probes = diag_results['kennedy_probes']

    overlap = T_tt.intersection(K_tk)
    tecpg_only = T_tt - K_tk
    kennedy_only = K_tk - T_tt

    conc_mt, conc_gt = zip(*overlap) if overlap else ([], [])
    df_conc = pd.DataFrame({'mt_id': conc_mt, 'gt_id': conc_gt})

    t_only_mt, t_only_gt = zip(*tecpg_only) if tecpg_only else ([], [])
    df_t_only = pd.DataFrame({'mt_id': t_only_mt, 'gt_id': t_only_gt})

    k_only_mt, k_only_gt = zip(*kennedy_only) if kennedy_only else ([], [])
    df_k_only = pd.DataFrame({'mt_id': k_only_mt, 'gt_id': k_only_gt})

    catalog_cpgs = set(catalog_df['mt_id'].dropna())
    catalog_probes = set(catalog_df['gt_id'].dropna())

    def annotate_reasons(df, reason='concordant'):
        if len(df) == 0:
            df['non_overlap_reason'] = []
            df['cpg_in_tecpg_universe'] = []
            df['probe_in_tecpg_universe'] = []
            df['cpg_in_kennedy_file'] = []
            df['probe_in_kennedy_file'] = []
            return df

        cpg_in_k = df['mt_id'].isin(kennedy_cpgs)
        probe_in_k = df['gt_id'].isin(kennedy_probes)
        df['cpg_in_kennedy_file'] = cpg_in_k
        df['probe_in_kennedy_file'] = probe_in_k

        df['cpg_in_tecpg_universe'] = df['mt_id'].isin(catalog_cpgs)
        df['probe_in_tecpg_universe'] = df['gt_id'].isin(catalog_probes)

        if reason == 'tecpg_only':
            reasons = []
            for c, p in zip(cpg_in_k, probe_in_k):
                if c and p:
                    reasons.append('kennedy_tested_and_missed')
                else:
                    reasons.append('kennedy_universe_unknown')
            df['non_overlap_reason'] = reasons
        else:
            df['non_overlap_reason'] = reason

        return df

    df_conc = annotate_reasons(df_conc, 'concordant')
    df_t_only = annotate_reasons(df_t_only, 'tecpg_only')

    if len(df_k_only) > 0:
        k_merge = pd.merge(
            df_k_only, kennedy_df, left_on=['mt_id', 'gt_id'],
            right_on=[cols['cpg'], cols['probe']], how='left'
        )
        # catalog_df here is df_matched. So checking if a Kennedy pair is in catalog
        # is just checking if it is in catalog_df.
        matched_pairs = set(zip(catalog_df['mt_id'], catalog_df['gt_id']))
        in_catalog = [p in matched_pairs for p in zip(k_merge['mt_id'], k_merge['gt_id'])]

        reasons = []
        for i, row in k_merge.iterrows():
            if in_catalog[i]:
                reasons.append('tested_and_missed')
            elif not row['cpg_in_tecpg_universe']:
                reasons.append('ineligible_cpg')
            elif not row['probe_in_tecpg_universe']:
                reasons.append('ineligible_probe')
            else:
                reasons.append('tested_and_missed')
        k_merge['non_overlap_reason'] = reasons
        df_k_only = k_merge[['mt_id', 'gt_id', 'non_overlap_reason',
                             'cpg_in_tecpg_universe', 'probe_in_tecpg_universe']]
        df_k_only = df_k_only.copy()
        df_k_only['cpg_in_kennedy_file'] = df_k_only['mt_id'].isin(kennedy_cpgs)
        df_k_only['probe_in_kennedy_file'] = df_k_only['gt_id'].isin(kennedy_probes)
    else:
        df_k_only['non_overlap_reason'] = []
        df_k_only['cpg_in_tecpg_universe'] = []
        df_k_only['probe_in_tecpg_universe'] = []
        df_k_only['cpg_in_kennedy_file'] = []
        df_k_only['probe_in_kennedy_file'] = []

    tecpg_cols = ['mt_id', 'gt_id', tecpg_p_col, 'mt_est', 'mt_t', 'region', 'fdr_est']
    tecpg_cols_to_merge = [c for c in tecpg_cols if c in catalog_df.columns]

    k_cols_base = [
        cols['pval'], cols['tstat'], cols['beta'],
        cols['beta_sd'], cols['status'], cols['distance']
    ]
    k_cols_valid = [c for c in k_cols_base if c and c in kennedy_df.columns]

    def merge_all(df):
        if len(df) == 0:
            empty_cols = tecpg_cols_to_merge[2:] + k_cols_valid
            for c in empty_cols:
                if c not in df.columns:
                    df[c] = []
            return df
        df = pd.merge(df, catalog_df[tecpg_cols_to_merge], on=['mt_id', 'gt_id'], how='left')
        df = pd.merge(
            df, kennedy_df[[cols['cpg'], cols['probe']] + k_cols_valid],
            left_on=['mt_id', 'gt_id'], right_on=[cols['cpg'], cols['probe']], how='left'
        )
        if cols['cpg'] in df.columns and cols['cpg'] != 'mt_id':
            df = df.drop(columns=[cols['cpg']])
        if cols['probe'] in df.columns and cols['probe'] != 'gt_id':
            df = df.drop(columns=[cols['probe']])
        return df

    df_conc = merge_all(df_conc)
    df_t_only = merge_all(df_t_only)
    df_k_only = merge_all(df_k_only)

    df_conc.to_csv(os.path.join(outdir, 'pairs_concordant.tsv'), sep='\t', index=False)
    df_t_only.to_csv(os.path.join(outdir, 'pairs_tecpg_only.tsv'), sep='\t', index=False)
    df_k_only.to_csv(os.path.join(outdir, 'pairs_kennedy_only.tsv'), sep='\t', index=False)


def add_provenance_to_summary(summary_text, prov):
    header = []
    header.append("========================================")
    header.append("PROVENANCE")
    header.append("========================================")
    header.append(f"tecpg version: {prov['tecpg_version']}")
    header.append(f"git SHA: {prov['git_sha']}")
    header.append(f"timestamp: {prov['timestamp_utc']}")
    header.append(f"argv: {' '.join(prov['argv'])}")
    header.append(f"batch_size: {prov['batch_size']}")
    header.append("Inputs:")
    header.append(
        f"  tecpg catalog: {prov['inputs']['tecpg_catalog']['path']} "
        f"(SHA256: {prov['inputs']['tecpg_catalog']['sha256']})")
    header.append(f"  kennedy: {prov['inputs']['kennedy']['path']} "
                  f"(SHA256: {prov['inputs']['kennedy']['sha256']})")
    header.append("Thresholds:")
    for k, v in prov['thresholds'].items():
        header.append(f"  {k}: {v}")
    header.append("========================================\n")
    return "\n".join(header) + summary_text


def build_summary_text(
    args, num_merged, pearson_r_beta, spearman_r_beta, r2_beta, beta_col,
    pearson_r_t, spearman_r_t, r2_t, tstat_col,
    grid_results, diag_results, thresholds, cols, kennedy_df
):

    summary = f"""Benchmark Summary
=================
Input tecpg file: {args.tecpg}
Input Kennedy file: {args.kennedy}
Kennedy p-value threshold: {args.kennedy_thresh}
tecpg p-value threshold: {args.tecpg_thresh}

Mapping & Merging
-----------------
Overlapping Pairs Mapped: {num_merged}
"""
    if 'status' in cols and cols['status']:
        statuses = kennedy_df[cols['status']].unique()
        summary += "\nEligibility by Status:\n"
        for s in statuses:
            sub = kennedy_df[kennedy_df[cols['status']] == s]
            el = sub['eligible'].sum()
            summary += f"  {s}: {el} / {len(sub)} eligible\n"
    else:
        summary += f"\nOverall Eligibility: {kennedy_df['eligible'].sum()} / {len(kennedy_df)}\n"

    summary += f"""
Comparison A: Statistical Concordance
-------------------------------------
Effect Size (mt_est vs {beta_col}):
  Pearson r:  {pearson_r_beta:.4f}
  Spearman rho: {spearman_r_beta:.4f}
  R^2:        {r2_beta:.4f}

Test Statistic (mt_t vs {tstat_col}):
  Pearson r:  {pearson_r_t:.4f}
  Spearman rho: {spearman_r_t:.4f}
  R^2:        {r2_t:.4f}

Comparison B: Hit Overlap (Directional Rates)
---------------------------------------------
Note: Confirmation denominators are LOWER BOUNDS (except kennedy_testable).
The Kennedy file is not a full universe, so we cannot know how many of our
hits they actually tested and found insignificant vs never tested.

--- Recovery Matrix (tecpg_thresh \\ kennedy_thresh) ---
"""
    header = "          " + "".join([f"{t:10.1e}" for t in thresholds])
    summary += header + "\n"
    for tt in thresholds:
        row_str = f"{tt:10.1e}"
        for tk in thresholds:
            res = grid_results[(tk, tt)]
            row_str += f"{res['recovery']:10.3f}"
        summary += row_str + "\n"

    summary += "\n--- Confirmation (Raw Lower Bound) Matrix ---\n"
    summary += header + "\n"
    for tt in thresholds:
        row_str = f"{tt:10.1e}"
        for tk in thresholds:
            res = grid_results[(tk, tt)]
            row_str += f"{res['confirmation_raw']:10.3f}"
        summary += row_str + "\n"

    summary += f"""
--- Diagonal (Like-for-like at {args.tecpg_thresh}) ---
  Recovery:                      {diag_results['recovery']:.4f}
  Confirmation (Raw Lower Bound): {diag_results['confirmation_raw']:.4f}
  Confirmation (Kennedy Testable):{diag_results['confirmation_kennedy_testable']:.4f}
  Jaccard Index:                 {diag_results['jaccard']:.4f}

Outputs
-------
Summary saved to: benchmark_summary.txt
Plots saved to: concordance_scatter.png, overlap_venn_diagonal.png
Pair lists saved to: pairs_*.tsv
"""
    if args.upset:
        summary += "UpSet plot saved to: overlap_upset_diagonal.png\n"

    return summary


def _sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_kennedy(path, df, sep, cols, thresholds):
    profile = {}
    profile['path'] = path
    profile['sha256'] = _sha256sum(path)
    profile['byte_size'] = os.path.getsize(path)
    profile['delimiter'] = sep
    profile['column_list'] = list(df.columns)

    profile['row_count'] = int(len(df))
    cpg_col = cols['cpg']
    query_col = cols['probe']

    # Must use dropna to match valid pairs
    valid = df.dropna(subset=[cpg_col, query_col])
    profile['unique_pair_count'] = int(len(set(zip(valid[cpg_col], valid[query_col]))))
    assert profile['row_count'] == profile['unique_pair_count'], \
        "Kennedy row count does not equal unique pair count!"

    decades = {}
    if cols['pval'] in df.columns:
        pvals = df[cols['pval']].dropna()
        for t in thresholds:
            decades[t] = {
                'n': int((pvals < t).sum()),
                'trans_n': 0,  # To be filled below
                'trans_fraction': 0.0
            }

    if cols['status'] in df.columns:
        status_counts = {k: int(v) for k, v in df[cols['status']].fillna('None').value_counts().to_dict().items()}
        profile['status_composition'] = {
            'counts': status_counts,
            'fractions': {k: v / profile['row_count'] for k, v in status_counts.items()},
            'sum': sum(status_counts.values())
        }
        assert profile['status_composition']['sum'] == profile['row_count'], "Kennedy status sum != row count"

        # fill trans_n
        if cols['pval'] in df.columns:
            for t in thresholds:
                t_mask = df[cols['pval']] < t
                trans_mask = df[cols['status']] == 'TRANS'
                trans_n = (t_mask & trans_mask).sum()
                decades[t]['trans_n'] = int(trans_n)
                decades[t]['trans_fraction'] = trans_n / decades[t]['n'] if decades[t]['n'] > 0 else 0.0

    profile['p_column_decade_table'] = decades

    na_counts = {}
    for c in df.columns:
        n_na = df[c].isna().sum()
        na_counts[c] = {'count': int(n_na), 'conditionality': None}

        if c == cols.get('distance') and cols['status'] in df.columns:
            # check if NA is conditional on TRANS
            if n_na > 0 and n_na == (df[cols['status']] == 'TRANS').sum():
                # check exact set match
                na_set = set(df[df[c].isna()].index)
                trans_set = set(df[df[cols['status']] == 'TRANS'].index)
                if na_set == trans_set:
                    na_counts[c]['conditionality'] = 'TRANS'
        elif c == cols.get('in_dist') and cols['status'] in df.columns:
            # check if non-NA is conditional on IN
            n_non_na = df[c].notna().sum()
            if n_non_na > 0 and n_non_na == (df[cols['status']] == 'IN').sum():
                non_na_set = set(df[df[c].notna()].index)
                in_set = set(df[df[cols['status']] == 'IN'].index)
                if non_na_set == in_set:
                    na_counts[c]['conditionality'] = 'IN'

    profile['na_counts'] = na_counts

    if cols['distance'] in df.columns:
        d = df[cols['distance']].dropna()
        if len(d) > 0:
            profile['distance'] = {
                'min': float(d.min()),
                'max': float(d.max()),
                'n_negative': int((d < 0).sum()),
                'n_positive': int((d > 0).sum()),
                'n_na': int(df[cols['distance']].isna().sum()),
                'sign_split': {}
            }
            if cols['status'] in df.columns:
                for stat in df[cols['status']].dropna().unique():
                    mask = df[cols['status']] == stat
                    d_stat = df.loc[mask, cols['distance']].dropna()
                    profile['distance']['sign_split'][stat] = {
                        'negative': int((d_stat < 0).sum()),
                        'positive': int((d_stat > 0).sum())
                    }

    # Chromosome column formatting
    if cols['probe_chrom'] in df.columns:
        chroms = df[cols['probe_chrom']].dropna().unique()
        profile['chrom_formatting'] = {}
        if len(chroms) <= 40:
            profile['chrom_formatting']['distinct_set'] = list(chroms)
        else:
            profile['chrom_formatting']['sample'] = list(chroms[:5])
            profile['chrom_formatting']['count'] = len(chroms)

        str_chroms = df[cols['probe_chrom']].dropna().astype(str)
        profile['chrom_formatting']['flags'] = {
            'float_like': bool(str_chroms.str.match(r'^[0-9]+\\.0$').any()),
            'pipe_delimited': bool(str_chroms.str.contains(r'\\|').any()),
            'nan_string': bool(str_chroms.str.lower().eq('nan').any())
        }

    return profile


def profile_catalog_post_stream(args, path, cat_metrics, schema, distinct_mt, distinct_gt, region_counter):
    profile = {}
    profile['path'] = path
    profile['sha256'] = _sha256sum(path)
    profile['byte_size'] = os.path.getsize(path)
    profile['row_count'] = cat_metrics['row_count']
    profile['row_group_count'] = int(cat_metrics['row_group_count'])
    profile['column_list'] = schema

    # Presence table
    presence = {}
    expected_cols = ['mt_id', 'gt_id', 'mt_est', 'mt_err', 'mt_t', 'mt_p',
                     'precise_mt_p', 'fdr_est', 'region', 'p_boot', 'mt_ig', 'ci_low', 'ci_high']
    for c in expected_cols:
        presence[c] = c in schema
    profile['presence_table'] = presence

    if presence['precise_mt_p']:
        profile['precise_mt_p'] = {
            'dtype': 'float64',  # PyArrow float64 by default usually
            'min': cat_metrics['precise_mt_p_min'],
            'max': cat_metrics['precise_mt_p_max'],
            'decades': cat_metrics['precise_mt_p_decades']
        }

    rc_sum = sum(region_counter.values())
    profile['region_composition'] = {
        'counts': dict(region_counter),
        'fractions': {k: v / rc_sum for k, v in region_counter.items()} if rc_sum > 0 else {},
        'sum': rc_sum
    }
    # Don't assert sum == row_count here, region could be null or there could be a schema mismatch in mock data.
    if 'region' in schema:
        assert rc_sum == profile['row_count'], \
            "Catalog region sum does not equal row count"

    profile['distinct_mt_id_count'] = len(distinct_mt)
    profile['distinct_gt_id_count'] = len(distinct_gt)

    profile['sample_mt_id'] = list(distinct_mt)[:5]
    profile['sample_gt_id'] = list(distinct_gt)[:5]

    # Check chromosome span if available
    profile['chrom_span_observation'] = None
    if 'mt_chrom' in schema and 'gt_chrom' in schema:
        expected_combos = len(cat_metrics['mt_chroms']) * len(cat_metrics['gt_chroms'])
        actual_combos = len(cat_metrics['chrom_pairs'])
        if actual_combos < expected_combos:
            profile['chrom_span_observation'] = (

                f"Catalog does not span all chromosome pairings "
                f"({actual_combos} observed vs {expected_combos} expected)."
            )

    return profile


def get_git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode().strip()
    except Exception:
        return "unknown"


def _resolve_tecpg_version() -> str:
    """Resolve the installed tecpg version string. Never raises."""
    try:
        v = importlib.metadata.version('tecpg')
        if isinstance(v, str) and v.strip():
            return v
    except Exception:
        pass
    try:
        import tecpg
        v = getattr(tecpg, '__version__', None)
        if isinstance(v, str) and v.strip():
            return v
    except Exception:
        pass
    return 'unknown'

def build_provenance(args, df_kennedy):
    prov = {
        'tecpg_version': _resolve_tecpg_version(),
        'git_sha': get_git_sha(),
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        'argv': sys.argv,
        'batch_size': args.batch_size,
        'thresholds': {
            'tecpg_thresh': args.tecpg_thresh,
            'kennedy_thresh': args.kennedy_thresh,
            'p_thresh': args.p_thresh
        },
        'inputs': {
            'tecpg_catalog': {
                'path': args.tecpg,
                'sha256': _sha256sum(args.tecpg)
            },
            'kennedy': {
                'path': args.kennedy,
                'sha256': _sha256sum(args.kennedy)
            }
        }
    }
    # Add confirmation_is_lower_bound for metrics
    return prov


def write_provenance_and_reports(args, num_merged, diag_results, grid_results, df_kennedy, cols, figs=None):
    if figs is None:
        figs = {}
    # Prepare JSON
    out_json = {
        'provenance': build_provenance(args, df_kennedy),
        'kennedy_profile': args.kennedy_profile_metrics,
        'catalog_profile': args.catalog_profile_metrics
    }

    if num_merged is not None and diag_results is not None and grid_results is not None:
        # Add confirmation lower bound caveat
        # Every confirmation metric in the JSON carries a sibling key: "confirmation_is_lower_bound": True
        for tk, tt in grid_results:
            grid_results[(tk, tt)]['confirmation_is_lower_bound'] = True
            grid_results[(tk, tt)]['confirmation_lower_bound_reason'] = \
                "Denominator includes all tecpg hits, not just those with kennedy coverage."
        diag_results['confirmation_is_lower_bound'] = True
        diag_results['confirmation_lower_bound_reason'] = \
            "Denominator includes all tecpg hits, not just those with kennedy coverage."

        # We need to drop large sets from diag_results before JSON serialization
        diag_json = {k: v for k, v in diag_results.items() if not isinstance(v, (set, pd.DataFrame))}
        grid_json = {}
        for (tk, tt), res in grid_results.items():
            grid_json[f"{tk}_{tt}"] = {k: v for k, v in res.items() if not isinstance(v, (set, pd.DataFrame))}

        out_json['results'] = {
            'diagonal': diag_json,
            'grid': grid_json,
            'num_merged': num_merged
        }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    with open(os.path.join(args.outdir, 'benchmark_metrics.json'), 'w') as f:
        json.dump(out_json, f, indent=2, cls=NpEncoder)

    # TSV long form
    if grid_results is not None:
        tsv_rows = []
        for (tk, tt), res in grid_results.items():
            for metric in ['recovery', 'confirmation_raw', 'confirmation_kennedy_testable', 'jaccard']:
                tsv_rows.append(f"{tt}	{tk}	all	{metric}	{res[metric]}")

        with open(os.path.join(args.outdir, 'benchmark_metrics.tsv'), 'w') as f:
            f.write("tecpg_thresh\tkennedy_thresh\tstratum\tmetric\tvalue\n")
            f.write("\n".join(tsv_rows) + "\n")

    if not args.no_html:
        modules = [
            build_provenance_module(build_provenance(args, df_kennedy)),
            build_reference_file_profile_module(args.kennedy_profile_metrics),
            build_catalog_profile_module(args.catalog_profile_metrics)
        ]

        if not args.characterize:
            modules.extend([
                build_eligibility_decomposition_module(df_kennedy, cols),
                build_recovery_grid_module(grid_results, [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]),
                build_confirmation_grid_module(grid_results, [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]),
                build_diagonal_summary_module(diag_results),
                build_trans_fraction_module(args),
                build_region_composition_module(args),
                build_concordance_module(args, figs)
            ])

        html_report = render_html("benchmark_kennedy", {}, modules)
        with open(os.path.join(args.outdir, 'benchmark_report.html'), 'w') as f:
            f.write(html_report)


def build_provenance_module(prov) -> QCModule:
    purpose = "Records the environment, configuration, and exact input files used."
    interpretation = 'What to check before trusting anything downstream: input SHA256s match the intended files; both thresholds are the intended ones; git SHA corresponds to a known commit. Note that `tecpg version: unknown` currently appears and should be fixed via importlib.metadata or the package __version__.'

    table_rows = [
        ["tecpg version", prov['tecpg_version']],
        ["git SHA", prov['git_sha']],
        ["Timestamp", prov['timestamp_utc']],
        ["tecpg Catalog Path", prov['inputs']['tecpg_catalog']['path']],
        ["tecpg Catalog SHA256", prov['inputs']['tecpg_catalog']['sha256']],
        ["Kennedy Path", prov['inputs']['kennedy']['path']],
        ["Kennedy SHA256", prov['inputs']['kennedy']['sha256']],
        ["Batch Size", str(prov['batch_size'])],
        ["Argv", " ".join(prov['argv'])],
        ["tecpg_thresh", str(prov['thresholds']['tecpg_thresh'])],
        ["kennedy_thresh", str(prov['thresholds']['kennedy_thresh'])],
    ]

    return QCModule(
        anchor="provenance", title="Provenance", status="INFO",
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(["Metric", "Value"], table_rows)
    )


def build_reference_file_profile_module(prof) -> QCModule:
    purpose = "Profiles the reference file structure, composition, and expected dimensions."
    interpretation = 'The Kennedy file is a SUGGESTIVE tier (p < 1e-5), not a hit list: roughly 88% of GTP rows are TRANS and at 1e-5 that stratum is close to what the test-space geometry alone would produce. Overlap statistics computed at 1e-5 against this file are therefore dominated by chance agreement in TRANS. The genome-wide significant tier is p < 1e-11 (2,466 GTP pairs), which is the comparable set. State that NA in `distance` is class-conditional (NA exactly for TRANS) and is not missing data.'

    status = "INFO"

    for k, v in prof['na_counts'].items():
        if v['conditionality'] is not None:
            # We detected conditionality, so it holds. Wait, if it didn't hold when it was supposed to?
            # We can't strictly know if it was 'supposed' to hold without hardcoding, so we just report it.
            pass

    rows = [
        ["Path", prof['path']],
        ["Rows", str(prof['row_count'])],
        ["Unique Pairs", str(prof['unique_pair_count'])],
        ["Columns", ", ".join(prof['column_list'])]
    ]

    return QCModule(
        anchor="reference-profile", title="Reference File Profile", status=status,
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(["Metric", "Value"], rows)
    )


def build_catalog_profile_module(prof) -> QCModule:
    purpose = "Profiles the tecpg catalog stream."
    interpretation = 'precise_mt_p is float64 and usable to ~1e-118; mt_p is float32-saturated with a block of exact zeros below roughly 1e-7 and must not be used for thresholding. If region shows GENEBODY far below the other near-gene classes, flag it as a known open question (gt_chromEnd is absent from the catalog schema, so gene-body assignment cannot be evaluated from a start coordinate alone) and state that the near-gene strata should be read with caution pending C4.'

    status = "INFO"
    if prof.get('chrom_span_observation'):
        status = "WARN"

    rows = [
        ["Rows", str(prof['row_count'])],
        ["Columns", ", ".join(prof['column_list'])],
        ["Distinct mt_id", str(prof['distinct_mt_id_count'])],
        ["Distinct gt_id", str(prof['distinct_gt_id_count'])]
    ]

    if prof.get('chrom_span_observation'):
        rows.append(["Chrom Span", prof['chrom_span_observation']])

    return QCModule(
        anchor="catalog-profile", title="Catalog Profile", status=status,
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(["Metric", "Value"], rows)
    )


def build_eligibility_decomposition_module(df_kennedy, cols) -> QCModule:
    purpose = "Decomposes the eligibility of Kennedy pairs in the tecpg universe."

    total = len(df_kennedy)
    eligible = df_kennedy['eligible'].sum()

    if total > 0 and eligible == total:
        interp_dynamic = ("Eligibility is 100%. Every non-overlap is a genuine method "
                          "disagreement rather than a coverage gap, which strengthens "
                          "interpretation of everything below.")
    else:
        interp_dynamic = ("Eligibility is below 100%. The recovery denominator is "
                          "conditioned on it and cross-cohort comparisons of recovery "
                          "are not like-for-like.")

    interpretation = ("This is the ceiling on recovery, and it must be read BEFORE "
                      "the recovery numbers. " + interp_dynamic)

    rows = [
        ["Total Kennedy Pairs", str(total)],
        ["CpG in tecpg", str(df_kennedy['cpg_in_tecpg_universe'].sum())],
        ["Probe in tecpg", str(df_kennedy['probe_in_tecpg_universe'].sum())],
        ["Pair in tecpg (Eligible)", str(eligible)]
    ]
    return QCModule(
        anchor="eligibility", title="Eligibility Decomposition", status="INFO",
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(["Metric", "Count"], rows)
    )


def build_recovery_grid_module(grid_results, thresholds) -> QCModule:
    purpose = "Displays the fraction of eligible Kennedy hits found in tecpg."
    interpretation = (
        "Read along the Kennedy axis: recovery should RISE as their threshold tightens, "
        "because their strongest hits should be the ones we recover best. A flat or "
        "falling profile in that direction would suggest we are matching their noise "
        "rather than their signal, and is the main thing to look for. Recovery falls as "
        "OUR threshold tightens, which is expected and not informative."
    )

    headers = ["tecpg \\ Kennedy"] + [f"{tk:g}" for tk in thresholds]
    rows = []
    for tt in thresholds:
        row = [f"{tt:g}"]
        for tk in thresholds:
            row.append(f"{grid_results[(tk, tt)]['recovery']:.3f}")
        rows.append(row)

    return QCModule(
        anchor="recovery-grid", title="Recovery Grid", status="INFO",
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(headers, rows)
    )


def build_confirmation_grid_module(grid_results, thresholds) -> QCModule:
    purpose = "Displays the fraction of tecpg hits found in Kennedy."
    interpretation = (
        "The raw denominator is a LOWER BOUND. Kennedy's tested universe (483,399 CpGs "
        "x 13,933 transcripts) is not recoverable from the supplement, which contains "
        "only CpGs that produced at least one suggestive hit. The kennedy_testable "
        "variant is the opposing bound and is biased UPWARD for the same reason. The "
        "two bracket the truth; neither is the answer."
    )

    headers = ["tecpg \\ Kennedy"] + [f"{tk:g}" for tk in thresholds]
    rows = []
    for tt in thresholds:
        row = [f"{tt:g}"]
        for tk in thresholds:
            row.append(f"{grid_results[(tk, tt)]['confirmation_raw']:.3f}")
        rows.append(row)

    return QCModule(
        anchor="confirmation-grid", title="Confirmation Grid", status="INFO",
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(headers, rows)
    )


def build_diagonal_summary_module(diag) -> QCModule:
    purpose = "Summary of matched p-value threshold overlap."
    interpretation = (
        "The like-for-like line. Note for context that Kennedy's own GTP-to-MESA "
        "replication was 44% cis / 30% distal / 27% trans using a single method across "
        "two cohorts; a cross-METHOD recovery within that range is a reasonable result, "
        "not a poor one. Jaccard is reported for the diagonal only and is dominated by "
        "the size ratio when the two hit sets differ substantially in size — it is "
        "secondary and should not be read as the headline."
    )
    rows = [
        ["Recovery", f"{diag['recovery']:.3f}"],
        ["Confirmation", f"{diag['confirmation_raw']:.3f}"],
        ["Jaccard", f"{diag['jaccard']:.3f}"]
    ]
    return QCModule(
        anchor="diagonal-summary", title="Diagonal Summary", status="INFO",
        purpose=purpose, interpretation=interpretation,
        table_html=render_table(["Metric", "Value"], rows)
    )


def build_trans_fraction_module(args) -> QCModule:
    purpose = "Compares the trans fraction curves."

    thresholds = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

    t_data = getattr(args, 'trans_fraction_data', {})
    k_data = getattr(args, 'kennedy_trans_fraction_data', {})

    meaningful_dirs = []
    for t in thresholds:
        if t in k_data and t in t_data and (k_data[t] > 0 or t_data[t] > 0):
            if t_data[t] > k_data[t] + 1e-6:
                meaningful_dirs.append("ABOVE")
            elif t_data[t] < k_data[t] - 1e-6:
                meaningful_dirs.append("BELOW")
            else:
                meaningful_dirs.append("EQUAL")

    if not meaningful_dirs:
        dyn_interp = "Curves could not be compared."
    elif all(d == "ABOVE" for d in meaningful_dirs):
        dyn_interp = ("Our trans fraction sits consistently ABOVE the reference curve "
                      "across all matched thresholds.")
    elif all(d == "BELOW" for d in meaningful_dirs):
        dyn_interp = ("Our trans fraction sits consistently BELOW the reference curve "
                      "across all matched thresholds.")
    else:
        # find inversion
        inversions = []
        for i in range(1, len(meaningful_dirs)):
            if meaningful_dirs[i] != meaningful_dirs[i-1]:
                inversions.append(f"{thresholds[i]:g}")
        dyn_interp = f"The direction inverts at threshold(s): {', '.join(inversions)}."

    interpretation = (
        "The reference file's own trans fraction is measured across log10(p). " + dyn_interp +
        " Above means less cis enrichment than the reference at equal stringency; below "
        "means more. Note that GTP and MESA agree to three decimals at 1e-5 despite "
        "differing 3.6x in sample size, which indicates the suggestive-tier trans fraction "
        "reflects test-space geometry rather than biology."
    )

    # Generate the curve figure
    fig = plt.figure(figsize=(6, 4))

    k_vals = [k_data.get(t, 0) for t in thresholds]
    t_vals = [t_data.get(t, 0) for t in thresholds]

    plt.plot(range(len(thresholds)), k_vals, label='Kennedy')
    plt.plot(range(len(thresholds)), t_vals, label='tecpg')
    plt.xticks(range(len(thresholds)), [f"{t:g}" for t in thresholds])
    plt.legend()

    fig.savefig(os.path.join(args.outdir, 'trans_fraction_curve.png'), dpi=300)
    b64 = fig_to_base64(fig)

    return QCModule(
        anchor="trans-fraction", title="Trans Fraction", status="INFO",
        purpose=purpose, interpretation=interpretation,
        figure_b64=b64
    )


# Kennedy et al. 2018 published GTP composition after per-pair probe-location
# selection (annotation.md sec 6.1 / paper text, NOT the supplement file).
# gene body / distal are not itemised in the doc, so only cis and trans are
# stamped as fixed reference constants; the rest is left None on purpose
# (a missing reference is preferable to a fabricated one).
KENNEDY_PAPER_GTP_COMPOSITION = {'cis': 0.473, 'gene body': None, 'distal': None, 'trans': 0.389}


def build_region_composition_module(args) -> QCModule:
    comp = getattr(args, 'region_composition_data', None)
    purpose = ("Region-composition crosswalk at the matched tier. tecpg's seven "
               "labels roll up to Kennedy's cis / gene body / distal / trans "
               "(annotation.md sec 6.2); Kennedy's supplement resolves only IN vs "
               "TRANS, so the file-derived like-for-like is near-gene (IN) vs trans.")
    if not comp or comp.get('tecpg_rollup') is None:
        return QCModule(
            anchor="region-composition", title="Region Composition Crosswalk",
            status="INFO", purpose=purpose,
            interpretation="No region column in the catalog; crosswalk skipped.",
            table_html="")

    ru = comp['tecpg_rollup']
    tn = comp['tecpg_n']
    labeled = tn - ru['unlabeled']
    # tecpg near-gene (Kennedy IN-equivalent) = cis + gene body + distal
    t_near = ru['cis'] + ru['gene body'] + ru['distal']
    t_trans = ru['trans']
    kn = comp.get('kennedy_n', 0)
    frac = lambda x, d: (x / d) if d > 0 else 0.0

    headers = ['Category', 'tecpg n', 'tecpg frac (of labeled)', 'Kennedy paper GTP (per-pair)']
    rows = []
    for cat in ('cis', 'gene body', 'distal', 'trans'):
        ref = KENNEDY_PAPER_GTP_COMPOSITION.get(cat)
        ref_s = f"{ref:.3f}" if ref is not None else "n/a"
        rows.append([cat, str(ru[cat]), f"{frac(ru[cat], labeled):.3f}", ref_s])
    rows.append(['unlabeled (no gene model / no coords)', str(ru['unlabeled']),
                 f"{frac(ru['unlabeled'], tn):.3f} (of all)", 'n/a'])

    # File-derived like-for-like: near-gene vs trans, both sides.
    ll_headers = ['Side', 'near-gene (IN)', 'trans', 'trans fraction']
    ll_rows = [
        ['tecpg (labeled)', str(t_near), str(t_trans), f"{frac(t_trans, t_near + t_trans):.3f}"],
        ['Kennedy (supplement)', str(comp['kennedy_in']), str(comp['kennedy_trans']),
         f"{frac(comp['kennedy_trans'], kn):.3f}"],
    ]

    interpretation = (
        f"Matched tier: tecpg p < {comp['tecpg_thresh']:g}, Kennedy p < {comp['kennedy_thresh']:g}. "
        "Read the tecpg trans fraction as expected to run HIGHER than Kennedy's: Kennedy select "
        "each probe's genomic location per pair (annotation.md sec 6.1), preferring the location "
        "that places the CpG in or near the gene, which moves pairs from trans into cis. tecpg "
        "uses one fixed position per probe, so this is a documented METHOD difference, not a "
        "discrepancy in findings. The Kennedy paper GTP column is the post-selection published "
        "composition (cis and trans only; gene body / distal are not itemised in the source). "
        "The unlabeled row is the population with coordinates but no resolved gene model "
        "(annotation.md sec 5.2) and is excluded from the labeled fractions."
    )
    table_html = render_table(headers, rows) + render_table(ll_headers, ll_rows)
    return QCModule(
        anchor="region-composition", title="Region Composition Crosswalk",
        status="INFO", purpose=purpose, interpretation=interpretation,
        table_html=table_html)


def _safe_spearman(a, b):
    """Spearman rho on aligned non-null pairs; None if <3 usable pairs or no variance."""
    if a is None or b is None:
        return None
    d = pd.DataFrame({'a': a, 'b': b}).dropna()
    if len(d) < 3 or d['a'].nunique() < 2 or d['b'].nunique() < 2:
        return None
    rho, _ = stats.spearmanr(d['a'], d['b'])
    return float(rho) if rho == rho else None


def influence_stratified_analysis(df_merged, tecpg_p_col, kennedy_p_col,
                                  beta_col, tstat_col, tecpg_thresh, kennedy_thresh,
                                  n_bins=10):
    """Stratify the tecpg<->Kennedy overlap by per-CpG leverage (mt_h_max).
    Returns a JSON-native dict; {'skipped': True, ...} if mt_h_max is absent."""
    if 'mt_h_max' not in df_merged.columns or df_merged['mt_h_max'].isna().all():
        return {'skipped': True, 'reason': 'mt_h_max absent from catalog'}

    df = df_merged[df_merged['mt_h_max'].notna()].copy()
    result = {
        'skipped': False,
        'n_pairs': int(len(df)),
        'median_mt_h_max': float(df['mt_h_max'].median()),
    }

    # (2) concordance low vs high (median split on mt_h_max)
    med = float(df['mt_h_max'].median())
    low = df[df['mt_h_max'] <= med]
    high = df[df['mt_h_max'] > med]
    ce_lo = _safe_spearman(low.get('mt_est'), low.get(beta_col)) if beta_col else None
    ce_hi = _safe_spearman(high.get('mt_est'), high.get(beta_col)) if beta_col else None
    ct_lo = _safe_spearman(low.get('mt_t'), low.get(tstat_col)) if tstat_col else None
    ct_hi = _safe_spearman(high.get('mt_t'), high.get(tstat_col)) if tstat_col else None
    result['concordance_low_high'] = {
        'median_split': med,
        'n_low': int(len(low)),
        'n_high': int(len(high)),
        'effect_spearman_low': ce_lo,
        'effect_spearman_high': ce_hi,
        'effect_delta_low_minus_high': (ce_lo - ce_hi) if (ce_lo is not None and ce_hi is not None) else None,
        't_spearman_low': ct_lo,
        't_spearman_high': ct_hi,
        't_delta_low_minus_high': (ct_lo - ct_hi) if (ct_lo is not None and ct_hi is not None) else None,
    }

    # (1) recovery by mt_h_max decile
    try:
        df['_decile'] = pd.qcut(df['mt_h_max'], n_bins, labels=False, duplicates='drop')
    except (ValueError, IndexError):
        df['_decile'] = 0
    deciles = []
    ranks, recs = [], []
    for d in sorted(df['_decile'].dropna().unique()):
        sub = df[df['_decile'] == d]
        ksig = sub[sub[kennedy_p_col] < kennedy_thresh]
        n_elig = int(len(ksig))
        n_conc = int((ksig[tecpg_p_col] < tecpg_thresh).sum())
        rec = (n_conc / n_elig) if n_elig > 0 else None
        deciles.append({
            'decile': int(d),
            'h_max_lo': float(sub['mt_h_max'].min()),
            'h_max_hi': float(sub['mt_h_max'].max()),
            'n_kennedy_sig': n_elig,
            'n_concordant': n_conc,
            'recovery': rec,
        })
        if rec is not None:
            ranks.append(int(d))
            recs.append(rec)
    result['recovery_by_decile'] = deciles
    result['recovery_trend_spearman'] = _safe_spearman(pd.Series(ranks), pd.Series(recs))

    return result


def build_concordance_module(args, figs) -> QCModule:
    purpose = "Venn diagram and effect-size scatter."
    interpretation = (
        "The diagnostic pattern: if Spearman EXCEEDS Pearson on effect size while falling "
        "BELOW it on the test statistic, the two coefficient scales are monotonically "
        "related but not commensurate, which is expected given a mixed model versus "
        "fixed-effect QR and is why the effect-size panel carries no y=x line. High "
        "test-statistic concordance with lower effect-size concordance is therefore a "
        "scale artifact and not a disagreement about which pairs are associated. Sign "
        "agreement in the quadrant annotation is the scale-invariant quantity and is the "
        "one to read. Note that a standardized-effect comparison using mt_err and beta.sd "
        "is planned for C3 and will resolve the scale question directly."
    )

    b64s = []

    if figs.get('concordance_scatter_fig'):
        fig = figs['concordance_scatter_fig']
        fig.savefig(os.path.join(args.outdir, 'concordance_scatter.png'), dpi=300)
        b64s.append(fig_to_base64(fig))

    if figs.get('venn_fig'):
        fig = figs['venn_fig']
        fig.savefig(os.path.join(args.outdir, 'overlap_venn_diagonal.png'), dpi=300)
        b64s.append(fig_to_base64(fig))

    if figs.get('upset_fig'):
        fig = figs['upset_fig']
        fig.savefig(os.path.join(args.outdir, 'overlap_upset_diagonal.png'), dpi=300)
        b64s.append(fig_to_base64(fig))

    # Hack to include multiple images in one module's HTML since QCModule takes one figure_b64.
    # We can inject them via table_html or just pick one for figure_b64 and the rest in table_html.
    # Actually, we can just put them all in table_html as <img> tags.
    table_html = ""
    for b64 in b64s:
        table_html += f'<img src="data:image/png;base64,{b64}" style="max-width:45%; margin: 5px;">'

    return QCModule(
        anchor="concordance", title="Concordance", status="INFO",
        purpose=purpose, interpretation=interpretation,
        table_html=table_html
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark tecpg output against Kennedy eQTL summary statistics.")
    parser.add_argument('-t', '--tecpg', required=True, help="Path to tecpg output parquet file")
    parser.add_argument('-k', '--kennedy', required=True, help="Path to Kennedy supplementary CSV/TSV file")
    parser.add_argument('-o', '--outdir', default='.', help="Directory to save output plots and summary")

    parser.add_argument('--tecpg-thresh', type=float, default=None, help="p-value threshold for tecpg hits")
    parser.add_argument('--kennedy-thresh', type=float, default=None, help="p-value threshold for Kennedy hits")
    parser.add_argument('--p-thresh', type=float, default=None, help="Deprecated. Use --kennedy-thresh")

    parser.add_argument('--kennedy-sep', default='\t', help="Separator for Kennedy file")
    parser.add_argument('--tecpg-p-col', default=None, help="Override for tecpg precise_mt_p column")
    parser.add_argument('--upset', action='store_true', help="Generate UpSet plot")
    parser.add_argument('--batch-size', type=int, default=500_000, help="Batch size for streaming catalog")
    parser.add_argument('--characterize', action='store_true', help="Profile inputs and exit without comparing")
    parser.add_argument('--no-html', action='store_true', help="Skip HTML report generation")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    resolved_k, resolved_t = resolve_thresholds(args.p_thresh, args.kennedy_thresh, args.tecpg_thresh)
    args.kennedy_thresh = resolved_k
    args.tecpg_thresh = resolved_t

    os.makedirs(args.outdir, exist_ok=True)

    logging.info(f"Loading Kennedy file: {args.kennedy}")
    df_kennedy = load_kennedy(args.kennedy, args.kennedy_sep)
    cols = resolve_kennedy_columns(df_kennedy.columns)

    cpg_col = cols['cpg']
    query_col = cols['probe']

    before_dropna = len(df_kennedy)
    df_kennedy = df_kennedy.dropna(subset=[cpg_col, query_col])
    after_dropna = len(df_kennedy)

    logging.info(
        f"Drop site benchmark_kennedy.dropna_keys[{query_col},{cpg_col}]: dropped "
        f"Kennedy rows with missing key columns: {before_dropna} -> {after_dropna} "
        f"({before_dropna - after_dropna} dropped)"
    )

    if cols['gene']:
        logging.info(f"Using column '{cols['gene']}' for Gene cardinality diagnostic.")
        genes = len(df_kennedy[cols['gene']].dropna().unique())
        logging.info(f"Kennedy unique gene symbols (annot.gene): {genes}")

    logging.info(f"Loading tecpg catalog: {args.tecpg}")
    schema = pq.ParquetFile(args.tecpg).schema_arrow.names
    tecpg_p_col = resolve_tecpg_pvalue_column(schema, args.tecpg_p_col)

    kennedy_pairs = set(zip(df_kennedy[cols['cpg']], df_kennedy[cols['probe']]))
    thresholds = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
    kennedy_cpgs = set(df_kennedy[cols['cpg']].dropna())
    kennedy_probes = set(df_kennedy[cols['probe']].dropna())

    # Profile Kennedy BEFORE streaming
    kennedy_profile_metrics = profile_kennedy(args.kennedy, df_kennedy, args.kennedy_sep, cols, thresholds)
    args.kennedy_profile_metrics = kennedy_profile_metrics

    df_matched, tecpg_threshold_counts, distinct_mt, distinct_gt, region_counter, cat_profile_metrics = \
        stream_catalog_and_match(
            args, schema, tecpg_p_col, kennedy_pairs, thresholds, kennedy_cpgs, kennedy_probes
        )

    catalog_profile_metrics = profile_catalog_post_stream(
        args, args.tecpg, cat_profile_metrics, schema, distinct_mt, distinct_gt, region_counter)
    args.catalog_profile_metrics = catalog_profile_metrics

    if args.characterize:
        logging.info("Characterization complete. Exiting (--characterize flag present).")
        write_provenance_and_reports(
            args, num_merged=None, diag_results=None, grid_results=None, df_kennedy=df_kennedy, cols=cols
        )
        sys.exit(0)

    args.tecpg_threshold_counts = tecpg_threshold_counts
    args.region_counter = region_counter
    args.cat_profile = cat_profile_metrics

    df_kennedy = compute_eligibility(distinct_mt, distinct_gt, df_kennedy, cols)

    loci_overlap = len(distinct_mt.intersection(kennedy_cpgs))
    logging.info(f"Overlapping distinct CpG loci: {loci_overlap}")

    genes_overlap = len(distinct_gt.intersection(kennedy_probes))
    logging.info(f"Overlapping distinct expression probes (exp.Probe): {genes_overlap}")

    logging.info("Merging datasets (inner join)...")
    df_merged = pd.merge(
        df_matched,
        df_kennedy,
        left_on=['mt_id', 'gt_id'],
        right_on=[cpg_col, query_col],
        how='inner',
        suffixes=('_tecpg', '_kennedy')
    )

    num_merged = len(df_merged)
    logging.info(f"Successfully mapped and merged {num_merged} pairs.")

    if num_merged == 0:
        logging.error("No overlapping pairs found between datasets. Please check the ID mapping.")
        sys.exit(1)

    beta_col = cols['beta']
    tstat_col = cols['tstat']

    if beta_col is None or tstat_col is None:
        logging.warning("Optional beta or T.stat column absent. Dependent concordance metrics will be skipped.")

    # Influence-stratified overlap: does replication fall / concordance degrade with CpG leverage?
    influence_stats = influence_stratified_analysis(
        df_merged, tecpg_p_col, cols['pval'], beta_col, tstat_col,
        args.tecpg_thresh, args.kennedy_thresh
    )
    with open(os.path.join(args.outdir, 'influence_stratified.json'), 'w') as _f:
        json.dump(influence_stats, _f, indent=2)
    logging.info(
        "Influence-stratified analysis -> influence_stratified.json "
        f"(skipped={influence_stats.get('skipped')})"
    )

    pearson_r_beta = spearman_r_beta = r2_beta = 0.0
    pearson_r_t = spearman_r_t = r2_t = 0.0
    concordance_scatter_fig = None
    venn_fig = None
    upset_fig = None


    panels_to_draw = []
    if beta_col and 'mt_est' in df_merged.columns:
        valid_beta = df_merged[['mt_est', beta_col]].dropna()
        dropped = len(df_merged) - len(valid_beta)
        logging.info(
            f"Drop site benchmark_kennedy.valid_beta[mt_est,{beta_col}]: "
            f"dropped merged pairs with missing effect-size values: {len(df_merged)} -> "
            f"{len(valid_beta)} ({dropped} dropped)"
        )
        if len(valid_beta) > 1:
            panels_to_draw.append(('beta', valid_beta))

    if tstat_col and 'mt_t' in df_merged.columns:
        valid_t = df_merged[['mt_t', tstat_col]].dropna()
        dropped = len(df_merged) - len(valid_t)
        logging.info(
            f"Drop site benchmark_kennedy.valid_t[mt_t,{tstat_col}]: "
            f"dropped merged pairs with missing test-statistic values: {len(df_merged)} -> "
            f"{len(valid_t)} ({dropped} dropped)"
        )
        if len(valid_t) > 1:
            panels_to_draw.append(('tstat', valid_t))

    if panels_to_draw:
        from matplotlib.colors import LogNorm
        import numpy as np
        fig, axes = plt.subplots(1, len(panels_to_draw), figsize=(7 * len(panels_to_draw), 6))
        if len(panels_to_draw) == 1:
            axes = [axes]

        for i, (p_type, valid_df) in enumerate(panels_to_draw):
            ax = axes[i]
            x = valid_df.iloc[:, 0]
            y = valid_df.iloc[:, 1]

            pearson_r, _ = stats.pearsonr(x, y)
            spearman_r, _ = stats.spearmanr(x, y)
            r2 = pearson_r**2

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            hb = ax.hexbin(x, y, gridsize=50, cmap='viridis', mincnt=1, norm=LogNorm(vmin=1))
            fig.colorbar(hb, ax=ax, label='pairs per cell')

            ax.axhline(0, color='lightgrey', lw=1, zorder=0)
            ax.axvline(0, color='lightgrey', lw=1, zorder=0)

            # Quadrants
            q1 = ((x > 0) & (y > 0)).sum()
            q2 = ((x < 0) & (y > 0)).sum()
            q3 = ((x < 0) & (y < 0)).sum()
            q4 = ((x > 0) & (y < 0)).sum()

            total_points = len(x)
            concordant = q1 + q3

            pct_q1 = (q1 / total_points) * 100 if total_points > 0 else 0
            pct_q2 = (q2 / total_points) * 100 if total_points > 0 else 0
            pct_q3 = (q3 / total_points) * 100 if total_points > 0 else 0
            pct_q4 = (q4 / total_points) * 100 if total_points > 0 else 0

            ax.text(0.98, 0.98, f"{q1:,} ({pct_q1:.1f}%)", transform=ax.transAxes, ha='right', va='top', fontsize=9)
            ax.text(0.02, 0.98, f"{q2:,} ({pct_q2:.1f}%)", transform=ax.transAxes, ha='left', va='top', fontsize=9)
            ax.text(0.02, 0.02, f"{q3:,} ({pct_q3:.1f}%)", transform=ax.transAxes, ha='left', va='bottom', fontsize=9)
            ax.text(0.98, 0.02, f"{q4:,} ({pct_q4:.1f}%)", transform=ax.transAxes, ha='right', va='bottom', fontsize=9)

            # Regression line
            min_val_x = x.min()
            max_val_x = x.max()
            fit_x = np.array([min_val_x, max_val_x])
            fit_y = slope * fit_x + intercept
            ax.plot(fit_x, fit_y, color='orange', linestyle='--', lw=2, label='OLS fit')

            text_stats = (f"Pearson r: {pearson_r:.3f}\n"
                  f"Spearman $\\rho$: {spearman_r:.3f}\n"
                  f"$R^2$: {r2:.3f}\n"
                  f"OLS Slope: {slope:.3f}\n"
                  "(OLS is asymmetric)")

            if p_type == 'beta':
                pearson_r_beta, spearman_r_beta, r2_beta = pearson_r, spearman_r, r2
                ax.set_xlabel('tecpg Effect Size (mt_est)')
                ax.set_ylabel(f'Kennedy Effect Size ({beta_col})')
                ax.set_title(f'Effect Size Concordance\n{concordant:,} / {total_points:,} concordant signs')
            else:
                pearson_r_t, spearman_r_t, r2_t = pearson_r, spearman_r, r2
                ax.set_xlabel('tecpg Test Statistic (mt_t)')
                ax.set_ylabel(f'Kennedy Test Statistic ({tstat_col})')
                ax.set_title(f'Test Statistic Concordance\n{concordant:,} / {total_points:,} concordant signs')
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x')

            ax.text(0.02, 0.5, text_stats, transform=ax.transAxes, fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend(loc='lower right', fontsize=9)

        plt.tight_layout()
        concordance_scatter_fig = fig

    logging.info("Calculating directional overlap rates (sweep)...")
    thresholds = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

    kennedy_cpgs = set(df_kennedy[cols['cpg']].dropna())
    kennedy_probes = set(df_kennedy[cols['probe']].dropna())
    grid_results = {}
    for tt in thresholds:
        for tk in thresholds:
            grid_results[(tk, tt)] = compute_overlap_rates(
                df_matched, df_kennedy, cols, tk, tt, tecpg_p_col,
                return_sets=False, kennedy_cpgs=kennedy_cpgs, kennedy_probes=kennedy_probes,
                tecpg_threshold_counts=tecpg_threshold_counts
            )

    logging.info("Calculating directional overlap rates (diagonal)...")
    diag_results = compute_overlap_rates(
        df_matched, df_kennedy, cols, args.kennedy_thresh, args.tecpg_thresh, tecpg_p_col,
        return_sets=True, kennedy_cpgs=kennedy_cpgs, kennedy_probes=kennedy_probes,
        tecpg_threshold_counts=tecpg_threshold_counts
    )

    logging.info("Exporting pair lists...")
    export_pair_lists(args.outdir, df_matched, df_kennedy, cols, diag_results, tecpg_p_col)

    T_tt = diag_results['T_tt']
    K_tk_E = diag_results['K_tk_E']

    ineligible_count = len(diag_results['K_tk']) - len(K_tk_E)

    fig_venn = plt.figure(figsize=(8, 6))
    v = venn2([T_tt, K_tk_E], set_labels=('tecpg hits', 'Kennedy hits (Eligible)'))
    plt.title(f'Diagonal Overlap\n({ineligible_count} ineligible Kennedy hits omitted)')
    venn_fig = fig_venn

    if args.upset and (len(T_tt) > 0 or len(K_tk_E) > 0):
        upset_data = upsetplot.from_contents({
            'tecpg': T_tt,
            'Kennedy (Eligible)': K_tk_E
        })
        fig_upset = plt.figure(figsize=(8, 6))
        upsetplot.plot(upset_data, fig=fig_upset)
        plt.title('UpSet Plot (Eligible)')
        upset_fig = fig_upset

    logging.info("Building summary...")
    summary = build_summary_text(
        args, num_merged, pearson_r_beta, spearman_r_beta, r2_beta, beta_col,
        pearson_r_t, spearman_r_t, r2_t, tstat_col,
        grid_results, diag_results, thresholds, cols, df_kennedy
    )

    prov = build_provenance(args, df_kennedy)
    summary = add_provenance_to_summary(summary, prov)

    print(summary)
    with open(os.path.join(args.outdir, 'benchmark_summary.txt'), 'w') as f:
        f.write(summary)


    if not args.characterize:
        t_data = {}
        k_data = {}
        for t in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]:
            if cols['pval'] in df_kennedy.columns and cols['status'] in df_kennedy.columns:
                k_mask = df_kennedy[cols['pval']] < t
                k_sub = df_kennedy[k_mask]
                k_n = len(k_sub)
                k_trans = (k_sub[cols['status']] == 'TRANS').sum()
                k_data[t] = k_trans / k_n if k_n > 0 else 0.0

            if tecpg_p_col in df_matched.columns and 'region' in df_matched.columns:
                t_mask = df_matched[tecpg_p_col] < t
                t_sub = df_matched[t_mask]
                t_n = len(t_sub)
                t_trans = (t_sub['region'] == 'TRANS').sum()
                t_data[t] = t_trans / t_n if t_n > 0 else 0.0

        args.trans_fraction_data = t_data
        args.kennedy_trans_fraction_data = k_data

        # C4: region-composition crosswalk at the matched (diagonal) tier.
        # Kennedy percentages are over their significant set (annotation.md sec 6.2),
        # so this is pinned to the matched tier, not the full test space.
        tt = args.tecpg_thresh
        kt = args.kennedy_thresh
        comp = {'tecpg_rollup': None, 'tecpg_n': 0,
                'kennedy_in': 0, 'kennedy_trans': 0, 'kennedy_n': 0,
                'tecpg_thresh': tt, 'kennedy_thresh': kt}
        if 'region' in df_matched.columns and tecpg_p_col in df_matched.columns:
            t_sig = df_matched[df_matched[tecpg_p_col] < tt]
            comp['tecpg_rollup'] = rollup_region_to_kennedy(t_sig['region'])
            comp['tecpg_n'] = int(len(t_sig))
        if cols['status'] in df_kennedy.columns and cols['pval'] in df_kennedy.columns:
            k_sig = df_kennedy[df_kennedy[cols['pval']] < kt]
            comp['kennedy_trans'] = int((k_sig[cols['status']] == 'TRANS').sum())
            comp['kennedy_in'] = int((k_sig[cols['status']] == 'IN').sum())
            comp['kennedy_n'] = int(len(k_sig))
        args.region_composition_data = comp

    write_provenance_and_reports(
        args, num_merged, diag_results, grid_results, df_kennedy, cols,
        figs={'concordance_scatter_fig': concordance_scatter_fig, 'venn_fig': venn_fig, 'upset_fig': upset_fig}
    )


if __name__ == '__main__':
    main()
