import argparse
import fnmatch
import logging
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WINDOWS = ["1-50", "200-250", "1000-1050"]

def parse_rank_windows(windows_str_list):
    windows = []
    for w in windows_str_list:
        try:
            start, end = map(int, w.split('-'))
            if start > end or start < 1:
                raise ValueError
            windows.append((start, end))
        except ValueError:
            logger.error(f"Invalid window format: {w}. Must be START-END with START <= END and START >= 1.")
            sys.exit(1)
    return windows

def detect_inflection(y, method='auto'):
    n = len(y)
    if n < 3:
        return None, "Not enough points"

    x = np.arange(n)

    use_kneed = False
    if method in ['auto', 'kneed']:
        try:
            from kneed import KneeLocator
            use_kneed = True
        except ImportError:
            if method == 'kneed':
                logger.error("kneed package not found but required by --inflection-method kneed.")
                sys.exit(1)
            use_kneed = False

    if use_kneed:
        kl = KneeLocator(x, y, curve='convex', direction='decreasing')
        knee = kl.knee
        if knee is not None:
            return knee, "kneed"

    # Fallback: max distance to chord
    # Normalize x and y to [0, 1] so that differing scales don't distort distance
    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y

    # chord line connects (x_norm[0], y_norm[0]) to (x_norm[-1], y_norm[-1])
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    # Update p_vec to use normalized values
    norm_points = np.column_stack((x_norm, y_norm))

    # Calculate distance from each point to the line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0, "chord"

    line_unitvec = line_vec / line_len
    p_vec = norm_points - p1

    # t is the scalar projection of p_vec onto the line
    t = np.dot(p_vec, line_unitvec)

    # nearest points on the line
    nearest = p1 + np.outer(t, line_unitvec)

    # distance
    dist = np.linalg.norm(norm_points - nearest, axis=1)

    knee = np.argmax(dist)
    return knee, "chord"

def load_data(filepath, rank_by):
    logger.info(f"Loading data from {filepath}...")
    try:
        parquet_file = pq.ParquetFile(filepath)
        schema = parquet_file.schema
        columns = schema.names

        # We need ranking metric, mt_ig, all _ig columns, region, mt_est/mt_t if available, mt_id, gt_id
        cols_to_load = ['mt_id', 'gt_id', 'region']
        if rank_by in columns:
            cols_to_load.append(rank_by)
        else:
            logger.error(f"Ranking metric '{rank_by}' not found in Parquet schema.")
            sys.exit(1)

        ig_cols = [c for c in columns if c.endswith('_ig')]
        cols_to_load.extend(ig_cols)

        if 'mt_est' in columns and 'mt_est' not in cols_to_load:
            cols_to_load.append('mt_est')
        if 'mt_t' in columns and 'mt_t' not in cols_to_load:
            cols_to_load.append('mt_t')

        # Optional precise p
        if 'p_boot' in columns and 'p_boot' not in cols_to_load:
            cols_to_load.append('p_boot')

        cols_to_load = list(set(cols_to_load).intersection(set(columns)))

        # Read the file in chunks to handle very large datasets without OOM
        batches = []
        for batch in parquet_file.iter_batches(columns=cols_to_load, batch_size=500000):
            batch_df = batch.to_pandas()
            if batch_df.index.names != [None]:
                batch_df = batch_df.reset_index()
            batches.append(batch_df)

        df = pd.concat(batches, ignore_index=True)

        # Fill NA region
        if 'region' in df.columns:
            df['region'] = df['region'].fillna('UNKNOWN')
        else:
            df['region'] = 'UNKNOWN'

        return df, ig_cols
    except Exception as e:
        logger.error(f"Failed to read Parquet file: {e}")
        sys.exit(1)

def print_top_pairs(df, sort_col, ascending, title, display_cols,
                    n=50, out_dir=None, filename=None):
    """Print (and optionally save) the top-N pairs ranked by `sort_col`.

    Operates on a sorted copy, so it is independent of the global --rank-by
    ordering applied elsewhere. NaNs in the sort column are pushed to the end.
    """
    if sort_col not in df.columns:
        logger.warning(f"Column '{sort_col}' not found; skipping '{title}'.")
        return

    cols = [c for c in display_cols if c in df.columns]
    top = (df.sort_values(by=sort_col, ascending=ascending, na_position='last')
             .head(n)[cols]
             .reset_index(drop=True))
    top.index = top.index + 1
    top.index.name = 'rank'

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    with pd.option_context('display.max_rows', n + 1,
                           'display.width', 200,
                           'display.max_columns', None,
                           'display.float_format', lambda v: f"{v:.4g}"):
        print(top.to_string())

    if out_dir and filename:
        path = os.path.join(out_dir, filename)
        top.to_csv(path)
        logger.info(f"Wrote {path}")

def _subsample(frame, n):
    """Random subsample for plot legibility/scale (deterministic)."""
    if n is not None and n > 0 and len(frame) > n:
        return frame.sample(n=n, random_state=0)
    return frame

def _compute_true_mad(meth_csv):
    """Per-CpG mean absolute deviation from the methylation matrix (CpG x samples).

    Uses MEAN absolute deviation (not median) to match the analytical-IG
    baseline mean_s|X - X_bar| with ig_baseline='mean'.
    """
    logger.info(f"Computing per-CpG mean abs deviation from {meth_csv} ...")
    M = pd.read_csv(meth_csv, index_col=0)
    mad = (M.sub(M.mean(axis=1), axis=0)).abs().mean(axis=1)
    mad.name = 'mad_meth'
    return mad

def plot_effect_diagnostics(df, out_dir, df_resid=None, meth_csv=None, plot_sample=200000):
    """Diagnostics for the |mt_est| <-> mt_ig_frac relationship.

    Produces three figures:
      1. effect_vs_mad.png                              -- |mt_est| vs methylation MAD (log-log):
         the mechanism. Large |beta| tracks LOW methylation variability, because the analytical
         IG is mt_ig = MAD(M) * |beta|, so an inflated coefficient implies a near-invariant probe.
      2. saliency_fraction_vs_effect_mad.png            -- the original frac vs |mt_est| scatter,
         colored by MAD: shows the high-|mt_est| floor points are exactly the low-MAD probes.
      3. saliency_fraction_vs_standardized_effect.png   -- frac vs a standardized effect
         (partial correlation r if --df given, else |mt_t|): variance/error-bounded, so the
         coefficient blow-up that stretches the |mt_est| axis no longer occurs.

    MAD source: independent (from --methylation-csv) when available, otherwise the recovered
    value mt_ig/|mt_est| from the IG identity. The recovered value is an algebraic re-expression
    of mt_ig and |mt_est|, not an independent measurement -- this is noted on figures 1 and 2.
    """
    cols = ['mt_id', 'mt_ig', 'mt_est', 'mt_ig_frac']
    if 'mt_t' in df.columns:
        cols.append('mt_t')
    work = df[cols].copy()
    work['abs_est'] = work['mt_est'].abs()

    # --- MAD: independent (preferred) or recovered from the IG identity ---
    mad_source = 'recovered (mt_ig / |mt_est|)'
    recovered = False
    if meth_csv:
        try:
            mad = _compute_true_mad(meth_csv)
            work = work.merge(mad, left_on='mt_id', right_index=True, how='left')
            n_missing = int(work['mad_meth'].isna().sum())
            if n_missing:
                logger.warning(f"{n_missing} pairs had no MAD match in methylation CSV (left NaN).")
            mad_source = f'independent (mean abs dev from {os.path.basename(meth_csv)})'
        except Exception as e:
            logger.warning(f"Could not compute MAD from {meth_csv}: {e}. Falling back to recovered MAD.")
            work['mad_meth'] = work['mt_ig'] / work['abs_est'].replace(0, np.nan)
            recovered = True
    else:
        work['mad_meth'] = work['mt_ig'] / work['abs_est'].replace(0, np.nan)
        recovered = True

    note = f"MAD source: {mad_source}."
    if recovered:
        note += " Recovered MAD is an algebraic re-expression of mt_ig and |mt_est|, not independent."

    # --- Standardized effect ---
    std_label = None
    if 'mt_t' in work.columns:
        if df_resid is not None and df_resid > 0:
            t = work['mt_t']
            work['std_effect'] = (t / np.sqrt(t ** 2 + df_resid)).abs()
            std_label = f"|partial correlation r|  (df={df_resid})"
        else:
            work['std_effect'] = work['mt_t'].abs()
            std_label = "|mt_t|  (t-standardized effect; pass --df for partial r)"

    # Plot 1: |mt_est| vs MAD (mechanism)
    p1 = _subsample(work.dropna(subset=['abs_est', 'mad_meth']), plot_sample)
    p1 = p1[(p1['abs_est'] > 0) & (p1['mad_meth'] > 0)]
    plt.figure(figsize=(8, 6))
    plt.scatter(p1['abs_est'], p1['mad_meth'], s=6, alpha=0.3, edgecolors='none')
    plt.xscale('log'); plt.yscale('log')
    plt.title("Methylation variability (MAD) vs |mt_est|")
    plt.xlabel("|mt_est|  (|beta|, log scale)")
    plt.ylabel("Methylation MAD (log scale)")
    plt.figtext(0.01, 0.01, note, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "effect_vs_mad.png"))
    plt.close()

    # Plot 2: fraction vs |mt_est|, colored by MAD (log color scale)
    from matplotlib.colors import LogNorm
    p2 = _subsample(work.dropna(subset=['abs_est', 'mt_ig_frac', 'mad_meth']), plot_sample)
    p2 = p2[p2['abs_est'] > 0]
    cvals = p2['mad_meth'].where(p2['mad_meth'] > 0).dropna()
    norm = LogNorm(vmin=cvals.min(), vmax=cvals.max()) if len(cvals) and cvals.min() > 0 else None
    plt.figure(figsize=(8.5, 6))
    sc = plt.scatter(p2['abs_est'], p2['mt_ig_frac'],
                     c=p2['mad_meth'].where(p2['mad_meth'] > 0),
                     s=8, alpha=0.5, edgecolors='none', cmap='viridis', norm=norm)
    plt.colorbar(sc, label="Methylation MAD" + (" (log)" if norm is not None else ""))
    plt.xscale('log')
    plt.title("Methylation Saliency Fraction vs |mt_est| (colored by MAD)")
    plt.xlabel("|mt_est| (log scale)")
    plt.ylabel("Methylation Saliency Fraction")
    plt.figtext(0.01, 0.01, note, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "saliency_fraction_vs_effect_mad.png"))
    plt.close()

    # Plot 3: fraction vs standardized effect
    if std_label is not None:
        p3 = _subsample(work.dropna(subset=['std_effect', 'mt_ig_frac']), plot_sample)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=p3['std_effect'], y=p3['mt_ig_frac'], alpha=0.3, edgecolor=None, s=8)
        plt.title("Methylation Saliency Fraction vs Standardized Effect")
        plt.xlabel(std_label)
        plt.ylabel("Methylation Saliency Fraction")
        plt.figtext(0.01, 0.01,
                    "Standardized effect bounds the coefficient by its variability/error, so "
                    "variance-inflated |beta| no longer extends the x-axis.", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "saliency_fraction_vs_standardized_effect.png"))
        plt.close()
    else:
        logger.warning("mt_t not available; skipping standardized-effect plot.")

    logger.info("Wrote effect-diagnostic plots: effect_vs_mad.png, "
                "saliency_fraction_vs_effect_mad.png, "
                "saliency_fraction_vs_standardized_effect.png")

def report_input_scales(df, denom_cols, covar_csv, meth_csv=None, out_dir=None):
    """Report per-feature INPUT scale (MAD of the model-input columns) next to IG
    attribution, to separate 'large attribution because large input scale' from
    'large attribution because large coefficient'.

    Because IG_k = MAD(X_k) * |beta_k|:
      - input MAD comes from the actual model inputs: C.csv for covariates;
        M.csv (or the recovered mt_ig/|mt_est|) summarized for methylation.
      - scale_adj|b| = mean|IG_k| / MAD(X_k) strips the input scale and is the
        cohort-portable coefficient size. For covariates MAD(X_k) is constant
        across pairs, so this equals mean|beta_k| exactly.
    If methylation's input MAD is orders of magnitude below the covariates', the
    near-zero methylation IG fraction is a scale artifact, not low importance.
    """
    if not covar_csv:
        logger.warning("Input-scale diagnostic skipped: pass --covariates-csv (C.csv) to "
                       "report covariate input scales alongside IG.")
        return
    try:
        C = pd.read_csv(covar_csv, index_col=0)
    except Exception as e:
        logger.warning(f"Could not read covariates CSV {covar_csv}: {e}. Skipping input-scale diagnostic.")
        return

    covar_mad = (C - C.mean()).abs().mean()

    # Methylation input scale: one summary number over the evaluated CpGs.
    abs_est = df['mt_est'].abs().replace(0, np.nan)
    if meth_csv:
        try:
            mad_series = _compute_true_mad(meth_csv)
            meth_mad = float(mad_series.reindex(pd.Index(df['mt_id'].unique())).median())
            meth_src = f"median per-CpG MAD from {os.path.basename(meth_csv)}"
        except Exception as e:
            logger.warning(f"Methylation MAD from {meth_csv} failed: {e}; using recovered.")
            meth_mad = float((df['mt_ig'] / abs_est).median())
            meth_src = "median recovered MAD (mt_ig/|mt_est|)"
    else:
        meth_mad = float((df['mt_ig'] / abs_est).median())
        meth_src = "median recovered MAD (mt_ig/|mt_est|)"

    # rows: (feature, input_mad, mean_ig)
    rows = [('mt_ig', meth_mad, float(df['mt_ig'].mean()))]
    missing = []
    for col in denom_cols:
        if col == 'mt_ig':
            continue
        name = col[:-3] if col.endswith('_ig') else col
        if name in covar_mad.index:
            rows.append((col, float(covar_mad[name]), float(df[col].mean())))
        else:
            missing.append(name)
    if missing:
        logger.warning(f"No matching covariate column in {os.path.basename(covar_csv)} for: "
                       f"{', '.join(missing)}")

    rows.sort(key=lambda r: (np.nan_to_num(r[2], nan=-1.0)), reverse=True)

    print("\n--- Input Scale vs IG Attribution ---")
    print(f"(methylation input scale: {meth_src})")
    print(f"{'feature':<14}{'input_MAD':>12}{'mean|IG|':>12}{'scale_adj|b|':>14}{'MADx_vs_mt':>12}")
    for name, imad, mig in rows:
        sadj = mig / imad if (imad and imad > 0) else float('nan')
        madx = imad / meth_mad if (meth_mad and meth_mad > 0) else float('nan')
        print(f"{name:<14}{imad:>12.4g}{mig:>12.4g}{sadj:>14.4g}{madx:>12.4g}")
    print("scale_adj|b| = mean|IG| / input_MAD  (input-scale-free coefficient size).")
    print("MADx_vs_mt   = input_MAD / methylation_MAD  (how many x wider the input is).")

    covar_mads = [r[1] for r in rows if r[0] != 'mt_ig' and r[1] == r[1] and r[1] > 0]
    if covar_mads and meth_mad and meth_mad > 0:
        maxratio = max(covar_mads) / meth_mad
        if maxratio >= 10:
            print(f"FLAG: largest covariate input MAD is {maxratio:.1f}x methylation's. "
                  "Covariate IG dominance is at least partly an input-scale effect; "
                  "compare scale_adj|b| (not raw IG) across features.")
        else:
            print(f"Input scales are within ~{maxratio:.1f}x across features; covariate IG "
                  "dominance is not primarily a scale artifact.")

    # Plot: input MAD vs mean|IG| (log-log) with a slope-1 reference. Points far
    # off the line have coefficient sizes that differ from the median feature.
    if out_dir:
        pts = [(n, i, m) for (n, i, m) in rows if i == i and m == m and i > 0 and m > 0]
        if len(pts) >= 2:
            xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
            plt.figure(figsize=(8, 6))
            plt.scatter(xs, ys, s=30)
            for n, i, m in pts:
                plt.annotate(n.replace('_ig', ''), (i, m), fontsize=7,
                             xytext=(3, 3), textcoords='offset points')
            c = np.exp(np.median(np.log(ys) - np.log(xs)))
            gx = np.array([xs.min(), xs.max()])
            plt.plot(gx, c * gx, 'r--', lw=1, label='slope 1 (IG proportional to input scale)')
            plt.xscale('log'); plt.yscale('log')
            plt.xlabel('Input MAD (model-input scale, log)')
            plt.ylabel('mean |IG| (log)')
            plt.title('Per-feature input scale vs IG attribution')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'input_scale_vs_ig.png'))
            plt.close()
            logger.info("Wrote input_scale_vs_ig.png")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Integrated Gradients Saliency distribution.")
    parser.add_argument("-i", "--input", required=True, help="Input Parquet file (e.g., bootstrap_merged.parquet).")
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory for plots.")
    parser.add_argument("--rank-by", default="mt_ig", choices=["mt_ig", "p_boot", "mt_t", "mt_est", "abs_t"], help="Metric to rank pairs by.")
    parser.add_argument("--rank-windows", nargs="+", default=DEFAULT_WINDOWS, help="Rank windows for summaries (e.g., 1-50 200-250).")
    parser.add_argument("--inflection-method", default="auto", choices=["auto", "kneed", "chord"], help="Method to detect inflection point.")
    parser.add_argument("--df", type=int, default=None,
                        help="Residual degrees of freedom (SAMPLES - COVARS - 2). If given, the "
                             "standardized-effect plot uses the partial correlation "
                             "r = mt_t / sqrt(mt_t^2 + df); otherwise it uses |mt_t| directly.")
    parser.add_argument("--methylation-csv", default=None,
                        help="Optional methylation matrix (CpG x samples). If given, per-CpG mean "
                             "absolute deviation is computed from it for the MAD-colored plot, "
                             "independently of the IG identity. Otherwise MAD is recovered "
                             "algebraically as mt_ig/|mt_est|.")
    parser.add_argument("--plot-sample", type=int, default=200000,
                        help="Max points to scatter in the effect-diagnostic plots (random "
                             "subsample for legibility at scale). Set <=0 to plot all.")
    parser.add_argument("--frac-exclude", nargs="+", default=None, metavar="PATTERN",
                        help="Glob patterns of *_ig feature columns to EXCLUDE from the "
                             "mt_ig_frac denominator (total_ig), e.g. 'Exp_PC*_ig'. mt_ig is "
                             "never excludable (it is the numerator). Default: include all "
                             "*_ig features in the denominator (faithful to raw IG).")
    parser.add_argument("--covariates-csv", default=None,
                        help="Optional covariate matrix (samples x covariates, e.g. C.csv). If "
                             "given, per-feature INPUT MAD is reported next to IG attribution, "
                             "plus a scale-adjusted |beta| = mean|IG|/input_MAD, so input-scale "
                             "effects can be separated from coefficient size. Required for the "
                             "input-scale diagnostic.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    windows = parse_rank_windows(args.rank_windows)

    df, ig_cols = load_data(args.input, args.rank_by)

    if 'mt_ig' not in ig_cols:
        logger.error("No 'mt_ig' column found in the dataset. Saliency evaluation requires Integrated Gradients output.")
        sys.exit(1)

    # Restrict to rows that actually carry IG. IG is produced only at the bootstrap
    # stage, so on a full master table most rows have NaN mt_ig (and NaN covariate
    # IG). Including them silently poisons every aggregate (fillna(0) turns absent
    # IG into a real-looking zero) and breaks the magnitude inflection (a NaN tail
    # collapses the chord to rank 1). Every saliency product below is therefore an
    # explicit description of the IG-bearing (bootstrapped, prioritized) set.
    n_total = len(df)
    df = df[df['mt_ig'].notna()].reset_index(drop=True)
    n_valid = len(df)
    if n_valid == 0:
        logger.error("No rows carry a non-null 'mt_ig'. Nothing to evaluate.")
        sys.exit(1)
    logger.info(f"IG coverage: {n_valid} / {n_total} pairs carry IG "
                f"({100.0 * n_valid / n_total:.3f}%); evaluating on those.")

    has_covar_ig = len(ig_cols) > 1

    # Absolute magnitudes
    for col in ig_cols:
        df[col] = df[col].abs()

    # Resolve the denominator feature set. mt_ig is always retained (it is the
    # numerator); --frac-exclude removes covariate features (e.g. expression PCs,
    # which are near-proxies for the outcome and otherwise dominate the denominator).
    denom_cols = list(ig_cols)
    if args.frac_exclude:
        excluded = sorted({c for c in ig_cols if c != 'mt_ig'
                           and any(fnmatch.fnmatch(c, pat) for pat in args.frac_exclude)})
        if excluded:
            denom_cols = [c for c in ig_cols if c not in excluded]
            logger.info(f"Excluding {len(excluded)} feature(s) from the mt_ig_frac "
                        f"denominator: {', '.join(excluded)}")
        else:
            logger.warning(f"--frac-exclude {args.frac_exclude} matched no *_ig columns; "
                           f"denominator unchanged.")

    if has_covar_ig:
        df['total_ig'] = df[denom_cols].sum(axis=1)
        df['mt_ig_frac'] = df['mt_ig'] / df['total_ig']
        # Guard genuine 0/0 (all retained IG exactly zero for a row); rare post-filter.
        df['mt_ig_frac'] = df['mt_ig_frac'].fillna(0)
    else:
        logger.warning("Only scalar 'mt_ig' is available. Per-feature IG analysis will be skipped, and fraction defaults to 1.0.")
        df['total_ig'] = df['mt_ig']
        df['mt_ig_frac'] = 1.0

    # Top-50 ranked tables (independent of --rank-by).
    # Note: all *_ig columns are already absolute magnitudes at this point,
    # so sorting mt_ig descending gives the highest methylation saliency.
    print_top_pairs(
        df, sort_col='mt_ig', ascending=False,
        title="TOP 50 PAIRS BY METHYLATION SALIENCY (|mt_ig|)",
        display_cols=['mt_id', 'gt_id', 'region', 'mt_ig', 'mt_ig_frac', 'mt_est', 'p_boot'],
        n=50, out_dir=args.out_dir, filename="top50_by_mt_saliency.csv",
    )

    if 'p_boot' in df.columns:
        print_top_pairs(
            df, sort_col='p_boot', ascending=True,
            title="TOP 50 PAIRS BY BOOTSTRAP P-VALUE (p_boot)",
            display_cols=['mt_id', 'gt_id', 'region', 'p_boot', 'mt_ig', 'mt_ig_frac', 'mt_est'],
            n=50, out_dir=args.out_dir, filename="top50_by_bootstrap_pvalue.csv",
        )
    else:
        logger.warning("'p_boot' column not present; skipping bootstrap p-value table.")

    # Sort
    asc = False
    if args.rank_by == 'p_boot':
        asc = True
        sort_col = 'p_boot'
    elif args.rank_by in ['mt_t', 'mt_est']:
        df['abs_rank'] = df[args.rank_by].abs()
        sort_col = 'abs_rank'
    else:
        sort_col = args.rank_by

    df = df.sort_values(by=sort_col, ascending=asc).reset_index(drop=True)

    # A. Console Summary Report
    print("=" * 60)
    print("SALIENCY EVALUATION REPORT")
    print("=" * 60)
    print(f"Total pairs analyzed: {len(df)}")

    if has_covar_ig:
        print("\n--- Methylation Saliency Fraction Distribution ---")
        fracs = df['mt_ig_frac']
        print(f"Min: {fracs.min():.4f}")
        print(f"Max: {fracs.max():.4f}")
        print(f"Mean: {fracs.mean():.4f}")
        print(f"Median: {fracs.median():.4f}")
        print(f"Q1 (25%): {fracs.quantile(0.25):.4f}")
        print(f"Q3 (75%): {fracs.quantile(0.75):.4f}")
    else:
        print("\n--- Saliency Magnitude Distribution (|mt_ig|) ---")
        mags = df['mt_ig']
        print(f"Min: {mags.min():.4f}")
        print(f"Max: {mags.max():.4f}")
        print(f"Mean: {mags.mean():.4f}")
        print(f"Median: {mags.median():.4f}")

    print("\n--- Methylation Saliency Fraction by Rank Bands ---")
    overall_mean = df['mt_ig_frac'].mean()
    print(f"Overall Mean Fraction: {overall_mean:.4f}")

    for start, end in windows:
        if start > len(df):
            logger.warning(f"Window {start}-{end} exceeds total pairs ({len(df)}). Skipping.")
            continue
        actual_end = min(end, len(df))
        band = df.iloc[start-1:actual_end]
        band_mean = band['mt_ig_frac'].mean()
        print(f"Rank {start}-{actual_end}: {band_mean:.4f}")

    print("\n--- Breakdown by Region ---")
    region_means = df.groupby('region')['mt_ig_frac'].mean()
    region_counts = df.groupby('region').size()
    for r in region_means.index:
        print(f"{r}: {region_means[r]:.4f} (n={region_counts[r]})")

    if has_covar_ig:
        print("\n--- Mean Saliency Proportion per Feature Class ---")
        print("(over denominator features; these proportions sum to ~1.0)")
        for col in denom_cols:
            prop = (df[col] / df['total_ig']).fillna(0).mean()
            print(f"{col}: {prop:.4f}")
        excluded = [c for c in ig_cols if c not in denom_cols]
        if excluded:
            print(f"Excluded from denominator (not in the sum above): {', '.join(excluded)}")
        print("Note: If covariate saliency is low, consider verifying if input scales are comparable.")

        # Decisive scale-vs-biology diagnostic: input MAD next to IG attribution.
        report_input_scales(df, denom_cols, args.covariates_csv,
                            meth_csv=args.methylation_csv, out_dir=args.out_dir)

    # Decay/inflection products are computed on series sorted by the quantity
    # being analyzed (descending magnitude), NOT by --rank-by. This keeps each
    # curve monotone in its own quantity regardless of how pairs were ranked
    # elsewhere, so the knee is always interpretable. region is carried along
    # so the fraction curve can still be colored by region in its own order.
    mag_curve = (df[['mt_ig', 'region']]
                 .sort_values('mt_ig', ascending=False)
                 .reset_index(drop=True))
    if has_covar_ig:
        frac_curve = (df[['mt_ig_frac', 'region']]
                      .sort_values('mt_ig_frac', ascending=False)
                      .reset_index(drop=True))

    print("\n--- Inflection Point Analysis ---")
    print("(curves sorted by descending magnitude, independent of --rank-by)")
    knee, method_used = detect_inflection(mag_curve['mt_ig'].values, args.inflection_method)
    if knee is not None:
        print(f"Magnitude curve (|mt_ig|) inflection point at rank {knee + 1} (method: {method_used})")
    else:
        print("Magnitude curve inflection point could not be detected.")

    if has_covar_ig:
        knee_frac, method_used_frac = detect_inflection(frac_curve['mt_ig_frac'].values, args.inflection_method)
        if knee_frac is not None:
            print(f"Fraction curve (mt_ig_frac) inflection point at rank {knee_frac + 1} (method: {method_used_frac})")
        else:
            print("Fraction curve inflection point could not be detected.")

    # B & C. Plots
    # Create the stacked proportional chart for rank windows
    if has_covar_ig:
        print("\nGenerating Rank-Windowed Stacked Proportional Saliency Plots...")
        for start, end in windows:
            if start > len(df):
                continue
            actual_end = min(end, len(df))
            band = df.iloc[start-1:actual_end].copy()

            # Recalculate proportions locally for plotting (denominator features
            # only, so each stacked bar sums to ~1.0).
            prop_columns = []
            for col in denom_cols:
                prop_col = f"{col}_prop"
                band[prop_col] = band[col] / band['total_ig']
                prop_columns.append(prop_col)

            band['locus_pair'] = band['mt_id'] + " - " + band['gt_id']
            plot_df = band.set_index('locus_pair')[prop_columns]
            plot_df = plot_df.iloc[::-1] # Reverse to match #1 at top

            import matplotlib as mpl
            colors = []
            cmap = mpl.colormaps.get_cmap('Pastel1') if hasattr(mpl.colormaps, 'get_cmap') else plt.cm.Pastel1
            covar_idx = 0
            for col in prop_columns:
                if col == 'mt_ig_prop':
                    colors.append('darkblue')
                else:
                    colors.append(cmap(covar_idx % 9))
                    covar_idx += 1

            fig, ax = plt.subplots(figsize=(12, max(6, (actual_end - start) * 0.2)))
            plot_df.plot.barh(stacked=True, color=colors, width=0.8, ax=ax)

            plt.title(f"Stacked Proportional Saliency Profile (Ranks {start}-{actual_end})")
            plt.xlabel("Proportion of Total Saliency")
            plt.ylabel("Locus Pair (CpG - Gene)")

            handles, labels = ax.get_legend_handles_labels()
            cleaned_labels = [label.replace('_ig_prop', '') for label in labels]
            ax.legend(handles, cleaned_labels, title="Features", loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.tight_layout()
            out_file = os.path.join(args.out_dir, f"saliency_profile_ranks_{start}_{actual_end}.png")
            plt.savefig(out_file)
            plt.close()

    # Histogram
    if has_covar_ig:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['mt_ig_frac'], bins=50)
        plt.title("Distribution of Methylation Saliency Fraction")
        plt.xlabel("Methylation Saliency Fraction")
        plt.ylabel("Count")
        plt.savefig(os.path.join(args.out_dir, "saliency_fraction_hist.png"))
        plt.close()

    # Scatter vs effect size
    if has_covar_ig and ('mt_est' in df.columns or 'mt_t' in df.columns):
        effect_col = 'mt_est' if 'mt_est' in df.columns else 'mt_t'
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[effect_col].abs(), y=df['mt_ig_frac'], alpha=0.3)
        plt.title(f"Methylation Saliency Fraction vs |{effect_col}|")
        plt.xlabel(f"|{effect_col}|")
        plt.ylabel("Methylation Saliency Fraction")
        plt.savefig(os.path.join(args.out_dir, f"saliency_vs_{effect_col}.png"))
        plt.close()

    # Effect-vs-saliency diagnostics: MAD-colored scatter, |mt_est| vs MAD, and
    # the standardized-effect view. Explains why high |mt_est| sits on the
    # fraction floor (variance-inflated coefficients on near-invariant probes).
    if has_covar_ig and 'mt_est' in df.columns:
        plot_effect_diagnostics(df, args.out_dir, df_resid=args.df,
                                meth_csv=args.methylation_csv,
                                plot_sample=args.plot_sample)

    # Box plot by region
    if has_covar_ig:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='region', y='mt_ig_frac', data=df)
        plt.title("Methylation Saliency Fraction by Region")
        plt.xlabel("Region")
        plt.ylabel("Methylation Saliency Fraction")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "saliency_fraction_by_region.png"))
        plt.close()

    # Scree-style decay curves.
    # Each curve uses its own magnitude-sorted series (built above), so the
    # rank axis is "rank by descending magnitude of that quantity" and is
    # decoupled from --rank-by.

    # Curve 1: Fraction vs Rank
    if has_covar_ig:
        frac_ranks = np.arange(1, len(frac_curve) + 1)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=frac_ranks, y=frac_curve['mt_ig_frac'], hue=frac_curve['region'],
                        alpha=0.5, edgecolor=None, s=10)
        if knee_frac is not None:
            plt.axvline(x=knee_frac + 1, color='r', linestyle='--', label=f'Inflection ({method_used_frac}): {knee_frac + 1}')
        plt.xscale('log')
        plt.title("Methylation Saliency Fraction vs Rank")
        plt.xlabel("Rank (by descending mt_ig_frac, log scale)")
        plt.ylabel("Methylation Saliency Fraction")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.figtext(0.01, 0.01, "Note: Structure describes candidate set, not genome-wide distribution.", fontsize=8)
        plt.savefig(os.path.join(args.out_dir, "saliency_fraction_decay_curve.png"))
        plt.close()

    # Curve 2: Magnitude vs Rank
    mag_ranks = np.arange(1, len(mag_curve) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(mag_ranks, mag_curve['mt_ig'].values, color='darkblue')
    if knee is not None:
        plt.axvline(x=knee + 1, color='r', linestyle='--', label=f'Inflection ({method_used}): {knee + 1}')
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Saliency Magnitude (|mt_ig|) vs Rank")
    plt.xlabel("Rank (by descending |mt_ig|, log scale)")
    plt.ylabel("Magnitude (|mt_ig|) (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.figtext(0.01, 0.01, "Note: Structure describes candidate set, not genome-wide distribution.", fontsize=8)
    plt.savefig(os.path.join(args.out_dir, "saliency_magnitude_decay_curve.png"))
    plt.close()

    print("\nEvaluation completed. Plots saved to:", args.out_dir)

if __name__ == "__main__":
    main()
