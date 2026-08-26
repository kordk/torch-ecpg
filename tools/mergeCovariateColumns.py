#!/usr/bin/env python3
"""mergeCovariateColumns.py — merge selected sidecar columns into a covariate matrix.

Several preprocessing stages produce per-sample scores that may become
covariates: ancestry components from ancestry_probes_report.py, surrogate
variables, latent factors. This tool merges a named subset of such a sidecar
into a covariate CSV, optionally renaming the columns, and refuses rather than
guesses when the join does not line up.

The tool is deliberately dumb about which columns are worth merging. Selecting
rs_PC1/rs_PC2 over gap_PC1/gap_PC2 is a per-dataset judgement made in the
pipeline configuration, not a rule that belongs in a shared tool.

Guarantees:
  - Sample set is preserved exactly. A sidecar covering more samples than the
    covariate file is fine (the extras are ignored); a sidecar missing any
    covariate sample is an error, not a silent inner join.
  - Existing columns are never overwritten. A name collision is an error.
  - Untouched columns are bit-identical: read uses float_precision='round_trip',
    because pandas' default C parser shifts roughly 15% of float64 values by one
    ULP on read, so a pass-through stage would otherwise perturb every column.

Usage:
  python3 tools/mergeCovariateColumns.py \\
      --covariates data_mesa/C_orig.csv \\
      --sidecar    data_mesa/ancestry_scores.csv \\
      --columns    rs_PC1,rs_PC2 \\
      --rename     Anc_PC1,Anc_PC2 \\
      --output     data_mesa/C_orig.anc.csv
"""
import argparse
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

READ_KW = {'index_col': 0, 'float_precision': 'round_trip'}


class MergeError(Exception):
    """Raised when the merge cannot be performed as requested."""


def merge_columns(cov: pd.DataFrame, side: pd.DataFrame, columns, rename=None):
    """Return cov with the named sidecar columns appended. Never mutates cov."""
    missing = [c for c in columns if c not in side.columns]
    if missing:
        raise MergeError(
            f"Sidecar is missing requested column(s) {missing}. "
            f"Columns present: {list(side.columns)}")

    if rename is None:
        rename = list(columns)
    if len(rename) != len(columns):
        raise MergeError(
            f"--rename has {len(rename)} name(s) for {len(columns)} column(s); "
            "supply one new name per selected column.")

    collisions = [n for n in rename if n in cov.columns]
    if collisions:
        raise MergeError(
            f"Output column name(s) {collisions} already exist in the covariate "
            "file. This tool never overwrites an existing column; choose "
            "different --rename targets.")

    absent = [s for s in cov.index if s not in side.index]
    if absent:
        show = absent[:5]
        raise MergeError(
            f"{len(absent)} covariate sample(s) are absent from the sidecar "
            f"(first few: {show}). Refusing to merge: an inner join here would "
            "silently drop samples. Check that the sidecar was generated from "
            "the same cohort and that sample IDs use the same format.")

    extra = len(side.index) - len(cov.index)
    if extra > 0:
        logger.info("Sidecar covers %d sample(s) beyond the covariate file; "
                    "these are ignored.", extra)

    out = cov.copy()
    for src, dst in zip(columns, rename):
        out[dst] = side.loc[cov.index, src].to_numpy(dtype=float)
        if pd.isna(out[dst]).all():
            raise MergeError(
                f"Merged column {dst!r} is entirely missing after alignment. "
                "This usually means the sample IDs matched only by coincidence "
                "of ordering; check the index of both files.")
        n_na = int(pd.isna(out[dst]).sum())
        if n_na:
            raise MergeError(
                f"Merged column {dst!r} has {n_na} missing value(s). A covariate "
                "with missing values needs an explicit decision rather than a "
                "default here.")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--covariates', required=True, help='Covariate CSV, samples in rows')
    p.add_argument('--sidecar', required=True, help='CSV of per-sample scores')
    p.add_argument('--columns', required=True,
                   help='Comma-separated sidecar columns to merge')
    p.add_argument('--rename', default=None,
                   help='Comma-separated output names, one per selected column')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--report', default=None, help='Optional JSON report path')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    for path, label in ((args.covariates, 'Covariate file'),
                        (args.sidecar, 'Sidecar file')):
        if not os.path.exists(path):
            logger.error('%s not found: %s', label, path)
            return 2

    columns = [c.strip() for c in args.columns.split(',') if c.strip()]
    rename = ([c.strip() for c in args.rename.split(',') if c.strip()]
              if args.rename else None)
    if not columns:
        logger.error('--columns selected nothing.')
        return 1

    cov = pd.read_csv(args.covariates, **READ_KW)
    side = pd.read_csv(args.sidecar, **READ_KW)
    cov.index = cov.index.astype(str)
    side.index = side.index.astype(str)
    logger.info('Covariates %s: %d samples x %d columns',
                args.covariates, cov.shape[0], cov.shape[1])
    logger.info('Sidecar    %s: %d samples x %d columns',
                args.sidecar, side.shape[0], side.shape[1])

    try:
        out = merge_columns(cov, side, columns, rename)
    except MergeError as exc:
        logger.error('%s', exc)
        return 1

    names = rename or columns
    for src, dst in zip(columns, names):
        logger.info('  merged %s -> %s', src, dst)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    out.to_csv(args.output)
    logger.info('Wrote %s: %d samples x %d columns',
                args.output, out.shape[0], out.shape[1])

    if args.report:
        payload = {
            'covariates': args.covariates,
            'sidecar': args.sidecar,
            'output': args.output,
            'columns_merged': dict(zip(columns, names)),
            'n_samples': int(out.shape[0]),
            'n_columns_in': int(cov.shape[1]),
            'n_columns_out': int(out.shape[1]),
            'columns_out': list(out.columns),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or '.',
                    exist_ok=True)
        with open(args.report, 'w') as fh:
            json.dump(payload, fh, indent=2)
        logger.info('Wrote %s', args.report)

    return 0


if __name__ == '__main__':
    sys.exit(main())
