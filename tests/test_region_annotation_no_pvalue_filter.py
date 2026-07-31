"""Guards for D1: assignRegionToEcpg_parquet.py annotates, it never filters on p.

The module formerly carried `PVALCUTOFF = np.float32(1e-6)` and logged it as
"Using default p-value cutoff", but no code path ever compared against it. These
tests pin the real behaviour (every input row is emitted) and prevent the dead
constant and its misleading log line from returning.
"""
import os
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO_ROOT, "tools", "assignRegionToEcpg_parquet.py")

# Straddles the former 1e-6 cutoff: one row below, two rows above.
FIXTURE_PVALUES = {
    "cgSTRONG": 2.29e-14,   # below 1e-6 -> would have survived a live filter
    "cgWEAK": 7.87e-05,     # above 1e-6 -> would have been dropped
    "cgNULLISH": 4.63e-02,  # above 1e-6 -> would have been dropped
}


@pytest.fixture
def annotated_run(tmp_path):
    """Run the annotator on a 3-row fixture and return (stdout+stderr, output df)."""
    gene_bed = tmp_path / "G.bed6"
    gene_bed.write_text("chr1\t100000\t120000\tGENE1\t0\t+\n")

    meth_bed = tmp_path / "M.bed6"
    meth_bed.write_text(
        "chr1\t110000\t110001\tcgSTRONG\t0\t+\n"
        "chr1\t110500\t110501\tcgWEAK\t0\t+\n"
        "chr1\t111000\t111001\tcgNULLISH\t0\t+\n"
    )

    in_parquet = tmp_path / "in.parquet"
    df = pd.DataFrame(
        {
            "mt_id": list(FIXTURE_PVALUES),
            "gt_id": ["GENE1"] * 3,
            "mt_est": [0.5, 0.3, 0.1],
            "mt_t": [8.0, 4.0, 2.0],
            "mt_p": [1e-14, 7.9e-5, 4.6e-2],
            "precise_mt_p": list(FIXTURE_PVALUES.values()),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), in_parquet)

    out_parquet = tmp_path / "out.parquet"
    proc = subprocess.run(
        [sys.executable, TOOL,
         "-d", str(in_parquet), "-g", str(gene_bed),
         "-m", str(meth_bed), "-o", str(out_parquet)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"annotator failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout + proc.stderr, pq.read_table(out_parquet).to_pandas()


def test_all_rows_survive_regardless_of_pvalue(annotated_run):
    """Load-bearing guard: the annotator emits every row it is given.

    Reinstating the p-value filter the deleted constant implied drops the two
    rows above 1e-6 and turns this red.
    """
    _, out = annotated_run
    assert len(out) == 3
    assert set(out["mt_id"]) == set(FIXTURE_PVALUES)
    # the two rows above the former cutoff are present and annotated
    above = out[out["precise_mt_p"] > 1e-6]
    assert len(above) == 2
    assert above["region"].notna().all()


def test_no_pvalue_exclusions_reported(annotated_run):
    """The exclusion summary must report zero p-value exclusions."""
    log, _ = annotated_run
    assert "p-value filter: 0" in log


def test_module_exposes_no_pvalcutoff_symbol():
    """The dead constant must not return."""
    import tools.assignRegionToEcpg_parquet as A
    assert not hasattr(A, "PVALCUTOFF"), (
        "PVALCUTOFF is back. The annotator does not filter on p; a constant "
        "implying otherwise is misleading (see docs/ecpg-filtering-prioritization.md D1)."
    )


def test_no_cutoff_log_line(annotated_run):
    """The misleading 'Using default p-value cutoff' line must not return."""
    log, _ = annotated_run
    assert "p-value cutoff" not in log
