import json
import os
import subprocess
import sys
import pandas as pd
import numpy as np

from tools.summarize_permute import build_summary_text

def test_build_summary_text_populated():
    report = {
        "metadata": {
            "n_pairs_input": 100,
            "n_pairs_dropped_unmappable_chrom": 10,
            "n_pairs_scored": 90,
            "n_cis": 40,
            "n_trans": 50
        },
        "arms": {
            "calibration": {
                "all": {"bulk_spearman_corr": 0.99, "bulk_median_abs_log_ratio": 0.01},
                "cis": {"bulk_spearman_corr": 0.98, "bulk_median_abs_log_ratio": 0.02},
                "trans": {"bulk_spearman_corr": 0.97, "bulk_median_abs_log_ratio": 0.03}
            },
            "null_sanity": {
                "lambda_trans": 1.05,
                "lambda_cis": 1.10
            },
            "stratify_decision": {
                "recommendation": "stratification_warranted",
                "delta_median_log10_ratio": 0.5,
                "test_p": 1e-4,
                "ks_p": 1e-5
            }
        }
    }

    text = build_summary_text(report)
    assert "stratification_warranted" in text, "Verdict should be included in the text"
    assert "0.0200" in text, "Cis median difference should be formatted and included"
    assert "1.05" in text, "lambda_trans should be included"

def test_build_summary_text_missing():
    report = {
        "metadata": {
            "n_pairs_input": 100,
            "n_pairs_dropped_unmappable_chrom": 0,
            "n_pairs_scored": 100,
            "n_cis": 0,
            "n_trans": 0
        },
        "arms": {
            "calibration": {},
            "null_sanity": {},
            "stratify_decision": {
                "status": "skipped_insufficient_data"
            }
        }
    }
    text = build_summary_text(report)
    assert "skipped due to insufficient data" in text
    assert "bulk not computed" in text

def test_smoke_summarize_permute(tmp_path):
    report_path = tmp_path / "eval_permute_report.json"
    perm_path = tmp_path / "permutation_results.parquet"
    out_dir = tmp_path / "out"
    m_annot_path = tmp_path / "M.bed6"
    g_annot_path = tmp_path / "G.bed6"

    report_data = {
        "metadata": {"n_pairs_input": 10},
        "arms": {
            "calibration": {"all": {"bulk_spearman_corr": 0.9, "bulk_median_abs_log_ratio": 0.1}},
            "null_sanity": {"lambda_trans": 1.0, "lambda_cis": 1.0},
            "stratify_decision": {"recommendation": "single_global_null_adequate", "delta_median_log10_ratio": 0.0, "test_p": 0.5, "ks_p": 0.5}
        }
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f)

    df = pd.DataFrame({
        "mt_id": ["m1", "m2", "m3"],
        "gt_id": ["g1", "g2", "g3"],
        "mt_t": [1.0, 2.0, 3.0],
        "perm_mt_p": [0.1, 0.05, 0.01]
    })
    df.to_parquet(perm_path)

    m_annot = pd.DataFrame({
        0: ["chr1", "chr2", "chr3"],
        1: [1, 2, 3],
        2: [10, 20, 30],
        3: ["m1", "m2", "m3"],
        4: [0, 0, 0],
        5: ["+", "+", "+"]
    })
    m_annot.to_csv(m_annot_path, sep="\t", header=False, index=False)

    g_annot = pd.DataFrame({
        0: ["chr1", "chr2", "chr4"],
        1: [1, 2, 3],
        2: [10, 20, 30],
        3: ["g1", "g2", "g3"],
        4: [0, 0, 0],
        5: ["+", "+", "+"]
    })
    g_annot.to_csv(g_annot_path, sep="\t", header=False, index=False)

    cmd = [
        sys.executable, "tools/summarize_permute.py",
        "--perm-output", str(perm_path),
        "--report", str(report_path),
        "--df", "100",
        "--m-annot", str(m_annot_path),
        "--g-annot", str(g_annot_path),
        "--out-dir", str(out_dir)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    assert (out_dir / "permute_vs_analytic_summary.md").exists()
    assert (out_dir / "qq_perm_vs_analytic.png").exists()
    assert (out_dir / "dist_overlap_p.png").exists()
    assert (out_dir / "dist_tstat.png").exists()
