import json
import os
import subprocess
import sys
import pandas as pd
import numpy as np

from tools.summarize_permute import build_summary_text, assign_family

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

def test_build_summary_text_per_region():
    report = {
        "metadata": {
            "n_pairs_input": 100,
            "n_pairs_dropped_unmappable_chrom": 0,
            "n_pairs_scored": 100,
            "n_by_region": {
                "TRANS": 50, "DISTAL5": 10, "DISTAL3": 10,
                "CIS5": 5, "PROMOTER": 5, "GENEBODY": 10, "CIS3": 10
            }
        },
        "arms": {
            "stratify_decision": {
                "mode": "per_region",
                "per_region": {
                    "TRANS": {"status": "reference", "n_bulk": 50},
                    "DISTAL5": {"status": "ok", "n_bulk": 10, "median_log10_ratio": 0.0027, "delta_vs_trans": 0.01, "mw_p": 0.5, "lambda": 1.01},
                    "DISTAL3": {"status": "ok", "n_bulk": 10, "median_log10_ratio": 0.003, "delta_vs_trans": 0.01, "mw_p": 0.5, "lambda": 1.02},
                    "CIS5": {"status": "insufficient_data", "n_bulk": 5},
                    "PROMOTER": {"status": "insufficient_data", "n_bulk": 5},
                    "GENEBODY": {"status": "insufficient_data", "n_bulk": 10},
                    "CIS3": {"status": "insufficient_data", "n_bulk": 10}
                },
                "recommendation": "insufficient_near_gene_coverage",
                "divergent_regions": []
            }
        }
    }

    text = build_summary_text(report)

    for r in ["TRANS", "DISTAL5", "DISTAL3", "CIS5", "PROMOTER", "GENEBODY", "CIS3"]:
        assert r in text, f"Region {r} should be in the summary text"

    assert "insufficient_near_gene_coverage" in text
    assert "0.0027" in text

    assert "same-chromosome" not in text
    assert "(Cis:" not in text

def test_assign_family_region_consumed_no_crash(monkeypatch):
    df = pd.DataFrame({
        "mt_id": ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
        "gt_id": ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        "mt_t": [1.0] * 8,
        "perm_mt_p": [0.1] * 8,
        "region": ["TRANS", "DISTAL5", "CIS5", "PROMOTER", "GENEBODY", "CIS3", "DISTAL3", None]
    })

    def raise_if_called(*args, **kwargs):
        raise AssertionError("label_strata must not be called on the region path")

    monkeypatch.setattr("eval_permute.label_strata", raise_if_called)

    # Missing annotations doesn't matter because region is present
    df_out, strat_mode = assign_family(df, "dummy_m.bed6", "dummy_g.bed6")

    assert strat_mode == 'region'
    assert len(df_out) == 7  # None dropped
    assert set(df_out['family'].unique()).issubset({"trans", "distal", "near_gene"})

def test_assign_family_fallback_tolerates_missing_annotations(tmp_path):
    df = pd.DataFrame({
        "mt_id": ["m1", "m_missing"],
        "gt_id": ["g1", "g_missing"]
    })

    m_annot_path = tmp_path / "M.bed6"
    g_annot_path = tmp_path / "G.bed6"

    m_annot = pd.DataFrame({
        'chrom': ["chr1"], 'chromStart': [1], 'chromEnd': [10],
        'name': ["m1"], 'score': [0], 'strand': ["+"]
    })
    m_annot.to_csv(m_annot_path, sep="\t", header=True, index=False)

    g_annot = pd.DataFrame({
        'chrom': ["chr1"], 'chromStart': [1], 'chromEnd': [10],
        'name': ["g1"], 'score': [0], 'strand': ["+"]
    })
    g_annot.to_csv(g_annot_path, sep="\t", header=True, index=False)

    df_out, strat_mode = assign_family(df, m_annot_path, g_annot_path)

    assert strat_mode == 'family2'
    assert len(df_out) == 1
    assert "m_missing" not in df_out["mt_id"].values
    assert set(df_out['family'].unique()).issubset({"trans", "near_gene"})

def test_end_to_end_region_parquet_exit0(tmp_path):
    report_path = tmp_path / "eval_permute_report.json"
    perm_path = tmp_path / "permutation_results.parquet"
    out_dir = tmp_path / "out"
    m_annot_path = tmp_path / "M.bed6"
    g_annot_path = tmp_path / "G.bed6"

    report_data = {
        "metadata": {"n_pairs_input": 10},
        "arms": {
            "calibration": {"all": {"bulk_spearman_corr": 0.9, "bulk_median_abs_log_ratio": 0.1}}
        }
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f)

    df = pd.DataFrame({
        "mt_id": ["m1", "m2"],
        "gt_id": ["g1", "g2"],
        "mt_t": [1.0, 2.0],
        "perm_mt_p": [0.1, 0.05],
        "region": ["TRANS", "CIS5"]
    })
    df.to_parquet(perm_path)

    # Incomplete annotations - m2/g2 missing
    m_annot = pd.DataFrame({
        'chrom': ["chr1"], 'chromStart': [1], 'chromEnd': [10],
        'name': ["m1"], 'score': [0], 'strand': ["+"]
    })
    m_annot.to_csv(m_annot_path, sep="\t", header=True, index=False)

    g_annot = pd.DataFrame({
        'chrom': ["chr2"], 'chromStart': [1], 'chromEnd': [10],
        'name': ["g1"], 'score': [0], 'strand': ["+"]
    })
    g_annot.to_csv(g_annot_path, sep="\t", header=True, index=False)

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
        'chrom': ["chr1", "chr2", "chr3"],
        'chromStart': [1, 2, 3],
        'chromEnd': [10, 20, 30],
        'name': ["m1", "m2", "m3"],
        'score': [0, 0, 0],
        'strand': ["+", "+", "+"]
    })
    m_annot.to_csv(m_annot_path, sep="\t", header=True, index=False)

    g_annot = pd.DataFrame({
        'chrom': ["chr1", "chr2", "chr4"],
        'chromStart': [1, 2, 3],
        'chromEnd': [10, 20, 30],
        'name': ["g1", "g2", "g3"],
        'score': [0, 0, 0],
        'strand': ["+", "+", "+"]
    })
    g_annot.to_csv(g_annot_path, sep="\t", header=True, index=False)

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
