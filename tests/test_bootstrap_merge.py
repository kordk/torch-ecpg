"""Characterization of the Stage-9 join in tecpg/bootstrap.py.

The bootstrap covers a small candidate subset of the catalog, and its results
are left-joined onto the master. This file pins what that join does today,
including one behaviour that is a defect: IG columns present on the master are
dropped before the join and repopulated only for the bootstrapped pairs, so
they end up null on every unbootstrapped row. That is characterized here rather
than fixed, so the change that fixes it has something to move against.
"""
import numpy as np
import pandas as pd
import pytest

from tecpg.bootstrap import BOOTSTRAP_RESULT_COLUMNS, merge_bootstrap_into_master

IG_COLUMNS = ["mt_ig", "Exp_PC1_ig"]


def _master(n=5, with_ig=True):
    df = pd.DataFrame({
        "mt_id": [f"cg{i}" for i in range(n)],
        "gt_id": [f"ENSG{i}" for i in range(n)],
        "mt_est": np.linspace(0.1, 0.5, n),
        "precise_mt_p": np.linspace(1e-9, 1e-4, n),
    })
    if with_ig:
        df["mt_ig"] = np.linspace(1.0, 5.0, n)
        df["Exp_PC1_ig"] = np.linspace(10.0, 50.0, n)
    return df


def _res(indices, with_ig=True):
    n = len(indices)
    df = pd.DataFrame({
        "mt_id": [f"cg{i}" for i in indices],
        "gt_id": [f"ENSG{i}" for i in indices],
        "mt_est_boot_mean": np.full(n, 0.2),
        "mt_est_boot_std": np.full(n, 0.02),
        "ci_low": np.full(n, 0.1),
        "ci_high": np.full(n, 0.3),
        "p_boot": np.full(n, 0.001),
        "degenerate_resamples": np.zeros(n, dtype=int),
    })
    if with_ig:
        df["mt_ig"] = np.full(n, 99.0)
        df["Exp_PC1_ig"] = np.full(n, 88.0)
    return df


def test_join_keeps_every_master_row():
    """The join is a left join: the catalog is never subset by it."""
    merged = merge_bootstrap_into_master(_master(), _res([1]), IG_COLUMNS)
    assert len(merged) == 5
    assert list(merged["mt_id"]) == [f"cg{i}" for i in range(5)]


def test_bootstrap_columns_are_null_outside_the_candidate_set():
    """Unbootstrapped rows carry no bootstrap result, by design."""
    merged = merge_bootstrap_into_master(_master(), _res([1, 3]), IG_COLUMNS)
    for col in BOOTSTRAP_RESULT_COLUMNS:
        assert merged.loc[merged["mt_id"] == "cg1", col].notna().all()
        assert merged.loc[merged["mt_id"] == "cg0", col].isna().all()


def test_ig_columns_from_master_are_currently_discarded():
    """CHARACTERIZATION OF A DEFECT, not an endorsement.

    The master carries genome-wide IG for every row. The join drops those
    columns and repopulates them only for bootstrapped pairs, so an
    unbootstrapped row loses a value it already had.
    """
    master = _master()
    assert master["mt_ig"].notna().all()

    merged = merge_bootstrap_into_master(master, _res([1]), IG_COLUMNS)

    assert merged.loc[merged["mt_id"] == "cg1", "mt_ig"].iloc[0] == 99.0
    assert merged.loc[merged["mt_id"] == "cg0", "mt_ig"].isna().all()
    assert merged["mt_ig"].notna().sum() == 1
    assert merged["Exp_PC1_ig"].notna().sum() == 1
    assert "mt_ig_boot" not in merged.columns


def test_rejoining_does_not_produce_suffixed_duplicates():
    """A re-run replaces the bootstrap columns rather than adding _x and _y."""
    once = merge_bootstrap_into_master(_master(), _res([1]), IG_COLUMNS)
    twice = merge_bootstrap_into_master(once, _res([1]), IG_COLUMNS)
    assert list(once.columns) == list(twice.columns)
    for col in list(twice.columns):
        assert not col.endswith("_x") and not col.endswith("_y")


def test_master_without_ig_columns_takes_them_from_the_bootstrap():
    """When the mapper wrote no IG, the bootstrap values are the only ones."""
    merged = merge_bootstrap_into_master(_master(with_ig=False), _res([1]), IG_COLUMNS)
    assert "mt_ig" in merged.columns
    assert merged["mt_ig"].notna().sum() == 1


def test_id_columns_are_compared_as_strings():
    """A master with integer-like ids still joins against string ids."""
    master = _master()
    master["mt_id"] = [str(i) for i in range(5)]
    res = _res([1])
    res["mt_id"] = [1]
    merged = merge_bootstrap_into_master(master, res, IG_COLUMNS)
    assert merged.loc[merged["mt_id"] == "1", "p_boot"].notna().all()


def test_inputs_are_not_mutated():
    """The caller's frames survive the join unchanged.

    Both frames carry integer ids and no droppable column, so the string cast
    inside the join would be visible in the caller's data if it were applied
    in place.
    """
    master = _master(with_ig=False)
    master["mt_id"] = list(range(5))
    res = _res([1], with_ig=False)
    res["mt_id"] = [1]
    master_before = master.copy()
    res_before = res.copy()
    merge_bootstrap_into_master(master, res, [])
    pd.testing.assert_frame_equal(master, master_before)
    pd.testing.assert_frame_equal(res, res_before)
