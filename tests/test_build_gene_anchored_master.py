import pytest
import pandas as pd

from tools.build_gene_anchored_master import assemble_master


def test_assemble_dedup_union():
    cis_df = pd.DataFrame({
        'mt_id': ['mt1', 'mt2', 'mt3'],
        'gt_id': ['gt1', 'gt2', 'gt3'],
        'mt_t': [1.0, 2.0, 3.0],
        'mt_p': [0.1, 0.2, 0.3]
    })

    res_df = pd.DataFrame({
        'mt_id': ['mt2', 'mt3', 'mt4'],
        'gt_id': ['gt2', 'gt3', 'gt4'],
        'mt_t': [2.0005, 3.0, 4.0],
        'mt_p': [0.2, 0.3, 0.4]
    })

    # max delta is 0.0005 for mt2/gt2
    assembled = assemble_master(cis_df, res_df, mt_t_atol=1e-3)

    assert len(assembled) == 4

    # Check that it deduped correctly
    mt_ids = set(assembled['mt_id'])
    assert mt_ids == {'mt1', 'mt2', 'mt3', 'mt4'}

    # Check that both keep mt_p since both have it
    assert 'mt_p' in assembled.columns


def test_mt_t_disagreement_fails_closed():
    cis_df = pd.DataFrame({
        'mt_id': ['mt1'],
        'gt_id': ['gt1'],
        'mt_t': [1.0]
    })

    res_df = pd.DataFrame({
        'mt_id': ['mt1'],
        'gt_id': ['gt1'],
        'mt_t': [1.5]
    })

    # 0.5 > 1e-3, should fail
    with pytest.raises(ValueError, match="Overlap disagreement"):
        assemble_master(cis_df, res_df, mt_t_atol=1e-3)

    # Widening the tolerance should pass
    assembled = assemble_master(cis_df, res_df, mt_t_atol=1.0)
    assert len(assembled) == 1


def test_missing_required_column_fails_closed():
    cis_df = pd.DataFrame({
        'mt_id': ['mt1'],
        'gt_id': ['gt1'],
        'mt_t': [1.0]
    })

    res_df = pd.DataFrame({
        'mt_id': ['mt2'],
        'gt_id': ['gt2']
        # missing mt_t
    })

    with pytest.raises(
            ValueError,
            match="Source 'reservoir' is missing required columns"):
        assemble_master(cis_df, res_df, mt_t_atol=1e-3)


def test_reservoir_csv_input():
    cis_df = pd.DataFrame({
        'mt_id': ['mt1'],
        'gt_id': ['gt1'],
        'mt_t': [1.0],
        'mt_p': [0.1]
    })

    # Emulate CSV input lacking mt_p
    res_df = pd.DataFrame({
        'mt_id': ['mt2'],
        'gt_id': ['gt2'],
        'mt_t': [2.0]
    })

    assembled = assemble_master(cis_df, res_df, mt_t_atol=1e-3)

    assert len(assembled) == 2
    # mt_p must be completely dropped because it's missing in res_df
    assert 'mt_p' not in assembled.columns


def test_empty_result_fails_closed():
    empty_cis = pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t', 'mt_p'])
    empty_res = pd.DataFrame(columns=['mt_id', 'gt_id', 'mt_t'])

    with pytest.raises(
            ValueError,
            match="Assembled master dataframe is empty"):
        assemble_master(empty_cis, empty_res, mt_t_atol=1e-3)
