import pytest
import pandas as pd
import numpy as np
from tecpg.permute import _normalize_annotations, tecpg_mlr_qr_permute, _select_null_population
from tecpg.logger import Logger
from tecpg.cli import mlr
from click.testing import CliRunner
import os

def test_chrom_coding_oracle():
    """1. ORACLE — chrom coding"""
    logger = Logger()
    M_annot = pd.DataFrame({
        'chrom': ['chr19', '19', 19, 19.0, 'chrX', 'X', 'chrY', 'x', float('nan'), 'GL000220.1'],
        'chromStart': range(10)
    }, index=[f'm{i}' for i in range(10)])
    G_annot = pd.DataFrame({
        'chrom': ['chr19'] * 10,
        'chromStart': range(10),
        'strand': ['+'] * 10
    }, index=[f'g{i}' for i in range(10)])
    M = pd.DataFrame(np.random.rand(10, 5), index=M_annot.index)
    G = pd.DataFrame(np.random.rand(10, 5), index=G_annot.index)

    M_annot_n, G_annot_n, M_n, G_n = _normalize_annotations(M_annot, G_annot, M, G, logger)

    assert M_annot_n.loc['m0', 'chrom'] == 19
    assert M_annot_n.loc['m1', 'chrom'] == 19
    assert M_annot_n.loc['m2', 'chrom'] == 19
    assert M_annot_n.loc['m3', 'chrom'] == 19
    assert M_annot_n.loc['m4', 'chrom'] == -1
    assert M_annot_n.loc['m5', 'chrom'] == -1
    assert M_annot_n.loc['m6', 'chrom'] == -2
    assert M_annot_n.loc['m7', 'chrom'] == -1
    assert 'm8' not in M_annot_n.index
    assert 'm9' not in M_annot_n.index

    # integer exactness
    assert pd.api.types.is_integer_dtype(M_annot_n['chrom'].astype(int))

def test_qr_code_parity():
    """2. ORACLE — qr-code parity"""
    logger = Logger()
    M_annot = pd.DataFrame({'chrom': ['chrX', 'chrY'], 'chromStart': [0, 1]}, index=['m1', 'm2'])
    G_annot = pd.DataFrame({'chrom': ['chrX', 'chrY'], 'chromStart': [0, 1], 'strand': ['+', '-']}, index=['g1', 'g2'])
    M = pd.DataFrame(np.random.rand(2, 5), index=M_annot.index)
    G = pd.DataFrame(np.random.rand(2, 5), index=G_annot.index)

    M_annot_n, G_annot_n, M_n, G_n = _normalize_annotations(M_annot, G_annot, M, G, logger)

    assert M_annot_n.loc['m1', 'chrom'] == -1
    assert M_annot_n.loc['m2', 'chrom'] == -2
    assert G_annot_n.loc['g1', 'chrom'] == -1
    assert G_annot_n.loc['g2', 'chrom'] == -2

def test_idempotence(annotated_fixture):
    """3. IDEMPOTENCE"""
    logger = Logger()
    M, G, _, M_annot, G_annot = annotated_fixture(10, 5, 5)

    M_annot_n, G_annot_n, M_n, G_n = _normalize_annotations(M_annot, G_annot, M, G, logger)

    pd.testing.assert_frame_equal(M_annot_n, M_annot, check_names=False)
    pd.testing.assert_frame_equal(G_annot_n, G_annot, check_names=False)
    pd.testing.assert_frame_equal(M_n, M)
    pd.testing.assert_frame_equal(G_n, G)

def test_drop_and_trim(cli_shaped_annotated_fixture):
    """4. DROP + TRIM"""
    logger = Logger()
    M, G, _, M_annot, G_annot = cli_shaped_annotated_fixture(10, 5, 5)

    # 5 M rows, 1 dropped (NaN chrom). 5 G rows, 1 dropped (GL000220.1)
    M_annot_n, G_annot_n, M_n, G_n = _normalize_annotations(M_annot, G_annot, M, G, logger)

    assert len(M_annot_n) == 4
    assert len(G_annot_n) == 4

    assert M_annot_n.index.equals(M_n.index)
    assert G_annot_n.index.equals(G_n.index)

def test_fail_closed():
    """5. FAIL-CLOSED"""
    logger = Logger()
    M_annot = pd.DataFrame({'chrom': [float('nan'), float('nan')], 'chromStart': [0, 1]}, index=['m1', 'm2'])
    G_annot = pd.DataFrame({'chrom': ['chr19', 'chr19'], 'chromStart': [0, 1], 'strand': ['+', '-']}, index=['g1', 'g2'])
    M = pd.DataFrame(np.random.rand(2, 5), index=M_annot.index)
    G = pd.DataFrame(np.random.rand(2, 5), index=G_annot.index)

    with pytest.raises(ValueError, match="Normalization dropped all loci on one or both axes."):
        _normalize_annotations(M_annot, G_annot, M, G, logger)

def test_end_to_end_permute(cli_shaped_annotated_fixture, tmp_path, master_parquet_fixture):
    """6. END-TO-END"""
    # M/G generated here will have 5 rows.
    # cli_shaped drops 1 row on each side -> 4 rows.
    M, G, C, M_annot, G_annot = cli_shaped_annotated_fixture(10, 5, 5)

    # Re-home the NaN-dropping assertion to the null side, because
    # the master parquet determines the *observed* universe size.
    logger = Logger()
    M_annot_n, G_annot_n, M_n, G_n = _normalize_annotations(M_annot, G_annot, M, G, logger)
    null_M, null_G = _select_null_population(M_n, G_n, C, M_annot_n, G_annot_n, 'all', None, None, None,
                                             None, None, 42, logger)
    assert len(null_M) == 4, "NaN chrom dropping must apply to the null M universe."
    assert len(null_G) == 4, "NaN chrom dropping must apply to the null G universe."

    # Generate master parquet from the SAME cli_shaped M/G so the observed IDs match
    # Since regression_full doesn't drop NaNs internally on the observed side,
    # the master will have 5*5=25 rows.
    from tecpg.regression_full import regression_full
    out = regression_full(M, G, C, region='all', p_thresh=None,
                              methylation_only=True, logger=Logger())
    master = out.reset_index()

    assert len(master) == 25, "Master should have all 25 pairs before permute."

    master_parquet = str(tmp_path / 'master.parquet')
    master.to_parquet(master_parquet)

    output_file = str(tmp_path / "output.csv")

    # Running permute
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M, G=G, C=C,
        M_annot=M_annot, G_annot=G_annot,
        region='all',
        permutations=10,
        seed=42,
        output_file=output_file,
        logger=logger
    )

    assert os.path.exists(output_file)
    df = pd.read_csv(output_file)

    # Master has 25 rows. They should all be preserved, but only 16 scored.
    assert len(df) == 25

    scored_count = len(df.dropna(subset=['perm_mt_p']))
    assert scored_count == len(null_M) * len(null_G) == 16

    nan_count = df['perm_mt_p'].isna().sum()
    assert nan_count == 25 - 16 == 9

    assert '_x' not in df.columns and '_y' not in df.columns
    assert 'mt_p' in df.columns

    # Verify that mt_p is strictly preserved from the master
    merged_check = df.merge(master[['mt_id', 'gt_id', 'mt_p']], on=['mt_id', 'gt_id'], suffixes=('', '_master'))
    import numpy as np
    np.testing.assert_allclose(merged_check['mt_p'].values, merged_check['mt_p_master'].values, err_msg="mt_p values must remain unchanged")

    expected_cols = {'mt_id', 'gt_id', 'mt_t', 'mt_p', 'perm_mt_p', 'perm_seed', 'perm_n_perm'}
    assert expected_cols.issubset(set(df.columns))
    assert not any(c.endswith('_x') or c.endswith('_y') for c in df.columns)

    assert df['perm_mt_p'].min() > 0
    assert df['perm_mt_p'].max() <= 1
    assert df['perm_mt_p'].isna().sum() == nan_count

def test_drop_and_trim_trans_mask_oracle(cli_shaped_annotated_fixture):
    """Test decisive alignment assertion on stratum"""
    from tecpg.permute import _compute_trans_mask
    M, G, C, M_annot, G_annot = cli_shaped_annotated_fixture(10, 5, 5)

    logger = Logger()
    M_annot_n, G_annot_n, M_n, G_n = _normalize_annotations(M_annot, G_annot, M, G, logger)

    m_idx = M_n.index.astype(str)
    g_idx = G_n.index.astype(str)

    reported_pairs = pd.MultiIndex.from_product([m_idx, g_idx], names=['mt_id', 'gt_id']).to_frame(index=False)

    trans_mask = _compute_trans_mask(reported_pairs, M_annot_n, G_annot_n, 'trans', window_base=1000, downstream=1000, upstream=1000, logger=logger)

    # Fixture M cycle: chr1 (i=0), chr2 (i=1), chrX (i=2), nan (i=3) [dropped], chr7 (i=4)
    # Fixture G cycle: chr1 (i=0), chr7 (i=1), chrY (i=2), GL000220.1 (i=3) [dropped], chr2 (i=4)
    # Both arrays dropped index 3.
    # M remaining indices: cg001 (chr1), cg002 (chr2), cg003 (chrX), cg005 (chr7).
    # G remaining indices: ILMN_001 (chr1), ILMN_002 (chr7), ILMN_003 (chrY), ILMN_005 (chr2).
    # cg001_ILMN_001 -> cis (chr1 == chr1)
    # cg002_ILMN_005 -> cis (chr2 == chr2)
    # cg005_ILMN_002 -> cis (chr7 == chr7)
    # All others are trans.

    cis_pairs = set([
        (m_idx[0], g_idx[0]),
        (m_idx[1], g_idx[3]),
        (m_idx[3], g_idx[1])
    ])

    expected_trans_mask = np.array([
        (row['mt_id'], row['gt_id']) not in cis_pairs
        for _, row in reported_pairs.iterrows()
    ], dtype=bool)

    np.testing.assert_array_equal(trans_mask, expected_trans_mask)

def test_cli_regression():
    """7. CLI REGRESSION - exact pass-through assertions matrix"""
    from click.testing import CliRunner
    from unittest.mock import patch
    import os
    import json
    from tecpg.cli import cli
    from tecpg.logger import Logger

    runner = CliRunner()
    with runner.isolated_filesystem():
        cwd = os.getcwd()

        # Setup dummy filesystem for CLI
        os.makedirs(os.path.join(cwd, 'data'), exist_ok=True)
        os.makedirs(os.path.join(cwd, 'annot'), exist_ok=True)
        os.makedirs(os.path.join(cwd, 'output'), exist_ok=True)

        # Tiny valid datasets
        pd.DataFrame({'a': [1,2], 'b': [3,4]}, index=['m1', 'm2']).to_csv(os.path.join(cwd, 'data/M.csv'))
        pd.DataFrame({'a': [1,2], 'b': [3,4]}, index=['g1', 'g2']).to_csv(os.path.join(cwd, 'data/G.csv'))
        pd.DataFrame({'c': [1,2]}).to_csv(os.path.join(cwd, 'data/C.csv'), index=False)

        # Minimal BED6 annotations
        bed6_content = "chrom\tchromStart\tchromEnd\tname\tscore\tstrand\nchr1\t1\t2\tm1\t0\t+\nchr1\t1\t2\tm2\t0\t+\n"
        with open(os.path.join(cwd, 'annot/M.bed6'), 'w') as f: f.write(bed6_content)
        bed6_content_g = "chrom\tchromStart\tchromEnd\tname\tscore\tstrand\nchr1\t1\t2\tg1\t0\t+\nchr1\t1\t2\tg2\t0\t+\n"
        with open(os.path.join(cwd, 'annot/G.bed6'), 'w') as f: f.write(bed6_content_g)

        master_pq = os.path.join(cwd, 'dummy.parquet')
        pd.DataFrame({'mt_id': ['m1'], 'gt_id': ['g1'], 'mt_t': [0.0]}).to_parquet(master_pq)
        assert os.path.exists(master_pq)

        # Mock targets
        with patch('tecpg.cli.tecpg_mlr_qr') as mock_qr, \
             patch('tecpg.permute.tecpg_mlr_qr_permute') as mock_qr_permute, \
             patch('tecpg.cli.read_dataframes') as mock_read:

            # Make the dataframes loader return our mocked ones seamlessly without global config messing up
            mock_read.return_value = {
                "M.csv": pd.DataFrame({'a': [1,2], 'b': [3,4]}, index=['m1', 'm2']),
                "G.csv": pd.DataFrame({'a': [1,2], 'b': [3,4]}, index=['g1', 'g2']),
                "C.csv": pd.DataFrame({'c': [1,2]})
            }

            # Test 1: qr + --all
            # The top-level group is 'cli', then 'run', then 'mlr'
            result1 = runner.invoke(cli, ['--root-path', cwd, 'run', 'mlr', '--mlr-method', 'qr', '--all'])
            if result1.exit_code != 0:
                print(result1.output)
            assert result1.exit_code == 0, f"Failed: {result1.exception}"
            assert mock_qr.call_args.kwargs['M_annot'] is None
            assert mock_qr.call_args.kwargs['G_annot'] is None

            # Test 2: qr + --cis
            result2 = runner.invoke(cli, ['--root-path', cwd, 'run', 'mlr', '--mlr-method', 'qr', '--cis'])
            assert result2.exit_code == 0, f"Failed: {result2.exception}"
            assert isinstance(mock_qr.call_args.kwargs['M_annot'], pd.DataFrame)
            assert isinstance(mock_qr.call_args.kwargs['G_annot'], pd.DataFrame)

            # Test 3: qr_permute + --all (The FACT-1 fix)
            result3 = runner.invoke(cli, ['--root-path', cwd, 'run', 'mlr', '--mlr-method', 'qr_permute', '--all', '--master-parquet', master_pq], catch_exceptions=False)
            assert result3.exit_code == 0, f"Failed: {result3.exception}"
            assert isinstance(mock_qr_permute.call_args.kwargs['M_annot'], pd.DataFrame)
            assert isinstance(mock_qr_permute.call_args.kwargs['G_annot'], pd.DataFrame)
            assert mock_qr_permute.call_args.kwargs['output_p_threshold'] is None

            # Test 4: qr_permute + --cis + --output-p-threshold
            result4 = runner.invoke(cli, ['--root-path', cwd, 'run', 'mlr', '--mlr-method', 'qr_permute', '--cis', '--output-p-threshold', '0.05', '--master-parquet', master_pq], catch_exceptions=False)
            assert result4.exit_code == 0, f"Failed: {result4.exception}"
            assert isinstance(mock_qr_permute.call_args.kwargs['M_annot'], pd.DataFrame)
            assert isinstance(mock_qr_permute.call_args.kwargs['G_annot'], pd.DataFrame)
            assert mock_qr_permute.call_args.kwargs['output_p_threshold'] == 0.05


def test_end_to_end_permute_determinism(cli_shaped_annotated_fixture, tmp_path, master_parquet_fixture):
    """8. DETERMINISM TEST"""
    # M/G generated here will have 5 rows.
    # cli_shaped drops 1 row on each side -> 4 rows.
    M, G, C, M_annot, G_annot = cli_shaped_annotated_fixture(10, 5, 5)
    logger = Logger()

    # Generate master parquet aligned with the valid IDs
    from tecpg.regression_full import regression_full
    out = regression_full(M, G, C, region='all', p_thresh=None,
                              methylation_only=True, logger=Logger())
    master = out.reset_index()
    M_annot_n, G_annot_n, _, _ = _normalize_annotations(M_annot, G_annot, M, G, logger)
    master = master[master['mt_id'].isin(M_annot_n.index) & master['gt_id'].isin(G_annot_n.index)].reset_index(drop=True)
    master_parquet = str(tmp_path / 'master.parquet')
    master.to_parquet(master_parquet)

    output_file1 = str(tmp_path / "output1.csv")
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M, G=G, C=C,
        M_annot=M_annot, G_annot=G_annot,
        region='all',
        permutations=10,
        seed=42,
        output_file=output_file1,
        logger=logger
    )
    df1 = pd.read_csv(output_file1)

    output_file2 = str(tmp_path / "output2.csv")
    tecpg_mlr_qr_permute(
        master_parquet=master_parquet,
        M=M, G=G, C=C,
        M_annot=M_annot, G_annot=G_annot,
        region='all',
        permutations=10,
        seed=42,
        output_file=output_file2,
        logger=logger
    )
    df2 = pd.read_csv(output_file2)

    pd.testing.assert_series_equal(df1['perm_mt_p'], df2['perm_mt_p'])
