import pytest
import pandas as pd
import numpy as np
from tecpg.permute import _normalize_annotations, tecpg_mlr_qr_permute
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

def test_end_to_end_permute(cli_shaped_annotated_fixture, tmp_path):
    """6. END-TO-END"""
    M, G, C, M_annot, G_annot = cli_shaped_annotated_fixture(10, 5, 5)

    output_file = str(tmp_path / "output.csv")

    logger = Logger()

    # Running permute
    tecpg_mlr_qr_permute(
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

    # Expected rows: 4 * 4 = 16 (since 1 dropped from each axis)
    assert len(df) == 16

    assert set(df.columns) == {'mt_id', 'gt_id', 'mt_t', 'perm_mt_p', 'seed', 'n_perm'}
    assert df['perm_mt_p'].min() > 0
    assert df['perm_mt_p'].max() <= 1
    assert df['perm_mt_p'].isna().sum() == 0


























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
            result3 = runner.invoke(cli, ['--root-path', cwd, 'run', 'mlr', '--mlr-method', 'qr_permute', '--all'])
            assert result3.exit_code == 0, f"Failed: {result3.exception}"
            assert isinstance(mock_qr_permute.call_args.kwargs['M_annot'], pd.DataFrame)
            assert isinstance(mock_qr_permute.call_args.kwargs['G_annot'], pd.DataFrame)
            assert mock_qr_permute.call_args.kwargs['output_p_threshold'] is None

            # Test 4: qr_permute + --cis + --output-p-threshold
            result4 = runner.invoke(cli, ['--root-path', cwd, 'run', 'mlr', '--mlr-method', 'qr_permute', '--cis', '--output-p-threshold', '0.05'])
            assert result4.exit_code == 0, f"Failed: {result4.exception}"
            assert isinstance(mock_qr_permute.call_args.kwargs['M_annot'], pd.DataFrame)
            assert isinstance(mock_qr_permute.call_args.kwargs['G_annot'], pd.DataFrame)
            assert mock_qr_permute.call_args.kwargs['output_p_threshold'] == 0.05
