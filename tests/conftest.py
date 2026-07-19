import pytest
import pandas as pd
from tecpg.test_data import generate_data

@pytest.fixture
def annotated_fixture():
    def _make(sample_size, m_rows, g_rows, seed=42):
        M, G, C, M_annot, G_annot = generate_data(sample_size, m_rows, g_rows,
                                                   annotation=True, seed=seed)
        M_annot = M_annot.set_index("name")[["chrom", "chromStart"]]
        M_annot[["chrom", "chromStart"]] = M_annot[["chrom", "chromStart"]].astype(int)
        G_annot = G_annot.set_index("name")[["chrom", "chromStart", "strand"]]
        G_annot["strand"] = G_annot["strand"].replace({"+": 1, "-": -1})
        G_annot[["chrom", "chromStart", "strand"]] = G_annot[["chrom", "chromStart", "strand"]].astype(int)
        return M, G, C, M_annot, G_annot
    return _make


@pytest.fixture
def cli_shaped_annotated_fixture():
    def _make(sample_size, m_rows, g_rows, seed=42):
        M, G, C, M_annot, G_annot = generate_data(sample_size, m_rows, g_rows,
                                                   annotation=True, seed=seed)

        # Simulating the raw format handed by the CLI parser:
        # strings with chr-prefix, extra columns, some missing/X/Y mappings
        # M chroms: ['chr1', 'chr2', 'chrX', nan, 'chr7'] ... repeating if > 5
        # G chroms: ['chr1', 'chr7', 'chrY', 'GL000220.1', 'chr2'] ... repeating if > 5

        M_chroms_cycle = ['chr1', 'chr2', 'chrX', float('nan'), 'chr7']
        G_chroms_cycle = ['chr1', 'chr7', 'chrY', 'GL000220.1', 'chr2']

        M_annot_raw = pd.DataFrame(index=M_annot['name'])
        M_annot_raw['chrom'] = [M_chroms_cycle[i % 5] for i in range(len(M_annot_raw))]

        M_annot_raw['chromStart'] = range(len(M_annot_raw))
        M_annot_raw['chromEnd'] = range(len(M_annot_raw))
        M_annot_raw['score'] = 0
        M_annot_raw['strand'] = '+'

        G_annot_raw = pd.DataFrame(index=G_annot['name'])
        G_annot_raw['chrom'] = [G_chroms_cycle[i % 5] for i in range(len(G_annot_raw))]

        G_annot_raw['chromStart'] = range(len(G_annot_raw))
        G_annot_raw['chromEnd'] = range(len(G_annot_raw))
        G_annot_raw['score'] = 0
        G_annot_raw['strand'] = ['-'] * len(G_annot_raw)

        return M, G, C, M_annot_raw, G_annot_raw
    return _make


@pytest.fixture
def master_parquet_fixture(annotated_fixture, tmp_path):
    """Mapping output (a master parquet with mt_t) over the M×G universe,
    for the realigned qr_permute consume path."""
    def _make(sample_size=20, m_rows=6, g_rows=5, seed=42, region='all'):
        from tecpg.regression_full import regression_full
        from tecpg.logger import Logger
        M, G, C, M_annot, G_annot = annotated_fixture(sample_size, m_rows, g_rows, seed)
        out = regression_full(M, G, C, region=region, p_thresh=None,
                              methylation_only=True, logger=Logger())
        master = out.reset_index()  # -> columns mt_id, gt_id, mt_est, mt_err, mt_t, mt_p, ...
        path = tmp_path / 'master.parquet'
        master.to_parquet(path)
        return str(path), M, G, C, M_annot, G_annot, master
    return _make
