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
        M_annot_raw = pd.DataFrame(index=M_annot['name'])
        M_annot_raw['chrom'] = ['chr19'] * len(M_annot_raw)
        if len(M_annot_raw) > 0:
            M_annot_raw.loc[M_annot_raw.index[0], 'chrom'] = 'chrX'
        if len(M_annot_raw) > 1:
            M_annot_raw.loc[M_annot_raw.index[1], 'chrom'] = float('nan') # dropped

        M_annot_raw['chromStart'] = range(len(M_annot_raw))
        M_annot_raw['chromEnd'] = range(len(M_annot_raw))
        M_annot_raw['score'] = 0
        M_annot_raw['strand'] = '+'

        G_annot_raw = pd.DataFrame(index=G_annot['name'])
        G_annot_raw['chrom'] = ['chr19'] * len(G_annot_raw)
        if len(G_annot_raw) > 0:
            G_annot_raw.loc[G_annot_raw.index[0], 'chrom'] = 'chrY'
        if len(G_annot_raw) > 1:
            G_annot_raw.loc[G_annot_raw.index[1], 'chrom'] = 'GL000220.1' # unmappable, dropped

        G_annot_raw['chromStart'] = range(len(G_annot_raw))
        G_annot_raw['chromEnd'] = range(len(G_annot_raw))
        G_annot_raw['score'] = 0
        G_annot_raw['strand'] = ['-'] * len(G_annot_raw)

        return M, G, C, M_annot_raw, G_annot_raw
    return _make
