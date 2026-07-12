import pytest
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
