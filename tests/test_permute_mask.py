import numpy as np
import pandas as pd
from tecpg.permute import _compute_trans_mask
from tecpg.logger import Logger

def test_compute_trans_mask_semantics():
    logger = Logger()

    # Create dummy reported pairs
    reported_pairs = pd.DataFrame({
        'mt_id': ['cg1', 'cg2', 'cg3', 'cg4'],
        'gt_id': ['geneA', 'geneA', 'geneB', 'geneC']
    })

    # M_annot: cg1 (chr1), cg2 (chr1), cg3 (chr2), cg4 (chr3)
    M_annot = pd.DataFrame({
        'chrom': [1, 1, 2, 3],
        'pos': [100, 200, 300, 400]
    }, index=['cg1', 'cg2', 'cg3', 'cg4'])

    # G_annot: geneA (chr1), geneB (chr1), geneC (chr3)
    G_annot = pd.DataFrame({
        'chrom': [1, 1, 3],
        'pos': [150, 800, 400],
        'strand': [1, -1, 1]
    }, index=['geneA', 'geneB', 'geneC'])

    # Pair 1: cg1 (chr1) - geneA (chr1) => Same chrom => trans False, cis True
    # Pair 2: cg2 (chr1) - geneA (chr1) => Same chrom => trans False, cis True (dist 50)
    # Pair 3: cg3 (chr2) - geneB (chr1) => Diff chrom => trans True, cis False
    # Pair 4: cg4 (chr3) - geneC (chr3) => Same chrom => trans False, cis True (dist 0)

    # region='trans'
    mask_trans = _compute_trans_mask(
        reported_pairs, M_annot, G_annot, region='trans',
        window_base=0, downstream=1000, upstream=1000, logger=logger
    )

    expected_trans = np.array([False, False, True, False])
    np.testing.assert_array_equal(mask_trans, expected_trans, "Trans mask failed semantics test")

    # region='cis' with small window (dist must be < 100)
    # Pair 1: dist = g_pos - m_pos = 150 - 100 = 50. Strand 1 => window (-50, 50). Condition: -50 < 50 < 50 => False (boundary exclusive)
    # Actually wait:
    # Pair 1: g_pos=150, m_pos=100. dist=50. window (-60, 60) -> True.
    # Pair 2: g_pos=150, m_pos=200. dist=-50. window (-60, 60) -> True.
    # Pair 3: g_pos=800, m_pos=300, diff chrom -> False
    # Pair 4: g_pos=400, m_pos=400, dist=0 -> True

    mask_cis = _compute_trans_mask(
        reported_pairs, M_annot, G_annot, region='cis',
        window_base=0, downstream=60, upstream=60, logger=logger
    )
    expected_cis = np.array([True, True, False, True])
    np.testing.assert_array_equal(mask_cis, expected_cis, "Cis mask failed semantics test")

    print("Semantics tests passed.")

if __name__ == '__main__':
    test_compute_trans_mask_semantics()
