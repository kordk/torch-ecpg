import pandas as pd
import numpy as np
from tecpg.chrom import canonicalize_chrom

def test_canonicalize_chrom_battery():
    # Value oracle for canonicalize_chrom

    def check_equality(out, expected_values, s_name):
        expected = pd.Series(expected_values, name=s_name)
        # Use Int64 if no float values except nan, or let pandas handle it, but allow casting expected to match out dtype
        pd.testing.assert_series_equal(out, expected.astype(out.dtype))

    # 1. Standard mapping and normal strings
    s = pd.Series(['chr19', 'chrX', 'chrY', 'chr2'])
    out = canonicalize_chrom(s)
    check_equality(out, [19.0, -1.0, -2.0, 2.0], s.name)

    # 2. Bare variants
    s = pd.Series(['19', 'X', 'Y'])
    out = canonicalize_chrom(s)
    check_equality(out, [19.0, -1.0, -2.0], s.name)

    # 3. Mixed case
    s = pd.Series(['CHR19', 'chrx'])
    out = canonicalize_chrom(s)
    check_equality(out, [19.0, -1.0], s.name)

    # 4. Mito
    s = pd.Series(['chrM', 'MT', 'chrMT'])
    out = canonicalize_chrom(s)
    check_equality(out, [-3.0, -3.0, -3.0], s.name)

    # 5. Integer passthrough
    s = pd.Series([1, 2, 3], dtype=int)
    out = canonicalize_chrom(s)
    pd.testing.assert_series_equal(out, pd.Series([1, 2, 3], dtype=int, name=s.name))

    # 6. Unmappable / Scaffolds / None
    s = pd.Series(['chr19', None, 'scaffold_1'])
    out = canonicalize_chrom(s)
    check_equality(out, [19.0, np.nan, np.nan], s.name)

    # 7. Additional cases requested: np.nan, pd.NA, float, whitespace padding, empty strings, alt-contigs
    s = pd.Series([np.nan, pd.NA, 19.0, ' chr19 ', '', 'chr22_KI270879v1_alt'])
    out = canonicalize_chrom(s)
    check_equality(out, [np.nan, np.nan, 19.0, 19.0, np.nan, np.nan], s.name)

def test_canonicalize_matches_legacy_map_chrom():
    # Behavior-preservation oracle for the permute refactor
    def map_chrom(s):
        if pd.api.types.is_integer_dtype(s):
            return s
        s = s.astype('string').str.strip()
        s = s.str.replace(r'^chr', '', regex=True, case=False)
        s = s.str.upper()
        num = pd.to_numeric(s, errors='coerce')
        spec = s.map({'X': -1, 'Y': -2, 'MT': -3, 'M': -3})
        return num.fillna(spec)

    # Combine all test cases into one comprehensive battery
    battery = pd.Series([
        'chr19', 'chrX', 'chrY', 'chr2',
        '19', 'X', 'Y',
        'CHR19', 'chrx',
        'chrM', 'MT', 'chrMT',
        1, 2, 3,
        'chr19', None, 'scaffold_1',
        np.nan, pd.NA, 19.0, ' chr19 ', '', 'chr22_KI270879v1_alt'
    ])

    legacy_out = map_chrom(battery)
    new_out = canonicalize_chrom(battery)

    pd.testing.assert_series_equal(new_out, legacy_out)
