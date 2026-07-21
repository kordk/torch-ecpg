import pandas as pd

def canonicalize_chrom(s: pd.Series) -> pd.Series:
    """Canonicalize a chromosome column to signed numeric codes.
    Integer dtype passes through unchanged. Strings: strip a leading 'chr'
    (case-insensitive), uppercase, coerce numerals; map X/Y/MT/M -> -1/-2/-3/-3;
    anything unmappable (scaffolds, NaN) -> NaN (caller drops via dropna)."""
    if pd.api.types.is_integer_dtype(s):
        return s
    s = s.astype('string').str.strip()
    s = s.str.replace(r'^chr', '', regex=True, case=False)
    s = s.str.upper()
    num = pd.to_numeric(s, errors='coerce')
    spec = s.map({'X': -1, 'Y': -2, 'MT': -3, 'M': -3})
    return num.fillna(spec)
