import gzip
from typing import Dict, List, Optional, Tuple

from .logger import Logger

GEO_SAMPLE_KEYS = [
    'Sample_description',
    'Sample_geo_accession',
    'Sample_title',
]


def geo_dict(
    file_path: str,
    char_id: str = 'Sample_characteristics_ch1',
    decompress: Optional[bool] = None,
    simplify_covar: bool = False,
    *,
    logger: Logger = Logger(),
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Reads file_path as a geo data text file. Returns a dictionary that
    maps keys to data and a dictionary that maps characteristics with a
    list of values for each sample, based on char_id. If decompress is
    None, it will infer whether to decompress with gzip if the file_path
    ends with .gz. If set to a boolean, it will decompress if True.
    """
    if decompress is None:
        decompress = file_path.endswith('.gz')
    with gzip.open(file_path, 'rt') if decompress else open(
        file_path, 'r'
    ) as file:
        lines = file.readlines()

    data: Dict[str, List[str]] = {}
    chars: Dict[str, List[str]] = {}
    for line in lines:
        if line.startswith('!'):
            parts = line.removesuffix('\n').split('\t')
            key = parts[0][1:]
            values = [part[1:-1] for part in parts[1:]]

            if key == char_id:
                for part in values:
                    if not part:
                        continue
                    if ':' not in part:
                        logger.warning(
                            f'Skipping characteristic {part} because it does'
                            ' not include \':\''
                        )
                        continue
                    char_key, char_val = part.split(':')
                    if simplify_covar and char_key not in ['age', 'Sex']:
                        continue
                    if char_key not in chars:
                        chars[char_key] = []
                    chars[char_key].append(char_val.strip())
            else:
                data[key] = values
    return data, chars


def geo_samples(
    geo_data: Dict[str, List[str]]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Takes in a dictionary of geo_data returned by geo_dict. Returns a
    tuple of 3 lists of strings for Sample_description,
    Sample_geo_accession, and Sample_title respectively.

    Example Sample_title: "6088"
    Example Sample_geo_accession: "GSM1868036"
    Example Sample_description: "6164647041_R01C01"
    """
    return tuple(geo_data[key] for key in GEO_SAMPLE_KEYS)
