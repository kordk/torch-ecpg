from typing import Any, Dict, List, Tuple, Generator
import pandas


IterateResult = Generator[
    Tuple[
        Tuple[str, str],
        Tuple[List[Any], List[Any]],
    ],
    None,
    None,
]


class DataMG:
    def __init__(self, M: pandas.DataFrame, G: pandas.DataFrame) -> None:
        _M: pandas.DataFrame = M.reindex(sorted(M.columns), axis=1)
        _G: pandas.DataFrame = G.reindex(sorted(G.columns), axis=1)

        if not _M.columns.equals(_G.columns):
            print(f'{_M.columns = }')
            print(f'{_G.columns = }')
            raise ValueError('Columns do not match')

        self.M: Dict[str, List[Any]] = _M.T.to_dict('list')
        self.G: Dict[str, List[Any]] = _G.T.to_dict('list')

        self.person_labels = list(_M.columns)
        self.m_labels = list(_M.index)
        self.g_labels = list(_G.index)

        self.size = (
            len(self.person_labels),
            len(self.m_labels),
            len(self.g_labels),
        )

    def iterate(self) -> IterateResult:
        for m_label, m_row in self.M.items():
            for g_label, g_row in self.G.items():
                yield (m_label, g_label), (m_row, g_row)
