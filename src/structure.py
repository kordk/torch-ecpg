from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Generator,
    TypeVar,
)
from matplotlib import pyplot as plt
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


T = TypeVar('T')
FlatData = Dict[Tuple[str, str], T]
CompressedData = Dict[str, Dict[str, T]]


class ComputeResult:
    def __init__(
        self,
        flatdata: Optional[FlatData] = None,
    ) -> None:
        if flatdata is None:
            self.flatdata: FlatData = {}
            self.first = []
            self.last = []
        else:
            self.flatdata = flatdata
            _first, _last = set(), set()
            for first, last in flatdata.keys():
                _first.add(first)
                _last.add(last)
            self.first = list(_first)
            self.last = list(_last)

    def __setitem__(self, key: Tuple[str, str], item: T):
        self.flatdata[key] = item
        if key[0] not in self.first:
            self.first.append(key[0])
        if key[1] not in self.last:
            self.last.append(key[1])

    def __str__(self) -> str:
        return str(self.flatdata)

    def __repr__(self) -> str:
        return repr(self.flatdata)

    def where(self, condition: Callable[[T], bool]) -> 'ComputeResult':
        data = {}
        for key, value in self.flatdata.items():
            if condition(value):
                data[key] = value
        return ComputeResult(flatdata=data)

    def data(self, flipped: bool = False) -> CompressedData:
        out: CompressedData = {}
        for (first, last), value in self.flatdata.items():
            if flipped:
                if first not in out:
                    out[first] = {}
                out[first][last] = value
            else:
                if last not in out:
                    out[last] = {}
                out[last][first] = value
        return out

    def nested(
        self, flipped: bool = False, list_func: Any = list
    ) -> List[List[T]]:
        return list_func(
            list_func(inner.values()) for inner in self.data(flipped).values()
        )

    def dataframe(self, flipped: bool = False) -> pandas.DataFrame:
        return pandas.DataFrame(self.data(flipped))

    def visualize(self, imshow_kwargs: Dict[str, Any] = {}) -> None:
        fig, ax = plt.subplots()

        ax.imshow(self.dataframe(), **imshow_kwargs)
        ax.set_title('Correlation between methylation and gene expression')
        ax.set_xlabel('Methylation')
        ax.set_ylabel('Gene expression')
        ax.invert_yaxis()

        annot = ax.annotate(
            '',
            xy=(0, 0),
            xytext=(3, 3),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
        )
        annot.set_visible(False)

        def update_annotation(event: Any) -> None:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                annot.set_visible(False)
                return
            annot.set_visible(True)
            first = self.first[round(x)]
            last = self.last[round(y)]
            value = round(self.flatdata[first, last], 5)
            text = f'Value at {first}, {last}: {value}'
            annot.xy = x, y
            annot.set_text(text)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('motion_notify_event', update_annotation)

        plt.show()
