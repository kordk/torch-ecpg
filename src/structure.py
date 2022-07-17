from typing import (
    Any,
    Callable,
    Dict,
    List,
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


class FlatComputeResult:
    '''
    Data structure to store and process data that maps gene ids and
    methylation ids to values. Has visualization built in to view the
    data. Stores the data in a flat dictionary, which allows for
    methods like .which() and constant time value access, but is much
    more storage-costly. Do not use for large datasets.
    '''

    def __init__(
        self,
        flatdata: Optional[FlatData] = None,
    ) -> None:
        '''
        Initializes the ComputeResult object. Takes in an optional
        flatdata object to initialize object from another (see
        ComputeResult.where()). If flatdata not provided, object is
        initialized empty.
        '''
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

    def __setitem__(self, key: Tuple[str, str], item: T) -> None:
        '''
        Takes in a key (methylation site id and gene id) and
        maps it to the provided item. Use like a dictionary:
            self[M, G] = val

        Directly modifies self.flatdata.
        '''
        self.flatdata[key] = item
        if key[0] not in self.first:
            self.first.append(key[0])
        if key[1] not in self.last:
            self.last.append(key[1])

    def __str__(self) -> str:
        '''Returns str of self.flatdata'''
        return str(self.flatdata)

    def __repr__(self) -> str:
        '''Returns repr of self.flatdata'''
        return repr(self.flatdata)

    def where(self, condition: Callable[[T], bool]) -> 'FlatComputeResult':
        '''
        Returns a ComputeResult instance of self where flatdata is
        filtered by the provided condition. Condition is a callable that
        takes in each value and returns a bool representing whether it
        is included in the output.
        '''
        data = {}
        for key, value in self.flatdata.items():
            if condition(value):
                data[key] = value
        return FlatComputeResult(flatdata=data)

    def data(self, flipped: bool = False) -> CompressedData:
        '''
        Returns a nested dictionary of self.flatdata. If not flipped,
        returns
        {[first index]: {[last index]: (self.flatdata[first, last])}}.
        If flipped, returns
        {[last index]: {[first index]: (self.flatdata[first, last])}}.
        '''
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
        '''
        Returns a 2 dimensional list made using list_func. If not
        flipped, the outer list represents the first index, and if
        flipped, the outer list represents the last index. Essentially
        returns self.data(flipped) without keys.
        '''
        return list_func(
            list_func(inner.values()) for inner in self.data(flipped).values()
        )

    def dataframe(self, flipped: bool = False) -> pandas.DataFrame:
        '''Returns self.data(flipped) as pandas.DataFrame'''
        return pandas.DataFrame(self.data(flipped))

    def visualize_image(
        self, imshow_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        '''
        Visualizes the values at gene id, methylation site id as a
        matplotlib image using matplotlib.pyplot.imshow. The x-axis is
        methylation site id and the y-axis is gene id. The color in the
        image represents the value at the specified methylation site and
        gene. Takes in imshow_kwargs, a dictionary that maps keyword
        arguments to values, which is passed into imshow using **.
        '''
        fig, ax = plt.subplots()

        ax.imshow(self.dataframe(), **(imshow_kwargs or {}))
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
