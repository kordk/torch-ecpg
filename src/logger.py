from collections.abc import Mapping
from datetime import datetime
from time import perf_counter
from typing import Callable, Generator, List, Literal, Optional, TypeVar
from colorama import Fore as colors
from colorama import Style as style

RT = TypeVar('RT')
Modes = Literal['debug', 'info', 'warning', 'error']


class PassAsKwarg(Mapping):
    '''
    Allows the class instance to be passed into a function:

    obj = PassAsKwarg('object')
    function(**obj)

    The function is passed object=obj.
    '''

    def __init__(self, key: str) -> None:
        self.key = key

    def __iter__(self) -> Generator[str, None, None]:
        yield self.key

    def __len__(self) -> int:
        return 1

    def __getitem__(self, item) -> Optional['PassAsKwarg']:
        return self if item == self.key else None

    def __hash__(self) -> int:
        return id(self)


class Logger(PassAsKwarg):
    def __init__(
        self,
        verbosity: Optional[int] = None,
        debug: Optional[bool] = None,
        log_dir: Optional[str] = None,
        *,
        parent: Optional['Logger'] = None,
    ) -> None:
        super().__init__('logger')

        if parent is not None:
            self.verbosity = (
                parent.verbosity if verbosity is None else verbosity
            )
            self.is_debug = parent.is_debug if debug is None else debug
            self.log_dir = parent.log_dir if log_dir is None else log_dir

            self.date_format: str = parent.date_format

            self.debug_color: str = parent.debug_color
            self.debug_template: str = parent.debug_template

            self.info_color: str = parent.info_color
            self.info_template: str = parent.info_template

            self.warning_color: str = parent.warning_color
            self.warning_template: str = parent.warning_template

            self.error_color: str = parent.error_color
            self.error_template: str = parent.error_template

        else:
            self.verbosity = 1 if verbosity is None else verbosity
            self.is_debug = False if debug is None else debug
            self.log_dir = log_dir

            self.date_format: str = '%Y%m%d_%H%M%S'

            self.debug_color: str = colors.BLACK
            self.debug_template: str = '[DEBUG{modifier}] {message}'

            self.info_color: str = colors.WHITE
            self.info_template: str = '[INFO{modifier}] {message}'

            self.warning_color: str = colors.YELLOW
            self.warning_template: str = '[WARNING{modifier}] {message}'

            self.error_color: str = colors.RED + style.BRIGHT
            self.error_template: str = '[ERROR{modifier}] {message}'

        self.logs: List[str] = []

        self.count_func = self.info
        self.current_count: int = 0

        self.time_func = self.info
        self.get_time_func = perf_counter
        self.start_time = self.get_time_func()
        self.end_time = self.start_time
        self.times: List[float] = []

        self.timer_count: int = 0
        self.total_time: float = 0
        self.time_since_last: float = 0
        self.average_time: float = 0

        self.parent = parent

    def _add_to_logs(self, message: str) -> None:
        self.logs.append(message)
        if self.parent:
            self.parent._add_to_logs(message)

    def _log(self, message: str) -> None:
        print(message + style.RESET_ALL)
        self._add_to_logs(message + style.RESET_ALL)

    def _get_func(self, mode: str) -> Callable[..., None]:
        if mode == 'debug':
            return self.debug
        elif mode == 'info':
            return self.info
        elif mode == 'warning':
            return self.warning
        elif mode == 'error':
            return self.error
        else:
            raise ValueError(f'The mode \'{mode}\' is not valid')

    def debug(self, message: str, *args, modifier: str = '', **kwargs) -> None:
        if not self.is_debug:
            return
        formatted = self.debug_color + self.debug_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def info(self, message: str, *args, modifier: str = '', **kwargs) -> None:
        if self.verbosity > 1:
            return
        formatted = self.info_color + self.info_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def warning(
        self, message: str, *args, modifier: str = '', **kwargs
    ) -> None:
        if self.verbosity > 2:
            return
        formatted = self.warning_color + self.warning_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def error(self, message: str, *args, modifier: str = '', **kwargs) -> None:
        if self.verbosity > 3:
            return
        formatted = self.error_color + self.error_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def start_count(
        self,
        mode: Modes = 'info',
        message: str = 'Starting counter...',
        *args,
        **kwargs,
    ) -> None:
        self.count_func = self._get_func(mode)

        self.current_count = 0
        self.count_func(message, *args, **kwargs)

    def count(self, message: str, *args, increase: int = 1, **kwargs) -> None:
        self.current_count += increase
        formatted = message.format(*args, i=self.current_count, **kwargs)
        self.count_func(formatted, modifier='COUNT')

    def start_timer(
        self,
        mode: Modes = 'info',
        message: str = 'Starting timer...',
        *args,
        **kwargs,
    ) -> None:
        self.time_func = self._get_func(mode)

        self.start_time = self.get_time_func()
        self.times = [self.start_time]
        self.time_func(message, *args, **kwargs)

    def time(
        self, message: str, *args, ignore: bool = False, **kwargs
    ) -> None:
        if not ignore:
            time = self.get_time_func()
            self.times.append(time)
            self.end_time = time

        self.timer_count = len(self.times) - 1
        self.total_time = self.end_time - self.start_time
        self.time_since_last = self.end_time - self.times[-2]
        self.average_time = self.total_time / self.timer_count

        formatted = message.format(
            *args,
            **kwargs,
            i=self.timer_count,
            t=self.total_time,
            l=self.time_since_last,
            a=self.average_time,
        )
        self.time_func(
            formatted,
            modifier='TIMER',
        )

    def time_check(self, *args, **kwargs) -> None:
        self.time(*args, ignore=True, **kwargs)

    def save(self, file_name: Optional[str] = None) -> None:
        if self.log_dir is None:
            return
        if file_name is None:
            date_code = datetime.now().strftime(self.date_format)
            file_name = date_code + '_logfile.txt'
        with open(self.log_dir + file_name, 'w') as file:
            log_str = '\n'.join(self.logs)
            file.write(log_str)

    def history(self, chunk_size: Optional[int] = None) -> None:
        if chunk_size is None:
            chunk_size = len(self.logs)
        print(
            style.BRIGHT + colors.BLUE + '\nLogger history: ' + style.RESET_ALL
        )
        for i in range(0, len(self.logs), chunk_size):
            j = i + chunk_size
            chunk = '\n'.join(self.logs[i:j])
            print(chunk)
            if i + chunk_size < len(self.logs):
                input(f'Press enter to show up to {chunk_size} more...  ')

    def alias(self, *args, **kwargs) -> 'Logger':
        return Logger(*args, parent=self, **kwargs)

    @staticmethod
    def init(func: Callable[..., RT]) -> Callable[..., RT]:
        def wrapper(*args, **kwargs) -> RT:
            if 'logger' not in kwargs:
                kwargs['logger'] = None
            if kwargs['logger'] is None:
                kwargs['logger'] = Logger()

            func(*args, **kwargs)

        return wrapper
