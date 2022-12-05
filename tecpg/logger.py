import os
import time
from collections.abc import Mapping
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    TypeVar,
)

from colorama import Fore as colors
from colorama import Style as style

RT = TypeVar('RT')
Modes = Literal['debug', 'info', 'warning', 'error']


class PassAsKwarg(Mapping):
    """
    Allows the class instance to be passed into a function:

    obj = PassAsKwarg('object')
    function(**obj)

    The function is passed object=obj.
    """

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
    """
    Logging class that supports verbosity levels, debug, counting, log
    files, history, and timing.

    Logger.debug: prints a debug message if self.debug is True.
    Logger.info: prints an info message if verbosity is at least 1.
    Logger.warning: prints a warning message if verbosity is at least 2.
    Logger.error: prints an error message if verbosity is at least 3.

    Logger.start_counter: resets counter with a print style and message.
    Logger.count: increments counter and prints message with formatting.
    Logger.start_timer: resets the timer with a print style and message.
    Logger.time: adjusts timer variables and prints message with format.
    Logger.time_check: prints message with format without changing time.
    Logger.remaining_time: gets estimated time given iteration count.

    Logger.save: saves logs to a file.
    Logger.history: prints log history in optional chunks.
    Logger.alias: gets an alias to self with separate timer but same
    history.

    [DEPRECATED] Logger.init: decorates function by initializing logger.
    """

    def __init__(
        self,
        verbosity: Optional[int] = None,
        debug: Optional[bool] = None,
        log_dir: Optional[str] = None,
        carry_data: Optional[Dict[str, Any]] = None,
        timer_round: Optional[int] = None,
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
            self.timer_round = (
                parent.timer_round if timer_round is None else timer_round
            )

            self.carry_data = parent.carry_data

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
            self.timer_round = 4 if timer_round is None else timer_round
            self.carry_data = {} if carry_data is None else carry_data

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
        self.get_time_func = time.perf_counter
        self.start_time = self.get_time_func()
        self.end_time = self.start_time
        self.times: List[float] = []

        self.timer_count: int = 0
        self.total_time: float = 0
        self.time_since_last: float = 0
        self.average_time: float = 0

        self.parent = parent

    def add_to_logs(self, message: str) -> None:
        """Adds message to log history of both self and parent"""
        self.logs.append(message)
        if self.parent:
            self.parent.add_to_logs(message)

    def _log(self, message: str) -> None:
        """Logs message to console and adds message to log history"""
        print(message + style.RESET_ALL)
        self.add_to_logs(message + style.RESET_ALL)

    def _get_func(self, mode: str) -> Callable[..., None]:
        """Gets the print function based on a provided mode string"""
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
        """Prints debug message"""
        if not self.is_debug:
            return
        formatted = self.debug_color + self.debug_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def info(self, message: str, *args, modifier: str = '', **kwargs) -> None:
        """Prints info message (verbosity 1)"""
        if self.verbosity > 1:
            return
        formatted = self.info_color + self.info_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def warning(
        self, message: str, *args, modifier: str = '', **kwargs
    ) -> None:
        """Prints warning message (verbosity 2)"""
        if self.verbosity > 2:
            return
        formatted = self.warning_color + self.warning_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def error(self, message: str, *args, modifier: str = '', **kwargs) -> None:
        """Prints error message (verbosity 3)"""
        if self.verbosity > 3:
            return
        formatted = self.error_color + self.error_template.format(
            message=message.format(*args, **kwargs), modifier=modifier
        )
        self._log(formatted)

    def start_counter(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Resets and initializes the counter given a print mode and
        message.
        """
        mode, message, *message_args = args or ['info', 'Starting counter...']
        self.count_func = self._get_func(mode)

        self.current_count = 0
        self.count_func(message, *message_args, **kwargs)

    def count(self, *args, increase: int = 1, **kwargs) -> None:
        """
        Prints a specially formatted message and increments the counter.

        Formats:
        {i} = self.current_count

        increase: int = 1: value by which to increase the counter.
        """
        message, *message_args = args or [None]
        self.current_count += increase
        if message:
            formatted = message.format(
                *message_args, i=self.current_count, **kwargs
            )
            self.count_func(formatted, modifier='COUNT')

    def count_check(self, *args, **kwargs) -> None:
        """
        Alias to Logger.count, but with increase set to 0. Message will
        be formatted but count will not be increased.
        """
        self.count(*args, increase=0, **kwargs)

    def start_timer(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Resets and initializes the timer given a print mode and message.
        """
        mode, message, *message_args = args or ['info', 'Starting timer...']
        self.time_func = self._get_func(mode)

        self.start_time = self.get_time_func()
        self.times = [self.start_time]
        self.time_func(message, *message_args, **kwargs)

    def time(self, *args, ignore: bool = False, **kwargs) -> None:
        """
        Prints a specially formatted message and updates timer.

        Formats (rounded to self.timer_round):
        {i} = self.timer_count
        {t} = self.total_time
        {l} = self.time_since_last
        {a} = self.average_time

        ignore: bool = False: Formats message without updating timer.
            Logger.time_check() is preferred.
        """
        message, *message_args = args or [None]
        time = self.get_time_func()

        if not ignore:
            self.times.append(time)
            self.end_time = time

        self.timer_count = len(self.times) - 1
        self.total_time = time - self.start_time
        self.time_since_last = time - (
            self.end_time if ignore else self.times[-2]
        )
        self.average_time = self.total_time / self.timer_count

        if message:
            formatted = message.format(
                *message_args,
                **kwargs,
                i=self.timer_count,
                t=round(self.total_time, self.timer_round),
                l=round(self.time_since_last, self.timer_round),
                a=round(self.average_time, self.timer_round),
            )
            self.time_func(
                formatted,
                modifier='TIMER',
            )

    def time_check(self, *args, **kwargs) -> None:
        """
        Alias to Logger.time, but with ignore set to True. Message will
        be formatted but timer will not be updated.
        """
        self.time(*args, ignore=True, **kwargs)

    def remaining_time(self, n: int) -> float:
        """
        Returns the estimated remaining time given the total number of
        iterations (including previous iterations).
        """
        return self.average_time * (n - self.timer_count)

    def save(
        self, log_dir: Optional[str] = None, file_name: Optional[str] = None
    ) -> None:
        """
        Saves self.logs to log_dir (default: self.log_dir) + file_name
        (default: current time code).
        """
        if self.log_dir is None and log_dir is None:
            return
        if log_dir is None:
            log_dir = self.log_dir
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        if file_name is None:
            date_code = datetime.now().strftime(self.date_format)
            file_name = date_code + '_logfile.txt'
        file_path = os.path.join(log_dir, file_name)
        with open(file_path, 'w') as file:
            log_str = '\n'.join(self.logs)
            file.write(log_str)

    def history(self, chunk_size: Optional[int] = None) -> None:
        """
        Prints the log history. Optional chunk_size will return the log
        messages in batches of no more than chunk_size messages. If not
        provided, all messages will be printed.
        """
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
        """
        Returns a child logger with reset timer and counter but a log
        history that transfers up the alias chain. For example:
        logger = Logger(debug=True)
        logger.alias().alias().alias().alias().debug('Debug works!')
        # 'Debug works!'
        logger.history()
        # 'Debug works!'
        """
        return Logger(*args, parent=self, **kwargs)

    @staticmethod
    def init(func: Callable[..., RT]) -> Callable[..., RT]:
        """
        Status: Deprecated

        Initializes logger instance if not provided to the decorated
        function.
        """

        def wrapper(*args, **kwargs) -> RT:
            if 'logger' not in kwargs:
                kwargs['logger'] = None
            if kwargs['logger'] is None:
                kwargs['logger'] = Logger()

            func(*args, **kwargs)

        return wrapper
