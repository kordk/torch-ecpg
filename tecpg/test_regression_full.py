import copy
import os
import random
import re
import subprocess
from typing import Any, Callable, Dict, List


def parse_args(
    args: List[Any | Callable[[], Any]] | Any | Callable[[], Any]
) -> List[str]:
    if isinstance(args, list):
        return [str(arg()) if callable(arg) else str(arg) for arg in args]
    else:
        return [str(args()) if callable(args) else str(args)]


class Argument:
    def __init__(
        self,
        args: List[Any | Callable[[], Any]] | Any | Callable[[], Any],
        *,
        always: bool = False,
    ) -> None:
        self.args = parse_args(args)
        self.always = always


class ChoiceArgument:
    def __init__(
        self,
        *choices: List[
            List[Any | Callable[[], Any]] | Any | Callable[[], Any]
        ],
        always: bool = False,
    ) -> None:
        self.choices = [parse_args(choice) for choice in choices]
        self.always = always


def generate_args_lists(
    arguments: List[Argument | ChoiceArgument], limit: int | None = None
) -> List[List[str]]:
    args_lists: List[List[str]] = [[]]

    for argument in arguments:
        args_lists_copy = args_lists
        if argument.always:
            args_lists = []

        if isinstance(argument, Argument):
            args_lists.extend(
                [argument.args + arg_list for arg_list in args_lists_copy]
            )
        else:
            for choice in argument.choices:
                args_lists.extend(
                    [choice + arg_list for arg_list in args_lists_copy]
                )

    if limit is None:
        return args_lists
    return random.choices(args_lists, k=limit)


BASE_COMMAND = ['tecpg', 'run', 'mlr']
ARGUMENTS_FORMULA = [
    Argument('-f'),
    Argument(['-p', '0.05']),
    ChoiceArgument(
        '--all',
        ['--cis', '-u', '10000', '-d', '4000', '-w', '6000'],
        '--distal',
        '--trans',
        always=True,
    ),
    Argument(['-g', '200']),
    Argument(['-m', '200']),
]

DATA_COMMAND_TEMPLATE = 'tecpg data dummy -s {} -m {} -g {}'
DATA_COUNTS = [
    # [100, 100, 100],
    [300, 1000, 1000],
    # [300, 50000, 10000],
]
OUTPUT_PREFIX = (
    '<details>\n<summary>{command}</summary>\n\n```python\n{stdout}\n```\n<'
    '/details>\n\n'
)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
ARGS_LISTS = generate_args_lists(ARGUMENTS_FORMULA)


def test(cwd: str) -> str:
    command_list = [BASE_COMMAND + args for args in ARGS_LISTS]
    output = ''
    for index, command in enumerate(command_list, 1):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout = ANSI_ESCAPE.sub('', process.stdout.read().decode('ascii'))
        output += OUTPUT_PREFIX.format(
            command=' '.join(command), stdout=stdout
        )
        print(f'Completed {index} out of {len(ARGS_LISTS)}')

    return output.replace('\r\n', '\n')


def main() -> None:
    print(f'Testing with {len(ARGS_LISTS)} args')
    cwd = input('Enter the path to the working directory: ')
    output_path = os.path.join(cwd, 'regression_full_test_output.md')
    with open(output_path, 'w'):
        pass

    with open(output_path, 'a') as file:
        for counts in DATA_COUNTS:
            data_command = DATA_COMMAND_TEMPLATE.format(*counts)
            subprocess.run(data_command, cwd=cwd)
            file.write(
                'Tests with {} samples, {} methylation loci, and {} gene'
                ' expression loci:\n\n'.format(*counts)
            )
            output = test(cwd)
            file.write(output)


if __name__ == '__main__':
    main()
