import os
import re
import subprocess

BASE_COMMAND = ['tecpg', 'run', 'mlr-full']
ARGS_LIST = [
    [],
    ['--cis'],
    ['--distal'],
    ['--trans'],
    ['-p', '0.05'],
    ['-p', '0.05', '--cis'],
    ['-p', '0.05', '--distal'],
    ['-p', '0.05', '--trans'],
    ['-l', '2000'],
    ['-l', '2000', '--cis'],
    ['-l', '2000', '--distal'],
    ['-l', '2000', '--trans'],
    ['-l', '2000', '-p', '0.05'],
    ['-l', '2000', '-p', '0.05', '--cis'],
    ['-l', '2000', '-p', '0.05', '--distal'],
    ['-l', '2000', '-p', '0.05', '--trans'],
]
DATA_COMMAND_TEMPLATE = 'tecpg data dummy -s {} -m {} -g {}'
DATA_COUNTS = [
    [100, 100, 100],
    [300, 1000, 1000],
    [300, 50000, 10000],
]
OUTPUT_PREFIX = (
    '<details>\n<summary>{command}</summary>\n\n```python\n{stdout}\n```\n<'
    '/details>\n\n'
)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def test(cwd: str) -> str:
    command_list = [BASE_COMMAND + args for args in ARGS_LIST]
    output = ''
    for index, command in enumerate(command_list, 1):
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout = ANSI_ESCAPE.sub(
            '', process.stdout.read().decode('ascii').replace('\n\n', '\n')
        )
        output += OUTPUT_PREFIX.format(
            command=' '.join(command), stdout=stdout
        )
        print(f'Completed {index} out of {len(ARGS_LIST)}')

    return output


def main() -> None:
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