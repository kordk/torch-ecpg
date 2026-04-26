import os
import sys

def apply_blas_threads():
    blas_threads = None
    env_val = os.environ.get('TECPG_BLAS_THREADS')
    if env_val:
        try:
            blas_threads = int(env_val)
        except ValueError:
            pass

    for i, arg in enumerate(sys.argv):
        if arg == '--blas-threads' and i + 1 < len(sys.argv):
            try:
                blas_threads = int(sys.argv[i + 1])
            except ValueError:
                pass
        elif arg.startswith('--blas-threads='):
            try:
                blas_threads = int(arg.split('=', 1)[1])
            except ValueError:
                pass

    if blas_threads is not None and blas_threads > 0:
        val = str(blas_threads)
        for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
            if var not in os.environ:
                os.environ[var] = val

apply_blas_threads()

from .cli import start


def main() -> None:
    # Starts the command line interface
    start()

if __name__ == '__main__':
    main()
