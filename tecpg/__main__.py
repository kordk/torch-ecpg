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


def apply_cuda_alloc_conf():
    """Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at process
    start, before torch is imported, so PyTorch's caching allocator
    picks it up when it initializes.

    expandable_segments reduces GPU memory fragmentation by letting the
    caching allocator grow individual segments on demand instead of
    rounding every block up to a fixed segment size. On the chunked
    qr path used by tecpg this measurably lowers peak VRAM held by
    the allocator after a few gene-chunk iterations.

    The flag is harmless on CPU-only hosts: the env var is only
    consulted by PyTorch when the CUDA caching allocator is
    initialized, which never happens without a CUDA device. We
    therefore set it unconditionally rather than spinning up torch
    here just to call torch.cuda.is_available().

    User overrides are honored:
      - If PYTORCH_CUDA_ALLOC_CONF is already set in the environment,
        we leave it alone (the user knows what they want).
      - If TECPG_DISABLE_EXPANDABLE_SEGMENTS is truthy, skip the
        configuration entirely.
    """
    if os.environ.get('TECPG_DISABLE_EXPANDABLE_SEGMENTS', '').strip().lower() in ('1', 'true', 'yes'):
        return
    if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
        return
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


apply_cuda_alloc_conf()

from .cli import start


def main() -> None:
    # Starts the command line interface
    start()

if __name__ == '__main__':
    main()
