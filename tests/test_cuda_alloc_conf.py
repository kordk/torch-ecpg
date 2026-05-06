"""Unit tests for the PYTORCH_CUDA_ALLOC_CONF setup added in PR 3
(CUDA-only memory-pressure tuning).

These tests exercise the pure helper `apply_cuda_alloc_conf` in
`tecpg.__main__` directly. They do not require a GPU and do not import
torch — the helper is designed to run before torch is imported so that
PyTorch's caching allocator picks up the config when it initializes.
"""
import importlib
import os

import tecpg.__main__ as tecpg_main


def _restore_env(saved):
    """Restore the relevant env vars to their pre-test values."""
    for var in ('PYTORCH_CUDA_ALLOC_CONF', 'TECPG_DISABLE_EXPANDABLE_SEGMENTS'):
        if saved[var] is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = saved[var]


def _snapshot_env():
    return {
        var: os.environ.get(var)
        for var in ('PYTORCH_CUDA_ALLOC_CONF', 'TECPG_DISABLE_EXPANDABLE_SEGMENTS')
    }


def test_apply_cuda_alloc_conf_sets_default_when_unset():
    saved = _snapshot_env()
    try:
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        os.environ.pop('TECPG_DISABLE_EXPANDABLE_SEGMENTS', None)
        tecpg_main.apply_cuda_alloc_conf()
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True'
    finally:
        _restore_env(saved)


def test_apply_cuda_alloc_conf_preserves_user_override():
    """If PYTORCH_CUDA_ALLOC_CONF is already set, the helper must not
    overwrite it. Users tuning the allocator directly own that knob."""
    saved = _snapshot_env()
    try:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ.pop('TECPG_DISABLE_EXPANDABLE_SEGMENTS', None)
        tecpg_main.apply_cuda_alloc_conf()
        assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'max_split_size_mb:128'
    finally:
        _restore_env(saved)


def test_apply_cuda_alloc_conf_opt_out():
    """TECPG_DISABLE_EXPANDABLE_SEGMENTS=1 must skip configuration
    entirely (no key inserted)."""
    saved = _snapshot_env()
    try:
        os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        for value in ('1', 'true', 'TRUE', 'Yes', 'yes'):
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['TECPG_DISABLE_EXPANDABLE_SEGMENTS'] = value
            tecpg_main.apply_cuda_alloc_conf()
            assert 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ, (
                f'opt-out value {value!r} should have skipped config'
            )
    finally:
        _restore_env(saved)


def test_apply_cuda_alloc_conf_opt_out_falsy_values_do_not_skip():
    """An unset / empty / explicitly-falsy opt-out must NOT skip
    configuration. Only the documented truthy values disable it."""
    saved = _snapshot_env()
    try:
        for falsy in ('', '0', 'false', 'no'):
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
            os.environ['TECPG_DISABLE_EXPANDABLE_SEGMENTS'] = falsy
            tecpg_main.apply_cuda_alloc_conf()
            assert os.environ.get('PYTORCH_CUDA_ALLOC_CONF') == 'expandable_segments:True', (
                f'falsy opt-out {falsy!r} unexpectedly skipped config'
            )
    finally:
        _restore_env(saved)


def test_high_water_lowered_to_seventy_five_percent():
    """PR 3 lowers the GPU caching-allocator high-water mark from 0.85
    to 0.75 to trigger empty_cache() one chunk earlier under memory
    pressure. Both copies (processing.py + regression_full.py) must
    stay in sync."""
    from tecpg import processing, regression_full

    assert processing.HIGH_WATER == 0.75
    assert regression_full.HIGH_WATER == 0.75
