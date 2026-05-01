from tecpg.cli import _auto_save_threads

def test_auto_save_threads():
    # Small/laptop hosts keep the conservative formula.

    # 16 GB / 8 physical cores -> 2 (laptop path)
    assert _auto_save_threads(8, 16.0) == 2

    # 8 GB / 4 physical cores -> 2 (floor, laptop path)
    assert _auto_save_threads(4, 8.0) == 2

    # Just under the small-host threshold (cores >= 12 but ram < 32)
    # still uses the conservative formula.
    assert _auto_save_threads(12, 16.0) == 2

    # Larger hosts use the relaxed formula:
    # min(cores - 2, ram_gb // 4), capped at 32, floor 2.

    # 512 GB / 32 physical cores -> min(30, 128) = 30
    assert _auto_save_threads(32, 512.0) == 30

    # 1 TB / 128 physical cores -> 32 (capped)
    assert _auto_save_threads(128, 1024.0) == 32

    # 64 GB / 16 physical cores -> min(14, 16) = 14
    assert _auto_save_threads(16, 64.0) == 14
