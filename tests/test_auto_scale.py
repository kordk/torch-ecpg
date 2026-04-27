from tecpg.cli import _auto_save_threads

def test_auto_save_threads():
    # 16 GB / 8 physical cores -> 2
    assert _auto_save_threads(8, 16.0) == 2

    # 512 GB / 32 physical cores -> 8
    assert _auto_save_threads(32, 512.0) == 8

    # 1 TB / 128 physical cores -> 16 (capped)
    assert _auto_save_threads(128, 1024.0) == 16

    # 8 GB / 4 physical cores -> 2 (floor)
    assert _auto_save_threads(4, 8.0) == 2
