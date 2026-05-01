from tecpg.cli import _auto_save_threads, _host_class

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
    # min(cores - 2, ram_gb // 4), capped at 8, floor 2.
    # Cap was lowered from 32 to 8 because profiling on RAID6/dm-crypt
    # LUNs (klabdev) showed the device saturates well before 32 writers
    # and extra workers only add kernel-writeback CPU cost.

    # 512 GB / 32 physical cores -> min(30, 128) capped to 8
    assert _auto_save_threads(32, 512.0) == 8

    # 1 TB / 128 physical cores -> capped to 8
    assert _auto_save_threads(128, 1024.0) == 8

    # 64 GB / 16 physical cores -> min(14, 16) capped to 8
    assert _auto_save_threads(16, 64.0) == 8

    # Server-class hosts where the heuristic naturally falls below the
    # cap should not be inflated.
    # 32 GB / 12 cores -> min(10, 8) = 8 (just barely server-class)
    assert _auto_save_threads(12, 32.0) == 8


def test_host_class():
    # Minimum-class hosts: the 16 GB / 8-core target stays minimum.
    assert _host_class(8, 16.0) == 'minimum'
    assert _host_class(4, 8.0) == 'minimum'
    assert _host_class(11, 64.0) == 'minimum'  # below core threshold
    assert _host_class(32, 24.0) == 'minimum'  # below RAM threshold

    # Server-class hosts: both thresholds met.
    assert _host_class(12, 32.0) == 'server'
    assert _host_class(32, 512.0) == 'server'
    assert _host_class(128, 1024.0) == 'server'
