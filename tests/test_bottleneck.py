import pytest
from tecpg.logger import (
    analyze_bottleneck,
    LOW_RAM_GB,
    GPU_UNDERUTIL_RATIO,
    GPU_UNDERUTIL_SM_PCT,
    HIGH_CPU_PCT,
)

def test_analyze_bottleneck():
    # 1. Low RAM bottleneck
    res = analyze_bottleneck(
        gpu_time=1.0, total_time=2.0, h2d_time=0.1, d2h_time=0.1, write_time=0.1,
        util_sm=80.0, ram_avail_gb=LOW_RAM_GB - 0.5, cpu_percent=50.0
    )
    assert "low system RAM" in res

    # 2. GPU underutilized bottleneck
    res = analyze_bottleneck(
        gpu_time=0.1, total_time=1.0, h2d_time=0.1, d2h_time=0.1, write_time=0.1,
        util_sm=GPU_UNDERUTIL_SM_PCT - 10, ram_avail_gb=10.0, cpu_percent=50.0
    )
    assert "GPU underutilized" in res

    # 3. PCIe transfer dominant bottleneck
    res = analyze_bottleneck(
        gpu_time=0.2, total_time=1.0, h2d_time=0.5, d2h_time=0.5, write_time=0.1,
        util_sm=80.0, ram_avail_gb=10.0, cpu_percent=50.0
    )
    assert "PCIe transfer dominant" in res

    # 4. Disk I/O bottleneck
    res = analyze_bottleneck(
        gpu_time=0.5, total_time=1.5, h2d_time=0.1, d2h_time=0.1, write_time=0.8,
        util_sm=80.0, ram_avail_gb=10.0, cpu_percent=50.0
    )
    assert "disk I/O" in res

    # 5. CPU postprocessing bottleneck
    res = analyze_bottleneck(
        gpu_time=0.5, total_time=2.0, h2d_time=0.1, d2h_time=0.1, write_time=0.1,
        util_sm=GPU_UNDERUTIL_SM_PCT - 10, ram_avail_gb=10.0, cpu_percent=HIGH_CPU_PCT + 5
    )
    assert "CPU postprocessing" in res

    # 6. No obvious bottleneck (perfect world)
    res = analyze_bottleneck(
        gpu_time=0.9, total_time=1.0, h2d_time=0.05, d2h_time=0.05, write_time=0.01,
        util_sm=95.0, ram_avail_gb=10.0, cpu_percent=50.0
    )
    assert res is None
