import pytest
from unittest import mock
from tecpg.logger import Logger
import tecpg.gpu_monitor as gm

class MockRates:
    gpu = 85
    memory = 60

def test_gpu_monitor_utilization():
    logger = Logger()
    with mock.patch("tecpg.gpu_monitor.HAS_PYNVML", True), \
         mock.patch("tecpg.gpu_monitor.pynvml") as mock_pynvml, \
         mock.patch("tecpg.gpu_monitor.get_gpu_temp", return_value=50):

        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = MockRates()
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000 # 150W
        mock_pynvml.nvmlDeviceGetClockInfo.side_effect = [1500, 7000, 1500, 7000, 1500, 7000, 1500, 7000, 1500, 7000, 1500, 7000]

        monitor = gm.ThermalMonitor("dummy_handle", 80, logger, poll_interval=0.01)
        monitor.start()

        import time
        time.sleep(0.05) # Let it poll a few times

        monitor.stop()

        assert monitor.last_util_sm == 85
        assert monitor.last_util_mem == 60
        assert monitor.avg_util_sm == 85
        assert monitor.last_power == 150.0
        assert monitor.last_clock_sm == 1500
        assert monitor.last_clock_mem == 7000
