import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib

class TestGPUMismatch(unittest.TestCase):
    def setUp(self):
        # Save original modules to restore later
        self.original_pynvml = sys.modules.get('pynvml')
        self.original_torch = sys.modules.get('torch')
        self.original_gpu_monitor = sys.modules.get('tecpg.gpu_monitor')

    def tearDown(self):
        # Restore original modules
        if self.original_pynvml:
            sys.modules['pynvml'] = self.original_pynvml
        elif 'pynvml' in sys.modules:
            del sys.modules['pynvml']

        if self.original_torch:
            sys.modules['torch'] = self.original_torch
        elif 'torch' in sys.modules:
            del sys.modules['torch']

        if self.original_gpu_monitor:
            sys.modules['tecpg.gpu_monitor'] = self.original_gpu_monitor
        elif 'tecpg.gpu_monitor' in sys.modules:
            del sys.modules['tecpg.gpu_monitor']

    def test_repro_mismatch(self):
        # Create mocks
        mock_pynvml = MagicMock()
        mock_torch = MagicMock()

        # Setup torch mock
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "NVIDIA L4"

        # Setup pynvml mock
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        # Device Handles
        handle_A2 = MagicMock(name="Handle_A2")
        handle_L4 = MagicMock(name="Handle_L4")

        def get_handle(index):
            if index == 0: return handle_A2
            if index == 1: return handle_L4
            raise ValueError(f"Invalid index {index}")

        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle

        def get_name(handle):
            if handle == handle_A2: return b"NVIDIA A2"
            if handle == handle_L4: return b"NVIDIA L4"
            return b"Unknown"

        mock_pynvml.nvmlDeviceGetName.side_effect = get_name

        # Patch sys.modules
        sys.modules['pynvml'] = mock_pynvml
        sys.modules['torch'] = mock_torch

        # We need to ensure tecpg.gpu_monitor is (re)loaded to use the mocks
        if 'tecpg.gpu_monitor' in sys.modules:
            del sys.modules['tecpg.gpu_monitor']

        import tecpg.gpu_monitor as gpu_monitor
        from tecpg.logger import Logger

        # Use a dummy logger that doesn't rely on system state
        log = MagicMock(spec=Logger)
        log.carry_data = {}

        # Ensure CUDA_VISIBLE_DEVICES is NOT set
        with patch.dict('os.environ'):
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            handle = gpu_monitor.init_gpu_monitor(log)

            self.assertEqual(handle, handle_L4, "Should select L4 matching by name")

if __name__ == '__main__':
    unittest.main()
