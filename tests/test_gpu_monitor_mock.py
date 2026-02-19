import sys
import unittest
from unittest.mock import MagicMock, patch
import os

class MockNVMLError(Exception):
    pass

class TestGPUUUIDMatching(unittest.TestCase):
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

    def test_uuid_matching(self):
        # Create mocks
        mock_pynvml = MagicMock()
        mock_torch = MagicMock()
        mock_pynvml.NVMLError = MockNVMLError

        # Define UUIDs
        UUID_L4 = "GPU-12345678-1234-1234-1234-123456789abc"
        UUID_A2 = "GPU-87654321-4321-4321-4321-cba987654321"

        # Setup torch mock
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_props = MagicMock()
        mock_props.uuid = UUID_L4
        mock_props.name = "NVIDIA L4"
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.get_device_name.return_value = "NVIDIA L4"

        # Setup pynvml mock
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        # Device Handles
        handle_A2 = MagicMock(name="Handle_A2")
        handle_L4 = MagicMock(name="Handle_L4")

        # Mock pynvml.nvmlDeviceGetHandleByUUID
        # Simulate FAIL so we test the iteration logic
        mock_pynvml.nvmlDeviceGetHandleByUUID.side_effect = MockNVMLError("Unknown UUID")

        # Also mock iteration just in case code falls back to it
        def get_handle_by_index(index):
            if index == 0: return handle_A2  # NVML 0 is A2
            if index == 1: return handle_L4  # NVML 1 is L4
            raise ValueError(f"Invalid index {index}")
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle_by_index

        def get_uuid(handle):
            if handle == handle_A2: return UUID_A2.encode('utf-8')
            if handle == handle_L4: return UUID_L4.encode('utf-8')
            return b"Unknown"
        mock_pynvml.nvmlDeviceGetUUID.side_effect = get_uuid

        def get_name(handle):
            if handle == handle_A2: return b"NVIDIA A2"
            if handle == handle_L4: return b"NVIDIA L4"
            return b"Unknown"
        mock_pynvml.nvmlDeviceGetName.side_effect = get_name

        # Patch sys.modules
        sys.modules['pynvml'] = mock_pynvml
        sys.modules['torch'] = mock_torch

        # Reload module under test
        if 'tecpg.gpu_monitor' in sys.modules:
            del sys.modules['tecpg.gpu_monitor']

        import tecpg.gpu_monitor as gpu_monitor
        from tecpg.logger import Logger

        log = MagicMock(spec=Logger)
        log.carry_data = {}

        # Test Case 1: CUDA_VISIBLE_DEVICES is unset
        with patch.dict('os.environ'):
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            handle = gpu_monitor.init_gpu_monitor(log)

            # Should select L4 based on iterative UUID matching
            self.assertEqual(handle, handle_L4, "Should select L4 based on UUID matching")
            self.assertTrue(mock_pynvml.nvmlDeviceGetUUID.called)

    def test_name_mismatch_fallback(self):
        """Test fallback when UUID lookup fails AND Name has trailing space."""
        mock_pynvml = MagicMock()
        mock_torch = MagicMock()
        mock_pynvml.NVMLError = MockNVMLError

        UUID_L4 = "GPU-L4-REAL-UUID"
        UUID_A2 = "GPU-A2-REAL-UUID"

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_props = MagicMock()
        mock_props.uuid = UUID_L4
        mock_props.name = "NVIDIA L4"
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.get_device_name.return_value = "NVIDIA L4"

        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        handle_A2 = MagicMock(name="Handle_A2")
        handle_L4 = MagicMock(name="Handle_L4")

        mock_pynvml.nvmlDeviceGetHandleByUUID.side_effect = MockNVMLError("Fail")

        def get_handle_by_index(index):
            if index == 0: return handle_A2
            if index == 1: return handle_L4
            raise ValueError(f"Invalid index {index}")
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle_by_index

        def get_uuid(handle):
            if handle == handle_A2: return UUID_A2.encode('utf-8')
            if handle == handle_L4: return UUID_L4.encode('utf-8')
            return b"Unknown"
        mock_pynvml.nvmlDeviceGetUUID.side_effect = get_uuid

        # Simulate Name Mismatch (Trailing Space)
        def get_name(handle):
            if handle == handle_A2: return b"NVIDIA A2"
            if handle == handle_L4: return b"NVIDIA L4 " # Trailing Space
            return b"Unknown"
        mock_pynvml.nvmlDeviceGetName.side_effect = get_name

        sys.modules['pynvml'] = mock_pynvml
        sys.modules['torch'] = mock_torch

        if 'tecpg.gpu_monitor' in sys.modules:
            del sys.modules['tecpg.gpu_monitor']
        import tecpg.gpu_monitor as gpu_monitor
        from tecpg.logger import Logger
        log = MagicMock(spec=Logger)
        log.carry_data = {}

        handle = gpu_monitor.init_gpu_monitor(log)

        # Should still select L4 because UUID (normalized) matches
        self.assertEqual(handle, handle_L4, "Should select L4 despite name mismatch, via UUID")

if __name__ == '__main__':
    unittest.main()
