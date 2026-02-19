import sys
import unittest
from unittest.mock import MagicMock, patch
import os

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

        # Define UUIDs
        UUID_L4 = "GPU-12345678-1234-1234-1234-123456789abc"
        UUID_A2 = "GPU-87654321-4321-4321-4321-cba987654321"

        # Setup torch mock
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0

        # Scenario: PyTorch Device 0 is the L4 (UUID_L4), even though NVML might index it differently
        # We simulate torch.cuda.get_device_properties returning an object with .uuid
        mock_props = MagicMock()
        mock_props.uuid = UUID_L4
        mock_props.name = "NVIDIA L4"
        mock_torch.cuda.get_device_properties.return_value = mock_props

        # Setup pynvml mock
        mock_pynvml.nvmlDeviceGetCount.return_value = 2

        # Device Handles
        handle_A2 = MagicMock(name="Handle_A2")
        handle_L4 = MagicMock(name="Handle_L4")

        # Mock pynvml.nvmlDeviceGetHandleByUUID
        def get_handle_by_uuid(uuid):
            if isinstance(uuid, bytes):
                uuid = uuid.decode('utf-8')
            if uuid == UUID_L4: return handle_L4
            if uuid == UUID_A2: return handle_A2
            raise Exception(f"Unknown UUID: {uuid}")

        mock_pynvml.nvmlDeviceGetHandleByUUID.side_effect = get_handle_by_uuid

        # Also mock iteration just in case code falls back to it
        def get_handle_by_index(index):
            if index == 0: return handle_A2  # NVML 0 is A2
            if index == 1: return handle_L4  # NVML 1 is L4
            raise ValueError(f"Invalid index {index}")
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle_by_index

        def get_uuid(handle):
            if handle == handle_A2: return UUID_A2.encode('utf-8') # pynvml returns bytes
            if handle == handle_L4: return UUID_L4.encode('utf-8')
            return b"Unknown"
        mock_pynvml.nvmlDeviceGetUUID.side_effect = get_uuid

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
        # We expect it to prioritize UUID from PyTorch properties
        with patch.dict('os.environ'):
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            handle = gpu_monitor.init_gpu_monitor(log)

            # Should select L4 because PyTorch said its UUID is UUID_L4
            self.assertEqual(handle, handle_L4, "Should select L4 based on UUID matching")

            # Verify it actually called get_device_properties(0)
            mock_torch.cuda.get_device_properties.assert_called_with(0)

            # Verify it tried to get handle by UUID
            # The argument to nvmlDeviceGetHandleByUUID might be bytes or string depending on implementation
            # We check if it was called at all
            self.assertTrue(mock_pynvml.nvmlDeviceGetHandleByUUID.called)

if __name__ == '__main__':
    unittest.main()
