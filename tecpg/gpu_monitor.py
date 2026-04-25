import os
import time
import torch
import threading
from contextlib import contextmanager
from .logger import Logger

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class ThermalMonitor:
    def __init__(self, handle, threshold: int, logger: Logger, poll_interval: float = 2.0):
        self.handle = handle
        self.threshold = threshold
        self.logger = logger
        self.poll_interval = poll_interval

        self.last_temp = -1
        self.cool_event = threading.Event()
        self.cool_event.set()

        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self.handle is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        was_hot = False
        while not self._stop_event.is_set():
            temp = get_gpu_temp(self.handle)
            self.last_temp = temp

            if temp > self.threshold:
                if not was_hot:
                    self.logger.warning(f"GPU temperature {temp}C exceeds threshold {self.threshold}C. Throttling active.")
                    was_hot = True
                self.cool_event.clear()
            else:
                if was_hot:
                    self.logger.info(f"GPU temperature dropped to {temp}C. Resuming processing.")
                    was_hot = False
                self.cool_event.set()

            self._stop_event.wait(self.poll_interval)

    def should_throttle(self) -> bool:
        if self.handle is None:
            return False
        return not self.cool_event.is_set()


class DummyThermalMonitor:
    def __init__(self):
        self.handle = None
        self.last_temp = -1
        self.cool_event = threading.Event()
        self.cool_event.set()

    def start(self):
        pass

    def stop(self):
        pass

    def should_throttle(self) -> bool:
        return False


def _normalize_uuid(uuid) -> str:
    """Helper to normalize UUID strings for comparison."""
    if isinstance(uuid, bytes):
        uuid = uuid.decode('utf-8')
    return str(uuid).replace('GPU-', '').replace('MIG-', '').strip().lower()

def _normalize_name(name) -> str:
    """Helper to normalize device name strings for comparison."""
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    return str(name).strip().lower()

def init_gpu_monitor(logger: Logger) -> object:
    """Initializes the GPU monitoring handle. Returns None if unavailable."""
    if not HAS_PYNVML:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return None

        # Get the current device index used by PyTorch
        current_device_idx = torch.cuda.current_device()
        handle = None
        target_uuid = None

        # 1. Attempt to get UUID directly from PyTorch properties (most reliable)
        try:
            props = torch.cuda.get_device_properties(current_device_idx)
            if hasattr(props, 'uuid'):
                target_uuid = props.uuid
        except Exception:
            pass

        # 2. Fallback: Parse CUDA_VISIBLE_DEVICES if UUID not found in properties
        if not target_uuid:
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices:
                ids = [x.strip() for x in cuda_visible_devices.split(',') if x.strip()]
                if len(ids) > current_device_idx:
                    token = ids[current_device_idx]
                    # Check if the token is a UUID (starts with GPU- or MIG-)
                    if token.startswith('GPU-') or token.startswith('MIG-'):
                        target_uuid = token
                    else:
                        # It is a physical index. We can get the handle directly by index.
                        try:
                            physical_index = int(token)
                            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
                        except ValueError:
                            # Could be a UUID without prefix? rare.
                            target_uuid = token

        # 3. Robust Search: Iterate all NVML devices to match UUID or Name
        if not handle:
            norm_target_uuid = _normalize_uuid(target_uuid) if target_uuid else None

            try:
                torch_name = torch.cuda.get_device_name(current_device_idx)
                norm_torch_name = _normalize_name(torch_name)
            except Exception:
                torch_name = "Unknown"
                norm_torch_name = None

            uuid_match = None
            name_candidates = []

            for i in range(device_count):
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)

                    # UUID Check
                    if norm_target_uuid:
                        try:
                            dev_uuid = pynvml.nvmlDeviceGetUUID(h)
                            if _normalize_uuid(dev_uuid) == norm_target_uuid:
                                uuid_match = h
                                break # Exact UUID match is definitive
                        except pynvml.NVMLError:
                            pass

                    # Name Check
                    if norm_torch_name:
                        try:
                            dev_name = pynvml.nvmlDeviceGetName(h)
                            if _normalize_name(dev_name) == norm_torch_name:
                                name_candidates.append(h)
                        except pynvml.NVMLError:
                            pass

                except pynvml.NVMLError as e:
                    logger.warning(f"Error inspecting NVML device {i}: {e}")

            if uuid_match:
                handle = uuid_match
            elif len(name_candidates) == 1:
                handle = name_candidates[0]
            elif len(name_candidates) > 1:
                # Ambiguous name match.
                logger.warning(f"Multiple GPUs match name '{torch_name}'. Using index-based fallback.")

        # 4. Final Fallback: Direct Index
        if not handle:
             try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(current_device_idx)
             except pynvml.NVMLError as e:
                logger.warning(f"Failed to get handle by index {current_device_idx}: {e}")

        # Final Verification / Logging
        if handle:
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes): name = name.decode('utf-8')

                try:
                    uuid = pynvml.nvmlDeviceGetUUID(handle)
                    if isinstance(uuid, bytes): uuid = uuid.decode('utf-8')
                    logger.info(f"Initialized GPU monitoring for device: {name} (UUID: {uuid})")
                except:
                    logger.info(f"Initialized GPU monitoring for device: {name}")
            except:
                logger.info(f"Initialized GPU monitoring.")

        return handle

    except pynvml.NVMLError as e:
        logger.warning(f"Failed to initialize GPU monitoring: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error initializing GPU monitoring: {e}")
        return None

def get_gpu_temp(handle: object) -> int:
    """Returns the current GPU temperature in Celsius. Returns -1 on error."""
    if handle is None:
        return -1
    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return temp
    except pynvml.NVMLError:
        return -1

def report_thermal_status(monitor: object, threshold: int, logger: Logger):
    """Reports the current GPU temperature from the monitor and the thermal threshold."""
    if monitor is None or monitor.handle is None:
        logger.info("GPU Thermal Status: Monitor not active")
        return

    temp = monitor.last_temp
    if temp == -1:
        logger.warning("GPU Thermal Status: Error reading temperature (Threshold: {0}C)", threshold)
    else:
        logger.info("GPU Thermal Status: {0}C (Threshold: {1}C)", temp, threshold)

def throttle_if_needed(monitor: object, threshold: int, wait_time: int, logger: Logger):
    """Sleeps if the monitor indicates the GPU exceeds the thermal threshold."""
    if monitor is None or not monitor.should_throttle():
        return

    try:
        # It's hot! Wait for the event to be set (cooling down)
        # We cap the wait at wait_time just to ensure we periodically wake
        # but cool_event.wait will return immediately when cool.
        monitor.cool_event.wait(timeout=wait_time)
    except Exception as e:
        logger.warning(f"Error during thermal check wait: {e}")

def shutdown_gpu_monitor(monitor: object):
    """Stops the thermal monitor thread and shuts down NVML."""
    if monitor is None:
        return
    monitor.stop()
    handle = monitor.handle
    if handle is None:
        return
    try:
        if HAS_PYNVML:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass

@contextmanager
def gpu_guardian(logger: Logger, thermal_threshold: int = 80):
    """Context manager for GPU monitoring lifecycle."""
    handle = init_gpu_monitor(logger)
    if handle is not None:
        monitor = ThermalMonitor(handle, thermal_threshold, logger)
    else:
        monitor = DummyThermalMonitor()

    monitor.start()
    try:
        yield monitor
    finally:
        shutdown_gpu_monitor(monitor)
