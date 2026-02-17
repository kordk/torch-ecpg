import os
import time
import torch
from contextlib import contextmanager
from .logger import Logger

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

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

        # Get handle for the current device being used by PyTorch
        current_device = torch.cuda.current_device()

        # Handle CUDA_VISIBLE_DEVICES mapping
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            ids = [x.strip() for x in cuda_visible_devices.split(',') if x.strip()]
            if len(ids) > current_device:
                try:
                    # Try to parse as integer index
                    real_index = int(ids[current_device])
                    handle = pynvml.nvmlDeviceGetHandleByIndex(real_index)
                except ValueError:
                    # Could be UUID, which is more complex, fallback to default logic
                    # or try getting by UUID if format matches
                    handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(current_device)

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        logger.info(f"Initialized GPU monitoring for device: {name}")
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

def throttle_if_needed(handle: object, threshold: int, wait_time: int, logger: Logger):
    """Checks GPU temperature and sleeps if it exceeds the threshold."""
    if handle is None:
        return

    try:
        temp = get_gpu_temp(handle)
        if temp > threshold:
            logger.warning(f"GPU temperature {temp}C exceeds threshold {threshold}C. Throttling for {wait_time}s...")
            time.sleep(wait_time)

    except Exception as e:
        logger.warning(f"Error during thermal check: {e}")

def shutdown_gpu_monitor(handle: object):
    """Shuts down NVML."""
    if handle is None:
        return
    try:
        if HAS_PYNVML:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        pass

@contextmanager
def gpu_guardian(logger: Logger):
    """Context manager for GPU monitoring lifecycle."""
    handle = init_gpu_monitor(logger)
    try:
        yield handle
    finally:
        shutdown_gpu_monitor(handle)
