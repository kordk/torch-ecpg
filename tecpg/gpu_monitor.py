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

def report_thermal_status(handle: object, threshold: int, logger: Logger):
    """Reports the current GPU temperature and thermal threshold."""
    if handle is None:
        logger.info("GPU Thermal Status: Monitor not active")
        return

    temp = get_gpu_temp(handle)
    if temp == -1:
        logger.warning("GPU Thermal Status: Error reading temperature (Threshold: {0}C)", threshold)
    else:
        logger.info("GPU Thermal Status: {0}C (Threshold: {1}C)", temp, threshold)

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
