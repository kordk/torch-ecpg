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

        # Get the current device index used by PyTorch
        current_device_idx = torch.cuda.current_device()
        handle = None
        target_uuid = None

        # 1. Attempt to get UUID directly from PyTorch properties (most reliable)
        try:
            props = torch.cuda.get_device_properties(current_device_idx)
            if hasattr(props, 'uuid'):
                target_uuid = props.uuid
                # PyTorch might return UUID with 'GPU-' prefix or not.
                # pynvml expects standard UUID string.
                # Typically UUIDs look like "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        except Exception as e:
            # logger.debug(f"Could not get UUID from torch properties: {e}")
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
            else:
                # No env var, and no UUID from props.
                # This is the tricky case where PyTorch index != NVML index potentially.
                # But without UUID, we can't do much better than index or name matching.
                pass

        # 3. If we have a target UUID, try to get handle by UUID
        if not handle and target_uuid:
            try:
                # Ensure it's a string
                uuid_str = str(target_uuid)
                # pynvml.nvmlDeviceGetHandleByUUID expects a string (str in Py3)
                # Some old versions might behave differently, but generally it takes str.
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid_str)
            except pynvml.NVMLError as e:
                # Try adding or removing "GPU-" prefix if failed
                try:
                    if uuid_str.startswith("GPU-"):
                        alt_uuid = uuid_str[4:]
                    else:
                        alt_uuid = "GPU-" + uuid_str
                    handle = pynvml.nvmlDeviceGetHandleByUUID(alt_uuid)
                except pynvml.NVMLError:
                    logger.warning(f"Could not find NVML device with UUID {target_uuid}. Error: {e}")

        # 4. If we still don't have a handle (no UUID found or UUID lookup failed)
        if not handle:
             # Fallback to Name Matching (for distinct GPUs) or Index Matching
            try:
                torch_name = torch.cuda.get_device_name(current_device_idx)
                matched_handle = None

                # Check for name match
                # Warning: If multiple GPUs have the same name, this might pick the wrong one
                # if indices are also swapped. But it's better than nothing.
                candidates = []
                for i in range(device_count):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    nvml_name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(nvml_name, bytes):
                        nvml_name = nvml_name.decode('utf-8')
                    if nvml_name == torch_name:
                        candidates.append(h)

                if len(candidates) == 1:
                    matched_handle = candidates[0]
                elif len(candidates) > 1:
                    # Ambiguous name match. If CUDA_VISIBLE_DEVICES is unset,
                    # we might assume PyTorch index maps to one of these?
                    # But we don't know which one.
                    # Fallback to direct index mapping if indices are within range?
                    # Or just pick the one with matching index if available?
                     logger.warning(f"Multiple GPUs match name '{torch_name}'. Using index-based fallback.")
                     pass

                if matched_handle:
                    handle = matched_handle
                else:
                    # Final fallback: Direct Index
                    handle = pynvml.nvmlDeviceGetHandleByIndex(current_device_idx)

            except Exception as e:
                logger.warning(f"Error during device matching: {e}. Falling back to index {current_device_idx}.")
                handle = pynvml.nvmlDeviceGetHandleByIndex(current_device_idx)

        # Final Verification / Logging
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        try:
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode('utf-8')
            logger.info(f"Initialized GPU monitoring for device: {name} (UUID: {uuid})")
        except:
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
