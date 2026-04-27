# Profiling Torch-eCpG GPU Bottlenecks

`profiling.sh` is a diagnostic tool designed to localize the root causes of GPU underutilization during `tecpg run mlr` workloads. It runs the regular Torch-eCpG CLI with specific debug logging enabled, captures host/GPU performance metrics, and bundles them into an archive for analysis.

**Crucially**, `profiling.sh` does not modify any `tecpg` code or its behavior; it simply wraps the existing commands and collects comprehensive debug and system-level telemetry.

## Quickstart

To run a fast smoke test (takes ~60 seconds) using synthetic data:
```bash
./profiling.sh -d dummy
```

To run a full diagnostic workload using realistic data and iterate through performance knobs (capping each run at 90s):
```bash
./profiling.sh -d gtp -D 600 --matrix
```

When it finishes, `profiling.sh` outputs an absolute path to a `.tar.gz` bundle with the collected data.

## The Output Bundle

The output archive contains `env.txt` at the root, and one or more run subdirectories (e.g. `run/` or `baseline/`, `prefetch/`, etc. when `--matrix` is used). Each subdirectory includes:

* **`cell.txt`**: The specific tecpg CLI command and environment variables used for that run.
* **`tecpg.log`**: Standard output and error emitted by `tecpg`.
* **`chunk_profile.tsv`**: Extracted granular debug metrics for every chunk processed.
* **`chunk_profile_summary.txt`**: Computed p50/p90/p99 latency quantiles across all chunks (prep, H2D, compute, D2H, post, save queue) and an overall bottleneck verdict.
* **`nvidia-smi-query.csv`**: Sub-second telemetry (GPU util, mem util, temperatures, and throttle/power capping reasons).
* **`nvidia-smi-dmon.csv`**: 1-second cadence hardware utilization monitoring.
* **`pidstat.csv`**: System-level CPU, I/O, and context switches for processes (important for host-bound diagnoses).
* **`iostat.txt` / `vmstat.txt` / `top.txt`**: Basic host/OS health metrics covering I/O wait and context switches.
* **`matrix_summary.csv`** (Only if `--matrix`): A roll-up table comparing overall throughput and metrics across varying configurations.
* **`nsys/` & `pyspy.svg`** (Optional, Baseline Only): NVIDIA Nsight Systems timeline and CPU py-spy flamegraphs, if those profilers were installed.

## Reading the Metrics

When reviewing a profile, start with `chunk_profile_summary.txt` and map the symptoms to likely causes:

| Symptom | Probable Cause | Next Step / Mitigation |
| ------- | -------------- | ---------------------- |
| **GPU starved (host-bound)**: `idle_ms` > 0.5 * `gpu_ms`, util near 0 | Producer thread or Save thread is bottle-necked. CPU is too slow or waiting on I/O. | Check `pidstat.csv` for 100% CPU on single threads; try `--prefetch-chunks 4` or check I/O save paths. |
| **H2D bound**: `h2d_ms` > 40% of total | PCIe bandwidth limit transferring Data to GPU. | Increase chunk size (`-g`, `-m`); Check `nvidia-smi topo -m` for PCIe link width issues. |
| **D2H/save bound**: `d2h_ms` + `write_ms` > 40% of total | Moving output tensors off GPU or enqueueing them to disk is too slow. | Slow save drive (check disk IO probe in `env.txt` / `iostat.txt`). Reduce `--output-dir` depth/latency. |
| **Kernel-launch bound**: `gpu_ms` < 2ms, high chunk/sec | Kernels execute faster than the CPU can schedule them. | Increase batch dimensions (`-g`, `-s` chunk size) to give the GPU more work per invocation. |
| **Compute bound**: `gpu_ms` > 60% of total, high GPU util | The GPU is legitimately fully utilized crunching matrices. | Turn on TF32 (`NVIDIA_TF32_OVERRIDE=1`), use a faster GPU. |
| **Thermal/power throttled**: >10% of samples in `nvidia-smi-query.csv` are throttled | Hardware is capping frequency due to heat/power constraints. | Check cooling; check `clocks_throttle_reasons.active` for specifics. |
| **NVML mapping confusion**: `util_sm` is 0 but `gpu_ms` is high | GPU tasks are executing on an unmonitored device. | Verify `CUDA_VISIBLE_DEVICES` matches index in `nvidia-smi-query.csv`. |

*Note on L4 Compute:* The NVIDIA L4 contains 24 GB of memory and is capable of ~30 TFLOPS FP32 and ~120 TFLOPS TF32. Reviewers can cross-reference the `gflops` outputs in `tecpg.log` to determine how close the workflow gets to the theoretical compute ceiling.
