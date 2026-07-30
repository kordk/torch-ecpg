# Docker Image for tecpg Pipeline

This directory contains the `Dockerfile` and related configurations to build the containerized version of the full `tecpg` analysis pipeline.

## Building the Image

To build the image, run the following command from the root of the repository (the `.dockerignore` file must be at the repo root for exclusions to take effect):

```bash
docker build -t tecpg-pipeline -f docker-related/Dockerfile .
```

## Running the Container

The container runs the `tecpg` CLI by default. Because it uses `CMD ["python3", "-m", "tecpg"]` without an `ENTRYPOINT`, you must provide the full command if you wish to override the default. You can mount a local directory for input data, intermediate files, and outputs.

A typical invocation on a GPU-enabled host looks like:

```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline python3 -m tecpg [tecpg_args...]
```

For example, to run the MLR module:
```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline python3 -m tecpg run mlr ...
```

To run a shell inside the container instead of the `tecpg` CLI:

```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline bash
```

### Important Runtime Requirements

1. **NVIDIA Driver Requirement:** The image is built on CUDA 12.4 and pins
   `torch==2.6.0+cu124`. CUDA minor-version compatibility means an `R525+`
   driver is generally sufficient; `R550+` is the version-matched baseline and
   is what klabdev should be held to. Torch is pinned because 2.6.0 is the last
   release published to the cu124 wheel index.
2. **Network Access (Egress):** The pipeline requires internet access to download certain datasets:
   - `tecpg data gtp/mesa` requires access to GEO to fetch raw datasets.
   - `pipelinePost.sh` requires access to UCSC to fetch the `cytoBand.txt` file if it is missing from the working directory.
