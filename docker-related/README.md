# Docker Image for tecpg Pipeline

This directory contains the `Dockerfile` and related configurations to build the containerized version of the full `tecpg` analysis pipeline.

## Building the Image

To build the image, run the following command from the root of the repository:

```bash
docker build -t tecpg-pipeline -f docker-related/Dockerfile .
```

## Running the Container

The default command of the container acts as the `tecpg` CLI. You can mount a local directory for input data, intermediate files, and outputs. A typical invocation on a GPU-enabled host looks like:

```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline [tecpg_args...]
```

To run a shell inside the container instead of the `tecpg` CLI:

```bash
docker run --rm -it --gpus all -v /path/to/host/dir:/work tecpg-pipeline bash
```

### Important Runtime Requirements

1. **NVIDIA Driver Requirement:** The underlying CUDA 11.8 base image requires the host to have an NVIDIA driver of at least roughly `R520+`.
2. **Network Access (Egress):** The pipeline requires internet access to download certain datasets:
   - `tecpg data gtp/mesa` requires access to GEO to fetch raw datasets.
   - `pipelinePost.sh` requires access to UCSC to fetch the `cytoBand.txt` file if it is missing from the working directory.
