# Docker Rebuild Design Memo

This document outlines the design and considerations for rebuilding the base Docker image against the current `dev` branch.

## 1. Current State
The existing Dockerfile (`docker-related/Dockerfile`) is built on `python:3.10.13-bullseye` (Debian 11). It attempts to provide GPU support by installing PyTorch via a wheel with the `cu118` index (`https://download.pytorch.org/whl/cu118`).

**Identified Issues & Fragilities:**
- **OS/GPU Misalignment:** A bare Python Debian image does not include proper CUDA libraries or drivers natively. Relying solely on the PyTorch wheel for CUDA works for PyTorch, but doesn't guarantee system-level compatibility or provide tools like `nvcc` for compiling other GPU-accelerated extensions.
- **Redundant `COPY` Layout:** The Dockerfile performs:
  1. `COPY ./proj/torch-ecpg/requirements.txt .` and others from a hardcoded host path (`./proj/torch-ecpg/`) which makes the build highly dependent on a specific directory structure outside the typical build context.
  2. `COPY tecpg .` followed by `COPY ./proj .`, and eventually a broad `COPY . .`. This is fragmented, breaks layer caching unnecessarily, and is poorly scoped.
- **`CMD` Ambiguity:** The command `CMD [ "python", "./tecpg" ]` expects `./tecpg` to be a runnable script/module at the root, but depending on the copy steps, this might conflict with the `tecpg/` directory.
- **No Multi-stage Build:** Build dependencies remain in the final image, increasing size unnecessarily.

## 2. Dependency Inventory

### Python Packages
Based on imports found across the `.py` scripts compared against `requirements.txt`.
*(Note: standard-library imports such as `os`, `sys`, `json` were excluded; only third-party distributions are listed.)*

| Import Name | PyPI Distribution | In `requirements.txt`? | Notes |
| :--- | :--- | :--- | :--- |
| `captum` | `captum` | Yes | |
| `click` | `click` | Yes | |
| `colorama` | `colorama` | Yes | |
| `fa2` | `fa2` | Yes | **Fragile:** Cython package requiring a build toolchain. |
| `GEOparse` | `GEOparse` | Yes | |
| `gseapy` | `gseapy` | Yes | |
| `jinja2` | `jinja2` | No | Inferred via pandas/templating, but missing explicit requirement. |
| `matplotlib` | `matplotlib` | Yes | |
| `matplotlib_venn` | `matplotlib-venn`| Yes | |
| `mygene` | `mygene` | Yes | |
| `networkx` | `networkx` | Yes | |
| `numpy` | `numpy` | Yes | |
| `pandas` | `pandas` | Yes | |
| `patsy` | `patsy` | No | Often a dependency of `statsmodels`. |
| `polars` | `polars` | Yes | |
| `psutil` | `psutil` | Yes | |
| `pyarrow` | `pyarrow` | Yes | |
| `pycircos` | `pycircos` | Yes | |
| `pyliftover` | `pyliftover` | Yes | |
| `pynvml` | `nvidia-ml-py` | Yes | Note: module `pynvml` maps to `nvidia-ml-py` |
| `pyranges` | `pyranges` | Yes | |
| `pytest` | `pytest` | No | |
| `requests` | `requests` | Yes | |
| `scipy` | `scipy` | Yes | |
| `seaborn` | `seaborn` | Yes | |
| `setuptools` | `setuptools` | Yes | |
| `sklearn` | `scikit-learn` | No | Explicit import `sklearn` missing from requirements. |
| `statsmodels`| `statsmodels` | Yes | |
| `torch` | `torch` | Yes | |
| `umap` | `umap-learn` | Yes | Note: module `umap` maps to `umap-learn` |
| `upsetplot` | `upsetplot` | Yes | |

### R Packages
Identified from `tools/install_dependencies.R`:
- **CRAN:** `pheatmap`
- **Bioconductor:** `EpiDISH`, `sva`, `IlluminaHumanMethylationEPICanno.ilm10b4.hg19`, `ExperimentHub`

### System Packages / Binaries
**Observed Binaries** (called in `pipeline.sh`, `pipelinePost.sh`, `tools/*.sh`):
- `curl`, `gunzip`, `sed`, `awk`, `cut`, `tail`, `Rscript`, `python3`

**Inferred Build Toolchains** (needed for source compilation of `fa2` and R packages; candidates for a build stage only):
- `build-essential` (gcc/g++/make) — for `fa2`'s Cython compile and R source packages.
- `libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev` — required by R packages when building from source.
- `python3-dev` — headers for any Cython build against the Python C API.
- `r-base` / `r-base-dev` (or the CRAN apt repo distribution).

## 3. Base-image Decision
**Recommendation:** **Option B - Move to an NVIDIA CUDA base image.**

**Concrete Image:** `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`

**Justification:**
- **CUDA/PyTorch Continuity:** Selecting CUDA 11.8 specifically matches the `cu118` PyTorch wheel currently used. This maintains a known-good Torch/CUDA pairing while ensuring the system-level CUDA libraries align perfectly with the framework, without introducing new variables.
- **Robust Environment:** Ubuntu 22.04 provides a modern glibc era which ensures compatibility for prebuilt Python wheels (e.g., `umap-learn`, `PyTorch`). It also has a much more robust packaging story for R and Bioconductor binaries compared to a bare Python Debian image.
- **Driver Compatibility:** This CUDA 11.8 base requires an NVIDIA driver of roughly `R520+`. This is a reasonable baseline but should be documented for users running older drivers.
- **Runtime Tradeoffs:** Runtime base images omit `nvcc` and CUDA headers, which is fine since `torch` is installed via a pip wheel. However, because compiling `fa2` and R packages requires a C toolchain, `build-essential` and development libraries will need to be added via `apt` in a build stage and excluded from the final image using a multi-stage build.

**Estimated Size:** ~4-6GB compressed (heavily dependent on PyTorch, CUDA runtime, R, and Bioconductor packages).

## 4. Proposed Structure
**Multi-stage Build:**
- **Stage 1 (Builder):** Based on the selected NVIDIA image. Installs `python3`, `r-base`, `build-essential`, `python3-dev`, and system libraries (`libcurl4-openssl-dev`, `libssl-dev`, `libxml2-dev`). Compiles Python packages (like `fa2`) and installs R/Bioconductor source packages into isolated environments/paths.
- **Stage 2 (Runtime):** Also based on the NVIDIA runtime image. Copies over the compiled Python/R libraries from the Builder stage. Installs only runtime dependencies (`python3`, `r-base`, `curl`, `awk`, etc.), keeping the image slimmer by omitting compilers and headers.
- **Layout Fixes:** Use a standard `WORKDIR /app`. Copy only `requirements.txt` first to cache pip installs, followed by `tools/install_dependencies.R` to cache R packages, and finally `COPY . .`. Set the `ENTRYPOINT` or `CMD` explicitly to `["python3", "-m", "tecpg"]` or `["bash"]`.

**Volumes and Runtime Data:**
- Users will be expected to bind-mount a host directory to a conventional working path, e.g., `/work`.
- Datasets, outputs, and annotations (e.g., `data_<dataset>/`, `output_<dataset>/`, `annot_<dataset>/`) should reside in this volume. This allows the host to persist outputs and reuse downloaded matrices, skipping redundant downloads via the existing script logic.

**Network:**
- **Egress:** The image requires runtime internet access for two main tasks:
  1. `tecpg data gtp/mesa` fetching data from GEO.
  2. `pipelinePost.sh` downloading `cytoBand.txt` from UCSC.
- **Recommendation:** Future test sets should vendor a stub `cytoBand.txt` at the dummy path to prevent test suites from failing if UCSC goes down.

**`.dockerignore`:**
```dockerignore
# Project Specific
data_*
output_*
annot_*
GTP/
MESA/
*/GTP/
cytoBand.txt

# Logs
*.log

# Data files at root
*.parquet
*.csv

# Python/Environment
.venv/
__pycache__/
*.egg-info/
.pytest_cache/

# Git
.git/

# Test data (Ensure test automation scripts reside outside the build context)
test/
```

## 5. Open Questions / Risks
- **R Installation Method on Ubuntu 22.04:** It remains an open question whether `r-base` should be installed directly from the Ubuntu 22.04 LTS package repositories or via the CRAN maintained `apt` repository. The latter provides more recent R versions, but the former might be sufficient and more stable for our specific Bioconductor dependencies.
- **Build Times & Fragility:** Compiling Bioconductor packages and `fa2` (Cython) from source is historically slow and sensitive to compiler/library versions. This risk is mitigated by using the multi-stage build, but build times could still be significant.
- **NVIDIA Driver Requirements:** Users must have driver `R520+` to support the CUDA 11.8 runtime container. Older hosts will fail, which is a known and accepted support boundary but should be flagged in user documentation.
