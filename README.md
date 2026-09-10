# tecpg

Torch-eCpG (`tecpg`) is a GPU-enabled expression quantitative trait
methylation (eQTM) mapper. It identifies expression-associated CpG (eCpG)
loci by regressing gene expression on methylation and covariates for every
CpG–transcript pair, using PyTorch batched tensor operations, and is driven
from a Python command-line interface.

**Which part of this README do you need?**

* **You have your own methylation, expression, and covariate matrices** and
  want to map eQTMs → [Part A — Core `tecpg` tool](#part-a--core-tecpg-tool).
  Start with the [5-minute quick start](#quick-start-5-minutes-no-downloads).
* **You want to see the complete worked pipeline** (preprocessing, mapping,
  FDR, bootstrap, permutation, figures) on public data →
  [Part B — Demonstration](#part-b--demonstration-the-gtpmesa-golden-path).

## Citation

If you use Torch-eCpG in your research, please cite:

> Kober, K.M., Berger, L., Roy, R. et al. Torch-eCpG: a fast and scalable eQTM
> mapper for thousands of molecular phenotypes with graphical processing
> units. *BMC Bioinformatics* 25, 71 (2024).
> https://doi.org/10.1186/s12859-024-05670-4

Torch-eCpG v2 was presented as a poster: Kober, K.M., Rau, A., Olshen, A.
*Torch-eCpG v2: A Scalable and Interpretable Framework for eQTM Mapping and
Multi-Omic Network Analysis.* 21st International Conference on Computational
Intelligence Methods for Bioinformatics and Biostatistics (CIBB 2026), Rome,
2–4 September 2026.

## Version

The current version on the `dev` branch is **2.0.0b2** (tag `v2.0.0-beta.2`).
The `main` branch carries the published v1 (`1.0.0`). Version strings follow
[PEP 440](https://peps.python.org/pep-0440/). See
[`CHANGELOG.md`](CHANGELOG.md) for what has changed between releases.

## Table of contents

**Part A — Core `tecpg` tool**

* [Installation](#installation)
* [Quick start (5 minutes, no downloads)](#quick-start-5-minutes-no-downloads)
* [CUDA and host profiles](#cuda-and-host-profiles)
* [Input data](#input-data)
* [Output](#output)
* [Chunking](#chunking)
* [Filtration](#filtration)
* [Running `tecpg run mlr` directly](#running-tecpg-run-mlr-directly)
* [Selecting a GPU when multiple are available](#selecting-a-gpu-when-multiple-are-available)
* [Performance tuning](#performance-tuning)
* [Documentation](#documentation)

**Part B — Demonstration: the GTP/MESA golden path**

* [Demo datasets (GTP and MESA)](#demo-datasets-gtp-and-mesa)
* [Quick start (the golden path)](#quick-start-the-golden-path)
* [Pipeline stages](#pipeline-stages)
* [Alternative annotation and assignment of regions](#alternative-annotation-and-assignment-of-regions)
* [Tools and helper scripts](#tools-and-helper-scripts)
* [Tests](#tests)
* [Acknowledgements](#acknowledgements)

# Part A — Core `tecpg` tool

The sections below document the `tecpg` command-line tool and library on
their own, independent of any particular dataset. Part A needs only Python;
no R, no downloads.

## Installation

Requirements: **Python 3.10 or newer**. A GPU is optional — `tecpg` runs on
CPU, but for GPU acceleration you need a CUDA-capable NVIDIA GPU and a
CUDA-enabled PyTorch build (see the note below).

Install the published version from the `main` branch:

```bash
pip install git+https://github.com/kordk/torch-ecpg.git
```

Install the current development version (v2) from the `dev` branch:

```bash
pip install git+https://github.com/kordk/torch-ecpg.git@dev
```

For development or debugging, install in editable mode and keep the clone in
place:

```bash
git clone https://github.com/kordk/torch-ecpg.git
cd torch-ecpg
git checkout dev          # optional
pip install --editable .
```

If the installation succeeded, `tecpg --help` prints the command-line help.
If `pip` is not on your `PATH`, use `python -m pip` or `python3 -m pip`.

> **PyTorch and CUDA.** `pip` installs whatever `torch` wheel is the default
> for your platform, which may be CPU-only. If `tecpg` reports that CUDA is
> unavailable on a machine that has an NVIDIA GPU, install a CUDA-enabled
> PyTorch build first using the selector at https://pytorch.org/get-started/locally/
> (matching your driver's CUDA version), then install `tecpg`. You can check
> with `python -c "import torch; print(torch.cuda.is_available())"`.

### R dependencies (pipeline and tools only)

The core `tecpg` CLI is pure Python and needs nothing beyond the `pip` install
above. The `pipeline*.sh` scripts in Part B and several of the helpers in
`tools/` call R, so if you intend to use any of them, install the R packages
once from the repository root with an R installation already on your `PATH`:

```bash
Rscript tools/install_dependencies.R
```

This installs `BiocManager` if it is absent, then `pheatmap` (CRAN) and
`EpiDISH`, `sva`, `ExperimentHub`, `minfi` and both Illumina array manifests
(`IlluminaHumanMethylation450kanno.ilmn12.hg19` and
`IlluminaHumanMethylationEPICanno.ilm10b4.hg19`) from Bioconductor. Packages
that are already present are skipped, so the script is safe to re-run; it exits
non-zero and names any package that could not be loaded afterwards.

The annotation packages are large and Bioconductor builds several of these from
source, so allow ten minutes or more on a first run.

### Docker

A containerized build of the full pipeline is defined in `docker-related/`
(`Dockerfile` on the `nvidia/cuda:12.4.1` runtime base). The image installs R
and runs `tools/install_dependencies.R` during the build, so the R step above
is only needed for a native install. Build it from the repository root so the
`.dockerignore` exclusions apply:

```bash
docker build -t tecpg-pipeline -f docker-related/Dockerfile .
```

See [`docker-related/README.md`](docker-related/README.md) for running the
image and saving/loading it. A pre-built image of the **published v1** (not
v2) is on Docker Hub at https://hub.docker.com/r/kordk/torch-ecpg.

## Quick start (5 minutes, no downloads)

Generate a small synthetic dataset and run a cis eQTM scan. This needs no R,
no GPU, and no external data, and is the fastest way to confirm that `tecpg`
is installed and working.

```bash
mkdir tecpg-demo && cd tecpg-demo

# 1. Generate synthetic data: 100 samples, 2,000 CpGs, 500 expression probes
tecpg data dummy --samples 100 --meth-rows 2000 --gene-rows 500 --seed 42

# 2. Regress every probe on every CpG, keeping pairs with p < 0.05
tecpg run mlr --all --p-thresh 0.05
```

Step 1 writes `data/M.csv`, `data/G.csv`, `data/C.csv` and
`annot/M.bed6`, `annot/G.bed6` under the current directory. Step 2 fits
1,000,000 regressions (2,000 CpGs × 500 probes) and writes the ~5% that pass
the p-value gate to `output/` — a single `out.parquet` when the data fits in
one chunk, or per-chunk files otherwise (see [Output](#output)).

Peek at the results:

```bash
python -c "import pandas as pd; print(pd.read_parquet('output/out.parquet').head())"
```

Each row is one CpG–probe pair; `mt_est`, `mt_err`, `mt_t`, and `mt_p` are the
methylation coefficient, its standard error, t-statistic, and p-value. The
dummy data is random, so the associations are meaningless — this run checks
wiring only. (Its BED6 positions are random too, which is why the example
uses `--all` rather than `--cis`: almost no random CpG lands within the cis
window of a random probe.)

**Next steps**

* Run on your own data: replace the files in `data/` and `annot/` with your
  matrices in the formats described under [Input data](#input-data), or point
  `tecpg` at other directories with `-i` (input), `-a` (annotation), and `-o`
  (output). A typical cis scan on real data is:

  ```bash
  tecpg -i /path/to/data -a /path/to/annot -o /path/to/output run mlr --cis
  ```

* The default p-value gate is `--p-thresh 0.001`; tighten it (e.g. `1e-5`)
  for genome-wide `--all` scans on real data to keep the output manageable.
* Add per-feature attribution and an influence screen:
  `tecpg run mlr --cis --compute-ig --compute-influence`.
* See `tecpg run mlr --help` for the complete option list.
* To see the full worked pipeline (preprocessing through figures), continue
  to [Part B](#part-b--demonstration-the-gtpmesa-golden-path).

## CUDA and host profiles

`tecpg` can calculate on the CPU or on a CUDA-enabled GPU. GPUs are generally
much faster than CPU for sufficiently large inputs.

The program automatically detects a CUDA device and uses it if available. To
force CPU computation, set `--threads` to a nonzero integer; this also sets the
number of CPU threads used.

The top-level CLI also accepts `--host-profile {auto,minimum,server}` (envvar
`TECPG_HOST_PROFILE`). `auto` (default) inspects the host (physical CPU count
and total RAM) and picks `minimum` for laptop-class hosts (`<12` cores or
`<32 GB`) and `server` otherwise. The resolved profile drives defaults for the
save pool, prefetch depth, and chunk auto-sizing. Explicit per-flag overrides
(`--save-threads`, `--output-format`, `--prefetch-chunks`,
`--gene-loci-per-chunk`, `--meth-loci-per-chunk`, `--blas-threads`) always win.

## Input data

Methylation values, gene expression values, and covariates are provided as CSV
or TSV files in the `<working>/data` directory. For methylation and gene
expression, columns are samples and rows are loci. For covariates, columns are
covariates and rows are samples. Sample identifiers must match across the
three files.

Annotation files are used for region filtration and live in
`<working>/annot`. They use the `BED6` format and store the positions of the
methylation and gene expression loci.

> **Note:** The `M.csv` / `G.csv` / `C.csv` / BED6 snippets below are taken
> from the GTP demo dataset purely as examples of the expected formats. The
> demo datasets themselves, and the `pipeline*.sh` scripts that produce these
> files, are documented in Part B.

Methylation (`data/M.csv`) — rows are CpGs, columns are samples:

```bash
head -5 data/M.csv | cut -d, -f1-5
```
```
,5881,5896,5915,5949
cg00000029,0.551142626425936,0.606679809418831,0.593760482022385,0.554829598676022
cg00000108,0.998563692332771,0.9979593001545,0.997893371350954,0.997293677663346
cg00000165,0.266529984719736,0.159711109475489,0.145981687514545,0.100000350688528
cg00000236,0.812799925026805,0.897011511592051,0.908067942964869,0.863719773724759
```

Gene expression (`data/G.csv`) — rows are expression probes, columns are
samples:

```bash
head -5 data/G.csv | cut -d, -f1-5
```
```
,5881,5896,5915,5949
ILMN_1762337,43.10106,48.30485,37.49239,43.99564
ILMN_2055271,61.09617,61.84258,47.78094,49.32763
ILMN_1736007,51.30634,45.80393,45.43285,40.39254
ILMN_2383229,48.15523,42.69902,35.71749,39.52501
```

Covariates (`data/C.csv`) — rows are samples, columns are covariates:

```bash
head -5 data/C.csv
```
```
,Sex,age
5881,1,44
5896,1,50
5915,0,52
5949,1,56
```

Annotation BED6 files for the gene expression and methylation loci (here, the
Illumina HumanHT-12 and MethylationEPIC arrays):

```bash
head -5 annot/*
```
```
==> annot/G.bed6 <==
chrom   chromStart      chromEnd        name            score   strand
2       128604584       128604633       ILMN_1792672    0       -
11      193773          193822          ILMN_3237022    0       +
13      44410552        44410601        ILMN_1904052    0       -
17      79524173        79524222        ILMN_1807600    0       -

==> annot/M.bed6 <==
chrom   chromStart      chromEnd        name            score   strand
20      61847650        61847650        cg18478105      0       -
X       24072640        24072640        cg09835024      0       -
9       131463936       131463936       cg14361672      0       +
17      80159506        80159506        cg01763666      0       +
```

Example data can be generated or downloaded with `tecpg data`:

```
$ tecpg data --help
Usage: tecpg data [OPTIONS] COMMAND [ARGS]...

  Base group for data management.

Commands:
  dummy  Generates dummy data.
  gtp    Downloads and extracts GTP data.
  mesa   Downloads and extracts MESA data.
```

`tecpg data dummy` prompts for `--samples`, `--meth-rows`, and `--gene-rows`
if they are not supplied; pass `--seed` for reproducible output and
`--no-annotation` to skip the BED6 files. See
[Demo datasets](#demo-datasets-gtp-and-mesa) for the two real-world datasets.

## Output

By default the output format is Parquet. `--output-format {auto,csv,parquet}`
overrides this; `auto` resolves to `parquet` on every host profile.

For `tecpg run mlr` without chunking, a single output file (`out.csv` or
`out.parquet`) is created in the output directory. With chunking on either
axis, per-chunk files named `{methylation chunk number}-{gene expression
chunk number}.{csv,parquet}` are written instead, and a sidecar
`sample_reservoir.csv` of unfiltered draws is produced for diagnostics.
`tools/mergeOutputs.py` combines the chunks into a single Parquet (or CSV)
file and skips `sample_reservoir.csv`.

Row labels identify the gene expression id and the methylation id. Column
labels follow one convention: methylation-related columns are prefixed `mt_`
and gene-expression-related columns `gt_`. For each regression the columns are
the estimate `est`, standard error `err`, Student's t statistic `t`, and
p-value `p` (e.g. `mt_est`, `mt_err`, `mt_t`, `mt_p`). With `--compute-ig`,
integrated-gradients saliency values are written alongside the regression
results, and `--compute-influence` adds the per-CpG maximum sample leverage
`mt_h_max`.

The Part B pipeline adds further columns after mapping: the high-precision
p-value (`precise_mt_p`), the assigned region (`region`), the global BH-FDR
q-value (`fdr_est`), the influence flag (`mt_influence_flag`), the empirical
bootstrap p-value `p_boot` with its `boot_seed`, and — from
`pipelinePermute.sh` — the permutation p-value `p_permute`, its BH q-value
`fdr_permute`, and the `perm_seed` / `perm_n_perm` provenance columns, so any
resampled result can be reproduced from the catalog alone.

## Chunking

If the input is too large, the computational device may run out of memory.
Chunking partitions the data into pieces that are computed and saved
separately. It trades parallelism (and therefore speed) for lower memory, so
avoid it where possible.

`tecpg run mlr` supports two kinds of chunking: methylation chunking and gene
expression chunking. Gene expression chunking is preferable where possible, as
it sacrifices less parallelism.

On server-class hosts, when neither `--gene-loci-per-chunk` nor
`--meth-loci-per-chunk` is supplied, the CLI picks both automatically from the
live RAM/GPU budget (80% target). Supplying exactly one of the two flags pins
that axis and auto-derives the other by bisection against the in-memory
peak-memory estimator (**anchored mode**). The auto-sizer accounts for
`--compute-ig` and applies a safety clamp on tight (~24 GB) VRAM. Minimum-class
hosts never auto-set chunk sizes — supply the flags explicitly there:

```bash
tecpg run mlr --cis --gene-loci-per-chunk 10000 --meth-loci-per-chunk 10000
```

**Note:** these two options accept only their long forms under
`tecpg run mlr` (the `-g` / `-m` short forms belong to the top-level
`--gene-file` / `--meth-file` options). The `data dummy` and `chunks`
subcommands keep their own `-g` / `-m` short flags.

## Filtration

You may want to include only certain regression results. There are two ways
of filtering:

1. **P-value filtration** — all p-values are computed first; results with a
   p-value above the supplied threshold (`--p-thresh`) are excluded from the
   output. This decreases output size, and therefore time, since saving is
   expensive.
2. **Region filtration** — requires the BED6 annotation files that give the
   positions of methylation and gene expression loci. Regressions are filtered
   by one of:
   * **Cis** (`--cis`): the CpG lies within a window of a configurable number
     of bases upstream and downstream of the gene's transcript start site, on
     the same chromosome.
   * **Distal** (`--distal`): the same logic as cis with different default
     window parameters.
   * **Trans** (`--trans`): the CpG and the gene lie on different
     chromosomes.
   * **All** (`--all`): no region filtration.

P-value filtration runs after the regression and saves output time. Region
filtration runs before the regression and saves both output time and
computation time.

## Running `tecpg run mlr` directly

If you need to invoke `tecpg run mlr` directly — for example to prototype a
non-default backend or to integrate `tecpg` into another pipeline — the
equivalent of the `pipeline.sh` mapping stage is:

```bash
tecpg -i data -a annot -o output run mlr \
    --mlr-method qr --cis --compute-ig --compute-influence
```

`--mlr-method` selects the backend (`qr` for mapping, `qr_bootstrap` for the
bootstrap stage, `qr_permute` for permutation testing). On the `qr` backend,
`--qr-impl {torch,householder}` selects the QR factorization; `torch`
(`torch.linalg.qr`) is the default, and `householder` is a CUDA-only batched
path. Chunk sizes are auto-selected on server-class hosts. See
[Chunking](#chunking) and [Performance tuning](#performance-tuning) for the
available overrides, and run `tecpg run mlr --help` for the authoritative,
up-to-date option list — the README intentionally does not reproduce it.

## Selecting a GPU when multiple are available

On shared development systems and clusters the host may have several GPUs.
Find the index of the one you want with `nvidia-smi` (the leftmost column),
then restrict `tecpg` to it with `CUDA_VISIBLE_DEVICES`:

```bash
# Use only GPU 1
CUDA_VISIBLE_DEVICES=1 tecpg run mlr --cis
```

Inside the process the selected device is renumbered as device 0, so no
`tecpg` flag is needed. Listing more than one index
(`CUDA_VISIBLE_DEVICES=1,0`) exposes both, with the first listed becoming
device 0.

## Performance tuning

The CLI exposes several knobs to overlap GPU compute with host I/O and BLAS
work. Most auto-resolve from the active `--host-profile` (see
[CUDA and host profiles](#cuda-and-host-profiles)) and rarely need to be
touched, but the following overrides are available when a run is GPU-, save-,
or CPU-bound:

* `--prefetch-chunks` (`TECPG_PREFETCH`): number of chunks to prefetch onto
  the GPU to overlap with compute (auto-resolved to `0` when CUDA is
  unavailable or `--host-profile=minimum`).
* `--save-threads` (`TECPG_SAVE_THREADS`): number of threads in the
  asynchronous save pool. The Parquet path uses a `ThreadPoolExecutor`
  (PyArrow releases the GIL); the CSV path uses a `ProcessPoolExecutor`.
  Auto-capped at 8 on server-class hosts (RAID/dm-crypt LUNs saturate well
  below 32 writers).
* `--blas-threads` (`TECPG_BLAS_THREADS`): host BLAS/OpenMP thread count
  (default `0`). Applied as a pre-import shim in `tecpg/__main__.py` so it
  is honored before NumPy/PyTorch initialize their thread pools.
* `--output-format {auto,csv,parquet}`: `auto` resolves to `parquet` on all
  host profiles.
* `--gene-loci-per-chunk` / `--meth-loci-per-chunk`: see
  [Chunking](#chunking). Supplying exactly one pins that axis and lets the
  auto-sizer choose the other.

Rules of thumb: if VRAM is full but GPU SM% is low, try `--prefetch-chunks 2`;
if CPU is saturated by writers, lower `--save-threads`; if host BLAS is
fighting the GPU feeder, set `--blas-threads 2`.

For a deeper investigation, `profiling.sh` drives `nvidia-smi`, `top`,
`vmstat`, and `pidstat` alongside PyTorch debug output, sweeps
prefetching / chunk size / TF32 / BLAS thread configurations, and emits an
environment-annotated results tarball plus a `Verdict:` line that classifies
the bottleneck (GPU-, save-, or CPU-bound). See
[`docs/profiling.md`](docs/profiling.md).

The per-chunk startup banner reports the effective values
(`save_threads_effective`, `prefetch_chunks_effective`,
`blas_threads_effective`, logical and physical CPU counts), and per-chunk
metrics (`gpu_idle_between_chunks_ms`, `save_queue_depth`, `prefetch_fill`)
are emitted with an end-of-run statistical summary to help diagnose
bottlenecks.

## Documentation

* `tecpg --help`, `tecpg run mlr --help`, and the other `--help` pages are the
  authoritative reference for command-line options.
* [`docs/ecpg-filtering-prioritization.md`](docs/ecpg-filtering-prioritization.md)
  — end-to-end walkthrough of how eCpGs are filtered, prioritized, tested for
  enrichment, and visualized across the `pipeline.sh` / `pipelinePost.sh`
  workflow (regions, p-values, qr stats, precise p-values, FDR, bootstrap
  scores, network nodes/edges).
* [`docs/annotation.md`](docs/annotation.md) — annotation sources, the BED6
  contract, and the GENCODE-derived probe-gene model used for region
  assignment.
* [`docs/mlr_qr_permute.md`](docs/mlr_qr_permute.md) — design, status, and
  output columns of the `qr_permute` permutation backend.
* [`docs/integrated_gradients.md`](docs/integrated_gradients.md) — how the IG
  saliency columns are computed and what they mean.
* [`docs/bootstrap_qr_unification.md`](docs/bootstrap_qr_unification.md) — the
  shared QR path behind the `qr` and `qr_bootstrap` backends.
* [`docs/profiling.md`](docs/profiling.md) — the `profiling.sh` bottleneck
  harness.
* [`docs/tools.md`](docs/tools.md) — inventory of every helper script under
  `tools/`.

Within the code, function docstrings and extensive type hints document the
library API.

# Part B — Demonstration: the GTP/MESA golden path

The remainder of this README is a self-contained demonstration of `tecpg` on
two public datasets (GTP and MESA), driven by the `pipeline*.sh` scripts and
the helper scripts under `tools/`. None of this is required to use `tecpg`
itself (Part A) — it is one complete, reproducible worked example, and it
requires the [R dependencies](#r-dependencies-pipeline-and-tools-only).

## Demo datasets (GTP and MESA)

Two real-world public datasets are bundled as turn-key demonstrations, in
addition to the synthetic `dummy` dataset used for smoke tests. Both are
downloaded directly from GEO by the `tecpg data` sub-commands and are the same
datasets used by Kennedy et al. *BMC Genomics* (2018) **19:476**
(`10.1186/s12864-018-4842-3`), whose published eCpG–transcript pairs are
automatically downloaded alongside the raw matrices for use as a benchmark
reference list.

* **GTP — Grady Trauma Project** (`tecpg data gtp`,
  `./pipeline.sh --dataset gtp`). A study of *n ≈ 340* primarily
  African-American adults recruited from urban primary-care clinics in
  Atlanta, GA, designed to characterize the genetic and epigenetic correlates
  of trauma exposure and PTSD. Whole-blood DNA methylation was assayed on the
  Illumina HumanMethylation450 BeadChip (GEO accession **GSE72680**, ~349k CpG
  loci) and gene expression on the Illumina HumanHT-12 v4 BeadChip (GEO
  accession **GSE58137**, ~39k expression probes). The matched Kennedy 2018
  eCpG list is pulled from `MOESM1_ESM.txt` of the supplementary materials.

* **MESA — Multi-Ethnic Study of Atherosclerosis** (`tecpg data mesa`,
  `./pipeline.sh --dataset mesa`). A multi-site, multi-ethnic longitudinal
  cohort focused on the subclinical-to-clinical progression of cardiovascular
  disease. CD14+ monocyte DNA methylation was assayed on the Illumina
  HumanMethylation450 BeadChip (GEO accession **GSE56046**) and matching gene
  expression on the Illumina HumanHT-12 v4 BeadChip (GEO accession
  **GSE56045**), giving a several-hundred-sample paired
  methylation/expression cohort. The matched Kennedy 2018 eCpG list is pulled
  from `MOESM2_ESM.txt`.

Both datasets share the same array combination (HumanMethylation450 +
HumanHT-12 v4), so the comprehensive BED6 annotation files shipped under
`demo/` apply unchanged to either, and `pipelinePre.sh` plus `pipeline.sh`
wire up identical processing for `--dataset gtp` and `--dataset mesa`
(data prep → probe blacklist → methylation-derived ancestry instruments →
EpiDISH cell proportions → categorical encoding → residualized PCA → MLR + IG
+ influence → merge → region annotation → precise p-values → BH-FDR /
diagnostics → influence flag → bootstrap candidate list → bootstrap
evaluation).

Two lighter options exist for checking wiring before committing to a full run:

* **`dummy`** — a small synthetic dataset generated locally. No download, no
  biological meaning. Skips the ancestry and EpiDISH stages.
* **`gtpsub`** — a locus-subsampled GTP build (10,000 CpGs and 5,000
  expression probes by default, seed 42) for fast wiring checks on *real*
  data. Requires the GTP download; like `dummy` it skips the ancestry and
  EpiDISH stages.

## Quick start (the golden path)

Prerequisites: `tecpg` installed ([Installation](#installation)) and the R
packages installed with `Rscript tools/install_dependencies.R`
([R dependencies](#r-dependencies-pipeline-and-tools-only)) — the pipeline
scripts call R at several stages. Run every command from the repository root.

The pipeline is always two steps: `pipelinePre.sh` prepares a dataset, then
`pipeline.sh` runs the mapping and downstream stages. Pick a dataset by how
much time you have:

| Dataset  | Download                     | Typical footprint                                   | Purpose                                   |
|----------|------------------------------|-----------------------------------------------------|-------------------------------------------|
| `dummy`  | none                         | minutes on a laptop CPU                             | smoke-test the wiring                     |
| `gtpsub` | GTP from GEO (multi-GB)      | minutes to tens of minutes once data is downloaded  | wiring check on real data                 |
| `gtp`    | GTP from GEO (multi-GB)      | hours; GPU and server-class RAM strongly recommended | full reproduction, cis or genome-wide     |
| `mesa`   | MESA from GEO (multi-GB)     | hours; GPU and server-class RAM strongly recommended | full reproduction, cis or genome-wide     |

```bash
# 1. Smoke-test the full pipeline on synthetic data (start here)
./pipelinePre.sh --dataset dummy
./pipeline.sh    --dataset dummy --mapping all

# 2. Wiring check on real (subsampled) GTP data
./pipelinePre.sh --dataset gtpsub
./pipeline.sh    --dataset gtpsub --mapping cis

# 3. Full CIS-only run on the GTP demo dataset
./pipelinePre.sh --dataset gtp
./pipeline.sh    --dataset gtp --mapping cis

# 4. Full genome-wide run on the MESA demo dataset
./pipelinePre.sh --dataset mesa
./pipeline.sh    --dataset mesa --mapping all
```

`pipelinePre.sh` downloads the dataset (GTP/MESA only), populates
`data_<dataset>/`, copies the comprehensive BED6 annotations from `demo/` into
`annot_<dataset>/`, and produces the `M.csv`, `G.csv`, and `C.csv` matrices.
`pipeline.sh` then walks through the mapping and downstream stages, writing
artifacts and diagnostic plots into `output_<dataset>/`. Both scripts skip
stages whose outputs already exist and accept `--start-stage <n>` to resume
(see the stage lists below), so an interrupted run can be picked up where it
stopped.

After `pipeline.sh` finishes, run `./pipelinePost.sh <dataset>` to produce
Circos / volcano / Manhattan / bipartite-network visualizations from
`output_<dataset>/bootstrap_merged.parquet`, and optionally
`./pipelinePermute.sh` to attach permutation p-values.

## Pipeline stages

The demonstration is driven by four orchestration scripts, run in order:
`pipelinePre.sh` → `pipeline.sh` → `pipelinePost.sh` → `pipelinePermute.sh`.
Each reuses the per-dataset working directories (`data_<dataset>/`,
`annot_<dataset>/`, `output_<dataset>/`) and can resume from any stage via
`--start-stage`.

### Preprocessing (`pipelinePre.sh`)

`pipelinePre.sh` prepares a dataset for analysis. It downloads or generates the
raw data, estimates immune cell proportions, and builds the residualized PCA
covariates, producing the `M.csv`, `G.csv`, and `C.csv` matrices (plus the BED6
annotations) that `pipeline.sh` consumes. Run it once per dataset before
`pipeline.sh`.

```bash
./pipelinePre.sh --help
./pipelinePre.sh --dataset dummy
./pipelinePre.sh --dataset gtp
./pipelinePre.sh --dataset gtpsub
./pipelinePre.sh --dataset mesa
```

Options:

* `-d, --dataset {dummy,gtp,gtpsub,mesa}` — which dataset to use. `dummy`
  generates a small synthetic dataset for testing; `gtp` downloads and prepares
  the Grady Trauma Project data via `tecpg data gtp`; `gtpsub` prepares a
  locus-subsampled GTP build; `mesa` does the same for MESA via
  `tecpg data mesa`.
* `-s, --start-stage STAGE` — resume from one of `all` (default), `prep`,
  `ancestry`, `cell_prop`, `pca`. Each stage is skipped automatically when its
  on-disk artifacts already exist, so any stage can run on its own.

The script creates per-dataset working directories `data_<dataset>/` and
`annot_<dataset>/`, and runs the following stages. Each stage name (in `code`)
matches the value accepted by `--start-stage`.

1. **`prep` — Data preparation** *(stage `[1/9]`)*. Downloads or
   generates the dataset (`tecpg data {dummy,gtp,mesa}`), copies the
   default comprehensive BED6 annotations into `annot_<dataset>/`
   (with a graceful fallback to the original `annoEPIC.hg19.bed6` /
   `annoHT12.hg19.bed6`), applies the probe blacklist
   (`tools/generateProbeBlacklist.sh` +
   `tools/exclude_blacklisted_probes.py`, scoped to the `METH_ARRAY`
   setting near the top of the script — `450k` by default) to produce
   `M.csv` from `M_orig.csv`, and runs `tools/exploreOmics.py` to write
   QC plots and an HTML report under `data_<dataset>/qc/`.
2. **`ancestry` — Methylation-derived ancestry instruments** *(stages
   `[1.4/9]` and `[1.45/9]`)*. `tools/ancestry_probes_report.py`
   evaluates ancestry instruments from the pre-blacklist methylation
   matrix and writes `ancestry_probes.json` /
   `ancestry_probes_report.html` plus an `ancestry_scores.csv` sidecar.
   Where configured (MESA), `tools/mergeCovariateColumns.py` then admits
   selected components (`rs_PC1`/`rs_PC2` as `Anc_PC1`/`Anc_PC2`) into
   the covariates. Skipped for `dummy` and `gtpsub`.
3. **`cell_prop` — Immune cell-proportion estimation and categorical
   encoding** *(stages `[1.5/9]` and `[1.6/9]`)*.
   `tools/estimateCellProportions.sh` runs EpiDISH (M-value aware) on
   real datasets to produce `C_post_cellTypes.csv`; skipped for `dummy`
   and `gtpsub` (random / thin data cause singular fits), where the
   covariates are copied through instead. `tools/encodeCategorical.py`
   then expands integer-coded categorical covariates (MESA:
   `racegendersite`) into indicator columns before residualization.
4. **`pca` — Residualization & PCA** *(stage `[2/9]`)*.
   `tools/residualize_pca.sh` generates expression and methylation
   principal components, which are merged with the cell-proportion
   covariates to produce the final `C.csv`.

When `pipelinePre.sh` finishes, `data_<dataset>/` contains `M.csv`, `G.csv`,
and `C.csv`, and the dataset is ready for `pipeline.sh`.

The annotation files used in the `prep` stage default to the comprehensive
BED6 annotations under `demo/` (`annoEPIC_comprehensive.hg19.bed6` and
`annoHT12_comprehensive.hg19.bed6`, built with a validated multi-source HT-12
mapping pipeline in which probes without positional evidence are emitted as
unmapped rather than given fabricated positions), with a graceful fallback to
the original `annoEPIC.hg19.bed6` / `annoHT12.hg19.bed6` files.

### Full analysis (`pipeline.sh`)

`pipeline.sh` runs the eQTM mapping and downstream analysis end to end. It
picks up the `M.csv`, `G.csv`, and `C.csv` matrices (and the BED6 annotations)
produced by `pipelinePre.sh`, so run `./pipelinePre.sh --dataset <dataset>`
first; the script exits with an error if those inputs are missing or empty. It
wraps `tecpg` and the helper scripts in `tools/` into a nine-stage workflow,
with structured logging, dataset-aware defaults, and the ability to resume
from any stage.

```bash
./pipeline.sh --help
./pipeline.sh --dataset dummy --mapping all
./pipeline.sh --dataset gtp   --mapping cis
./pipeline.sh --dataset mesa  --mapping all
./pipeline.sh --dataset gtp   --mapping all --start-stage merge
```

Options:

* `-d, --dataset {dummy,gtp,gtpsub,mesa}` — which dataset to use. Must match the
  dataset already prepared by `pipelinePre.sh`.
* `-m, --mapping {all,cis}` — region filter passed through to
  `tecpg run mlr` (`--all` or `--cis`).
* `-s, --start-stage STAGE` — resume from one of `all` (default), `map`,
  `merge`, `annotate`, `precise_p`, `summarize`, `influence_flag`, `boot_list`,
  `bootstrap`. Context variables (`DF`, `TOTAL_TESTS`) are recomputed from the
  on-disk artifacts so any stage can run on its own.

The script reuses the per-dataset working directories `data_<dataset>/`,
`annot_<dataset>/`, and `output_<dataset>/`, and runs the following
stages. Each stage name (in `code`) matches the value accepted by
`--start-stage`, so any individual step can be re-run in isolation.

1. **`map` — eQTM mapping** *(stage `[3/9]`)*. Runs `tecpg ... run
   mlr --mlr-method qr --<mapping> -p "$MAP_P_THRESH" --compute-ig
   --compute-influence`, with chunk sizes auto-selected by the CLI
   (overridable by exporting `TECPG_M_CHUNK` / `TECPG_G_CHUNK`).
   `MAP_P_THRESH` (default `0.001`, matching the CLI's own `-p` default) is
   the catalog's inclusion gate: pairs above it are never written. It is set
   explicitly in `pipeline.sh` so it appears in the run log. Logs are tee'd
   to `mlr_run_<dataset>.log` and `TOTAL_TESTS` is extracted from that log
   for downstream FDR.
2. **`merge` — Merge chunked output** *(stage `[4/9]`)*.
   `tools/mergeOutputs.py` combines per-chunk files into a single
   `output_<dataset>/merged.parquet`; intermediate chunk files are
   deleted.
3. **`annotate` — Region annotation** *(stage `[5/9]`)*. First derives a
   probe-gene map (`annot_<dataset>/probe_gene_model.tsv`) with
   `tools/build_probe_gene_model.py` from the GENCODE GTF at
   `$TECPG_GENCODE_GTF` (default
   `encode_beds/gencode.v49lift37.annotation.gtf.gz`, downloaded from
   GENCODE if absent — override the source with `$TECPG_GENCODE_GTF_URL`;
   `dummy` gets a synthetic fixed-span map instead, whose labels carry no
   biological meaning), reusing an existing map only when both the GTF and
   the staged `G.bed6` still match its header. Then
   `tools/assignRegionToEcpg_parquet.py` annotates each pair with one of
   seven strand-aware regions — `PROMOTER`, `GENEBODY`, `CIS5`, `CIS3`,
   `DISTAL5`, `DISTAL3`, `TRANS` (5′/3′ relative to the gene's strand) —
   using the gene spans from that map and writes `annotated.parquet`;
   pairs whose probe or gene lacks an annotation are summarized downstream
   as `UNKNOWN`. Missing-annotation probe IDs are collected into a sidecar
   `annotation_missing_ids.txt`.
4. **`precise_p` — High-precision p-values** *(stage `[6/9]`)*.
   `tools/recalculate_pvalues_parquet.py` replaces the normal-CDF
   approximation with Student's-t p-values using the degrees of
   freedom derived from `C.csv`, writing `annotated_pcalc.parquet`.
5. **`summarize` — FDR and summary** *(stage `[7/9]`)*.
   `tools/summarizeOutput_parquet.py` computes a global
   Benjamini–Hochberg FDR (using `TOTAL_TESTS` dynamically extracted
   from the `mlr` log), writes `summarized.parquet`, and emits QQ,
   histogram, and saliency diagnostic plots into `output_<dataset>/`.
6. **`influence_flag` — Single-point influence screen** *(stage
   `[7b/9]`)*. `tools/flagInfluence_parquet.py` derives
   `mt_influence_flag` from the mapper's `mt_h_max` leverage column under
   the configured rule (`INFLUENCE_RULE`, default `floor`, threshold
   `INFLUENCE_DELTA`, default `0.1`), writing a new
   `summarized.influence.parquet` plus a QC report under
   `output_<dataset>/influence_qc/`. Set `INFLUENCE_RULE=off` to skip the
   stage; downstream stages then consume `summarized.parquet`.
7. **`boot_list` — Bootstrap candidate list** *(stage `[8/9]`)*.
   `tools/createBootstrapList.py` selects the top hits (ranked by
   p-value, with per-region floors and caps) into `bootstrap_list.csv`.
8. **`bootstrap` — Bootstrap evaluation** *(stage `[9/9]`)*. Runs
   `tecpg ... run mlr --mlr-method qr_bootstrap --pairs-file ...
   --master-parquet ... --bootstrap-iterations 1000
   --bootstrap-batch-size 10 --compute-ig` to attach empirical
   bootstrap p-values to the top candidates and write
   `bootstrap_merged.parquet`.

#### Integrated Gradients (IG) covariates

The pipeline computes per-feature saliency (Integrated Gradients) to measure
the relative contribution of methylation vs. covariates. Because computing
this for every genome-wide eQTM pair inflates the intermediate output files,
the feature is scoped by stage using two variables near the top of
`pipeline.sh`:

* `MLR_IG_COVARIATES`: controls Stage 3 (genome-wide mapping). `"all"` (the
  current default) emits per-covariate IG columns; `"none"` emits only the
  scalar `mt_ig`; a comma-separated list restricts IG to those covariates.
* `BOOTSTRAP_IG_COVARIATES`: controls Stage 9 (bootstrap), default `"all"`.
  Because the bootstrap runs on a small, prioritized candidate list, full
  per-feature IG costs very little space while enabling fraction-based
  saliency analysis downstream.

### Post-processing (`pipelinePost.sh`)

`pipelinePost.sh` consumes `output_<dataset>/bootstrap_merged.parquet`
produced by `pipeline.sh` and runs the visualization and network-analysis
tools:

```bash
./pipelinePost.sh gtp
./pipelinePost.sh mesa
```

The script downloads the UCSC hg19 `cytoBand.txt` if missing and then runs
eleven stages, in order:

1. **Influence calibration bridge** — `tools/calibration_bridge.py` cross-checks
   the `mt_h_max` leverage screen against bootstrap fragility on the
   *unfiltered* catalogs, and `tools/fig_influence_dose_response.py` renders the
   dose-response and SE-ratio figures. Disable with `INFLUENCE_BRIDGE=off`.
2. **Influence filter** — drops rows whose CpG carries `mt_influence_flag`
   from both catalogs into `output_<dataset>/retained/`, so every downstream
   panel agrees on one retained universe (`INFLUENCE_MODE=exclude` by default;
   `ignore` restores the pre-influence behavior).
3. **Influence QC report** — `tools/influence_qc_report.py` renders a
   consolidated HTML report from the influence artifacts
   (`INFLUENCE_REPORT=off` to skip).
4. `cytoBand.txt` check / download.
5. `tools/plotCircos.py` — Circos plots of the eQTM architecture
   (`output_<dataset>/plots/`).
6. `tools/visualizeFindings.py` — volcano, Manhattan, scatter, and related
   plots. Generates a full set of figures for every available p-value
   column (bootstrap `p_boot`, `precise_mt_p`, `mt_p`) with prefixed
   filenames.
7. `tools/evaluateSaliency.py` — integrated-gradients saliency diagnostics
   for the bootstrap candidates, optionally re-run with `--frac-exclude`
   (`SALIENCY_FRAC_EXCLUDE`) so the saliency denominator excludes
   expression-derived IG.
8. `tools/annotate_bootstrap_concordance.py` — bootstrap / analytic
   concordance scores and a distribution summary.
9. `tools/runEnrichment.py` — functional (Enrichr/`gseapy`) and optional
   ENCODE ChromHMM enrichment of significant genes, written to
   `output_<dataset>/enrichment/`. Draws significant genes from the FDR
   summary (`summarized.parquet`) and the bootstrap IG ranking
   (`bootstrap_merged.parquet`). `tools/summarizeEnrichment.py` then
   renders a self-contained HTML summary.
10. `tools/exportBipartiteNetwork.py` — Cytoscape-formatted node and edge
    tables under `output_<dataset>/network/`. The universe is the
    FDR-significant catalog (`--max-fdr 0.05`), with `--top-k 100000` as a
    non-binding safety cap.
11. `tools/visualizeBipartiteNetwork.py` — energy-minimized bipartite
    network, UMAP of regulatory β-diversity, regulatory degree distribution,
    clustered bipartite adjacency heatmap, per-region stratified figures, and
    arc diagrams.

### Permutation testing (`pipelinePermute.sh`)

`pipelinePermute.sh` scores an existing mapping catalog against a
design-fixed Freedman–Lane permutation null (residualize on the covariates,
permute the residuals, refit) using the `qr_permute` backend, and builds
diagnostic and QC reports.
`qr_permute` is a **post-mapping consumer**: it reads the observed `mt_t` and
the `(mt_id, gt_id)` universe from a master parquet produced by an earlier
mapping run (for example `output_<dataset>/merged.parquet` from `pipeline.sh`),
so produce that master first. It requires `pipelinePre.sh` to have been run to
prepare the dataset.

```bash
./pipelinePermute.sh --help
./pipelinePermute.sh --dataset dummy --master-parquet output_dummy/merged.parquet
./pipelinePermute.sh --dataset gtp --master-parquet output_gtp/merged.parquet --permutations 100
./pipelinePermute.sh --dataset mesa --start-stage eval
```

Options (see `--help` for the full list):

* `-d, --dataset {dummy,gtp,gtpsub,mesa}` — which dataset to use. Must match
  the dataset already prepared by `pipelinePre.sh`.
* `--master-parquet PATH` — existing mapping output to score. Also accepts a
  `sample_reservoir.csv` directly.
* `--reservoir` — score the reservoir universe from a prior
  `--reservoir-count` map.
* `--cis-enrich` (default) — build a unified gene-anchored master: run a cis
  write-all map (`--cis-window`, default 1 Mb) and assemble its near-gene
  pairs with the reservoir's trans/distal pairs via
  `tools/build_gene_anchored_master.py`, so the per-region evaluation has the
  near-gene coverage a flat reservoir lacks.
* `-m, --mapping {all}` — the only supported method is `all`. `cis` is
  accepted by the parser but rejected at runtime because `qr_permute`'s null
  is trans-global.
* `-s, --start-stage STAGE` — resume from one of `all` (default), `permute`,
  `eval`.
* `--permutations`, `--subsample-mt-count` (default 2000),
  `--subsample-g-count` (default 2000), `--seed` — pass-through arguments to
  `tecpg run mlr`.
* `--total-tests N` — BH denominator for `fdr_permute`; required when the
  mainline annotation stage runs, and must be the mapping-grid `TOTAL_TESTS`
  used for `fdr_est`.
* `--no-assign-regions`, `--no-qc-report`, `--no-annotate-mainline` — skip the
  region-annotation, QC-report, and mainline-annotation work respectively.

> **NOTE:** `--subsample-mt-count` / `--subsample-g-count` subsample the NULL
> population only. The reported set is always the full M × G cross product;
> these flags do NOT reduce output size. To get a tractable reported set,
> physically subset `data_<ds>/M.csv` and `data_<ds>/G.csv` into a smaller
> `data_<ds>` first. Subsample LOCI, never SAMPLES — dropping samples changes
> DF.

> **NOTE:** The `dummy` dataset is a WIRING SMOKE TEST ONLY. Disbelieve its
> numbers. Dummy annotations are chrom=randrange(1,23) over random data, so
> cis and trans are exchangeable BY CONSTRUCTION and the stratify arm will
> return `single_global_null_adequate` trivially. It says nothing about real
> data.

The script runs in five stages, reusing the per-dataset working directories
`data_<dataset>/`, `annot_<dataset>/`, and `output_<dataset>/`:

1. **Region annotation** *(stage `[1/5]`)*. Assigns the canonical `region`
   column to the master with `tools/assignRegionToEcpg_parquet.py` (skipped
   with `--no-assign-regions`, in which case the evaluation falls back to
   2-way cis/trans strata).
2. **`permute` — Run permutations** *(stage `[2/5]`)*. Runs `tecpg ... run mlr
   --mlr-method qr_permute --all --output-format parquet` against the master,
   persisting the null accumulator as an `.npz` sidecar.
3. **`eval` — Evaluate output** *(stage `[3/5]`)*. Runs `tools/eval_permute.py`
   to audit the generated parquet and produce the diagnostic report
   `eval_permute_report.json`.
4. **Summary and QC report** *(stage `[4/5]`)*. `tools/summarize_permute.py`
   renders the 7-way region table and `tools/permute_qc_report.py` writes a
   self-contained HTML QC report.
5. **Mainline annotation** *(stage `[5/5]`)*. `tools/annotate_permute_p.py`
   writes `p_permute` / `fdr_permute` back onto the mainline catalogs using the
   calibration verdict and the supplied `--total-tests` BH denominator.

## Alternative annotation and assignment of regions

There are times when we may want to define our own classifications for a
region (e.g., CIS) and apply different annotations to our mapping data. The
standard, supported path is the Parquet-based classifier driven by
`pipeline.sh` (stage `annotate`, `[5/9]`).

To run it standalone against a merged Parquet produced by an out-of-band
`tecpg run mlr --all ...` invocation, first derive the probe-gene map from a
GENCODE GTF and then classify:

```bash
python3 tools/build_probe_gene_model.py \
    --gtf encode_beds/gencode.v49lift37.annotation.gtf.gz \
    --probe-bed annot/G.bed6 \
    --output annot/probe_gene_model.tsv

python3 tools/assignRegionToEcpg_parquet.py \
    -d output/merged.parquet \
    -g annot/G.bed6 \
    --gene-model annot/probe_gene_model.tsv \
    -m annot/M.bed6 \
    -o output/annotated.parquet
```

The probe BED supplies the `gt_*` probe coordinates; the probe-gene map
supplies the gene span (from the gene model, not the probe footprint) that the
region windows are measured against. See
[`docs/annotation.md`](docs/annotation.md) for details.

Pre-built comprehensive BED6 annotation files for the Illumina EPIC and
HT-12 v4 arrays are shipped under `demo/`:

* `demo/annoEPIC_comprehensive.hg19.bed6` and
  `demo/annoEPIC_comprehensive.hg38.bed6`
* `demo/annoHT12_comprehensive.hg19.bed6` and
  `demo/annoHT12_comprehensive.hg38.bed6`

These were generated with `tools/generate_annotations.py`, which uses a
validated multi-source HT-12 mapping pipeline (Re-Annotator → GEO → UCSC WG-6,
with NA fallback and provenance tracking) and correctly handles unmapped
probes, alternate/unplaced contigs, and pseudoautosomal labels; probes without
positional evidence stay unmapped instead of receiving fabricated positions.
The region defaults follow Kennedy et al. *BMC Genomics* (2018) **19:476**,
split by strand: `CIS5` / `CIS3` within 50 kb of the gene on its 5′ / 3′ side,
`DISTAL5` / `DISTAL3` beyond 50 kb on the same chromosome, `PROMOTER`
± 2.5 kb of the TSS, `GENEBODY` within the gene span, and `TRANS` on a
different chromosome. Override these in the script's defaults block if you
need different cutoffs. The script annotates every row it is given; it applies
no p-value filter of its own.

> **Legacy CSV path:** the original per-chunk CSV classifier
> `tools/assignRegionToEcpg.py` is retained for backwards compatibility with
> pre-Parquet outputs but is no longer the recommended entry point. New work
> should use the Parquet variant above, which is what `pipeline.sh` runs.

## Tools and helper scripts

The `tools/` directory contains the supporting scripts driven by
`pipelinePre.sh`, `pipeline.sh`, `pipelinePost.sh`, and `pipelinePermute.sh`.
They can also be invoked standalone; each accepts `--help`. The R-based tools
additionally require `Rscript tools/install_dependencies.R` — see
[R dependencies](#r-dependencies-pipeline-and-tools-only).

The full inventory, grouped by purpose (data preparation and QC, annotation,
mapping post-processing, influence and permutation diagnostics,
bootstrapping, visualization and network analysis, benchmarking and
profiling), is in [`docs/tools.md`](docs/tools.md).

## Tests

The test suite lives under `tests/` and is run with `pytest` (configuration in
`pytest.ini`, which excludes the manual smoke/mock and network-dependent
scripts). A minimal CI gate runs `pytest` on pull requests. Install the
development requirements first:

```bash
pip install --editable .
pip install -r requirements-dev.txt
pytest
```

See [`tests/README.md`](tests/README.md) for the per-test inventory and for
guidance on the longer permutation / bootstrap tests.

## Acknowledgements

This work was partially supported by an NIH NCI MERIT award (R37, CA233774,
PI: Kober) and Cancer Center Support Grant (P30, CA082103, Co-I: Olshen).
