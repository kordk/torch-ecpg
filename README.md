# tecpg

Torch-eCpG is a GPU enabled expression quantitative trait methylation (eQTM) mapper to identify expression associated CpG (eCpG) loci with python CLI using pytorch.

If you use Torch-eCpG in your research, please cite the following paper: Kober, K.M., Berger, L., Roy, R. et al. Torch-eCpG: a fast and scalable eQTM mapper for thousands of molecular phenotypes with graphical processing units. BMC Bioinformatics 25, 71 (2024). https://doi.org/10.1186/s12859-024-05670-4

Torch-eCpG v2 was presented as a poster: Kober, K.M., Rau, A., Olshen, A. Torch-eCpG v2: A Scalable and Interpretable Framework for eQTM Mapping and Multi-Omic Network Analysis. Poster presented at the 21st International Conference on Computational Intelligence Methods for Bioinformatics and Biostatistics (CIBB 2026), Sapienza Università di Roma, Rome, 2–4 September 2026.

The current version on the `dev` branch is **2.0.0b2**, released as v2.0.0
Beta 2 (tag `v2.0.0-beta.2`) for the 21st International Conference on Computational Intelligence Methods for Bioinformatics and Biostatistics (CIBB 2026), Sapienza Università di Roma, Rome, 2–4 September 2026. As of
`2.0.0b2.dev0` the project version scheme migrated to the
[PEP 440](https://peps.python.org/pep-0440/) standard, replacing the older
`X.Y.Z-dev` suffix with pre-release / development tags such as `2.0.0b1` (beta)
and `2.0.0b2.devN`. Since the `1.0.0` release on `main`, the project has grown a
preprocessing pipeline (`pipelinePre.sh`), an end-to-end analysis pipeline
(`pipeline.sh`), a downstream visualization/network pipeline
(`pipelinePost.sh`), a permutation-testing pipeline (`pipelinePermute.sh`)
driven by the `qr_permute` MLR backend, Parquet output by default, anchored auto
chunk-sizing, Integrated Gradients (IG), a single-point influence screen
(`--compute-influence`), an empirical bootstrap MLR backend, support for the
MESA dataset, comprehensive HT-12/EPIC BED6 annotations, a GENCODE-derived
probe-gene model for region assignment, chromatin-feature and functional
enrichment tooling, a host-profile aware CLI, and a CUDA Docker image
(`docker-related/`). See `CHANGELOG.md` for the full per-version history.

## How this README is organized

This README is split into two parts:

* **Part A — Core `tecpg` tool** documents the installable `tecpg` CLI/library itself: installation, CUDA/host profiles, input/output formats, chunking, region/p-value filtration, running `tecpg run mlr` directly, GPU selection, and performance tuning. This is all you need to use `tecpg` as a general-purpose eQTM mapper on your own data.
* **Part B — Demonstration: the GTP/MESA golden path** is a self-contained, reproducible demonstration built on public GTP and MESA data and the `pipeline*.sh` orchestration scripts. These scripts and datasets are *not* required to use `tecpg`; they exist to show one complete worked example end to end.

### Table of contents

**Part A — Core `tecpg` tool**

* [Installation](#installation)
* [CUDA](#cuda)
* [Input data](#input-data)
* [Output](#output)
* [Chunking](#chunking)
* [Filtration](#filtration)
* [Documentation](#documentation)
* [Running `tecpg run mlr` directly](#running-tecpg-run-mlr-directly)
* [Selecting a GPU when multiple are available](#selecting-a-gpu-when-multiple-are-available)
* [Performance tuning](#performance-tuning)

**Part B — Demonstration: the GTP/MESA golden path**

* [Demo datasets (GTP and MESA)](#demo-datasets-gtp-and-mesa)
* [Quick start (the golden path)](#quick-start-the-golden-path)
* [Pipeline stages](#pipeline-stages)
* [Alternative annotation and assignment of regions](#alternative-annotation-and-assignment-of-regions)
* [Tools and helper scripts](#tools-and-helper-scripts)
* [Tests](#tests)

# Part A — Core `tecpg` tool

The sections below document the `tecpg` command-line tool and library on their own, independent of any particular dataset.

## Installation

Pip install from github using `git+https://`.

```bash
pip install git+https://github.com/kordk/torch-ecpg.git
```
Pip install from github using `git+https://` for the dev branch.
```bash
pip install git+https://github.com/kordk/torch-ecpg.git@dev
```

If you want to be able to edit the code for debugging and development, install in editable mode and do not remove the directory.

```bash
cd [path/to/code/directory]
git clone https://github.com/kordk/torch-ecpg.git
cd torch-ecpg
pip install --editable .
```

`tecpg` is an entry point in the command line than calls the root CLI function. If the installation was successful, running `tecpg --help` should provide help with the command line interface.

If you have issues with using `pip` in the command line, try `python -m pip` or `python3 -m pip`.

### Docker

A containerized build of the full pipeline is defined in `docker-related/`
(`Dockerfile` on the `nvidia/cuda:12.4.1` runtime base). Build it from the
repository root so the `.dockerignore` exclusions apply:

```bash
docker build -t tecpg-pipeline -f docker-related/Dockerfile .
```

See [`docker-related/README.md`](docker-related/README.md) for running the
image and saving/loading it. A pre-built image of the published v1 is on
Docker Hub at https://hub.docker.com/r/kordk/torch-ecpg.

## CUDA

`tecpg` can calculate on the CPU or on a CUDA enabled GPU device. CUDA devices are generally faster than CPU computations for sufficiently large inputs.

The program will automatically determine whether there is a CUDA enabled device and use it if available. To force calculation on the CPU, set the `--threads` option to a nonzero integer. This will also set the number of CPU threads used.

The top-level CLI also accepts `--host-profile {auto,minimum,server}` (envvar
`TECPG_HOST_PROFILE`). `auto` (default) inspects the host (physical CPU count
and total RAM) and picks `minimum` for laptop-class hosts (`<12 cores` or
`<32 GB`) and `server` otherwise. The resolved profile drives defaults for the
save pool, prefetch depth, and chunk auto-sizing. (Output format is no longer
profile-dependent: since `2.0.0b2.dev32`, `--output-format auto` resolves to
`parquet` on every host profile.) Explicit per-flag overrides
(`--save-threads`, `--output-format`, `--prefetch-chunks`,
`--gene-loci-per-chunk`, `--meth-loci-per-chunk`, `--blas-threads`) always win.

## Input data

Methylation values, gene expression values, and covariates are provided in CSV or TSV files in the `<working>/data` directory. For methylation and gene expression, columns are for individual samples and each row is for a loci. For the covariates, the columns are the type of covariate and the rows are the sample. Annotation files are used for region filtration and are stored in the `<working>/annot`. They use the `BED6` standard and store the positions of the methylation or gene expression loci.

> **Note:** The concrete `M.csv` / `G.csv` / `C.csv` / BED6 snippets below are taken from the GTP demo dataset purely as examples of the expected formats. The GTP and MESA demo datasets themselves, and the `pipeline*.sh` scripts that produce these files, are documented in *Part B — Demonstration* below.

Methylation CSV datafiles from the GTP dataset (see Demonstration below):
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

Gene expression CSV datafiles from the GTP dataset (see Demonstration below):
```bash
head data/G.csv | cut -d, -f1-5
```
```
,5881,5896,5915,5949
ILMN_1762337,43.10106,48.30485,37.49239,43.99564
ILMN_2055271,61.09617,61.84258,47.78094,49.32763
ILMN_1736007,51.30634,45.80393,45.43285,40.39254
ILMN_2383229,48.15523,42.69902,35.71749,39.52501
```
```bash
head -5 data/C.csv
```

Covariate CSV datafiles from the GTP dataset (see Demonstration below):
```
,Sex,age
5881,1,44
5896,1,50
5915,0,52
5949,1,56
```

Annotation BED6 files for the gene expression and methylation data (i.e., Illumina HumanHT-12 and Illumina MethylationEPIC arrays):
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
Example data for evaluation can be created or downloaded with tecpg:
```bash
tecpg data --help
```
```
Usage: tecpg data [OPTIONS] COMMAND [ARGS]...

  Base group for data management.

Options:
  --help  Show this message and exit.

Commands:
  dummy  Generates dummy data.
  gtp    Downloads and extracts GTP data.
  mesa   Downloads and extracts MESA data.
```

See the *Demo datasets (GTP and MESA)* section below for background on
the two real-world demo datasets and where they come from.

## Output

By default, the output format is Parquet. Since `2.0.0b2.dev32`
`--output-format auto` resolves to `parquet` on every host profile; use
`--output-format {auto,csv,parquet}` to override.

For `tecpg run mlr` without chunking, a single output file (`out.csv` or
`out.parquet`) is created in the output directory. With chunking on either
axis, per-chunk files named `{methylation chunk number}-{gene expression
chunk number}.{csv,parquet}` are written instead, and a sidecar
`sample_reservoir.csv` of unfiltered draws is produced for diagnostics.
`tools/mergeOutputs.py` combines the chunks into a single Parquet (or CSV)
file (and explicitly skips `sample_reservoir.csv`, fixed in `1.25.3-dev`).

Row labels indicate the gene expression id and the methylation id. Column
labels follow the convention: methylation-related columns are prefixed
`mt_`, gene-expression-related columns are prefixed `gt_`. For each
regression the columns are the estimate `est`, the standard error `err`,
the Student's T statistic `t`, and the p-value `p` (e.g. `mt_est`, `mt_err`,
`mt_t`, `mt_p`). When `--compute-ig` is enabled, integrated-gradients
saliency values are written alongside the regression results, and
`--compute-influence` adds the per-CpG maximum sample leverage `mt_h_max`.
After the post-mapping stages of `pipeline.sh`, additional columns include
the high-precision p-value (`precise_mt_p`), the assigned region
(`region`/`Region`), the global BH-FDR q-value (`fdr_est`), the influence
flag (`mt_influence_flag`), and (after the bootstrap stage) the empirical
bootstrap p-value `p_boot` with its seed recorded as `boot_seed`.
`pipelinePermute.sh` can annotate the same catalogs with the permutation
p-value `p_permute` and its BH q-value `fdr_permute`, together with the
`perm_seed` and `perm_n_perm` provenance columns, so any resampled result
can be reproduced from the catalog alone.


## Chunking

If the input is too large, the computational device may run out of memory. Chunking can help prevent this by partitioning the data into chunks that are computed and saved separately. Chunking sacrifices parallelization, and thus speed, for lower memory. Avoid chunking wherever possible for speed.

For `tecpg run mlr`, there are two types of chunking: methylation chunking and gene expression chunking. Gene expression chunking is preferable to methylation chunking if possible, as it sacrifices parallelization less.

As of `1.21.0-dev`, the CLI's `_auto_chunk_sizes` helper picks
`--gene-loci-per-chunk` and `--meth-loci-per-chunk` automatically from the
live RAM/GPU budget (80% target) on server-class hosts when the user
supplies neither flag, and supports **anchored mode**: supplying exactly
one of the two flags pins that dimension and auto-derives the other via
bisection against the in-memory peak-memory estimator. The auto-sizer is
IG-aware (`1.22.2-dev`) and applies a safety clamp on tight (~24 GB) VRAM
when `--compute-ig` is enabled to prevent the previous OOM regression on
GTP-scale data. Minimum-class hosts never auto-set chunk sizes — supply
the flags explicitly there.

**Note:** As of `1.21.0-dev` the `tecpg run mlr` `--gene-loci-per-chunk`
and `--meth-loci-per-chunk` options no longer accept the `-g` / `-m`
short forms (they collided with the top-level `--gene-file` /
`--meth-file` short flags). Use the long forms exclusively for
`tecpg run mlr`. Migration example — replace:

```bash
# old (pre-1.21.0-dev): no longer accepted
tecpg run mlr --cis -g 10000 -m 10000
```

with the long forms:

```bash
# new (1.21.0-dev and later)
tecpg run mlr --cis --gene-loci-per-chunk 10000 --meth-loci-per-chunk 10000
```

The `data dummy` and `chunks` subcommands' own `-g`/`-m` short flags are
unchanged.

## Filtration

You may want to include only certain regression results. There are two ways of filtering the results:

1. P-value filtration - all p-values are computed first. Then, regression results with a p-value above a supplied threshold are excluded from the output. This decreases output size and thus increases speed as saving is an expensive operation.
2. Region filtration - region filtration requires annotation files that dictate the positions of methylation and gene expression ids. Then, regressions are filtered by one of the following methods:
   - Cis: the position of the methylation id is within a window containing a certain number of bases upstream and downstream from a certain number of bases (window_base) away from the transcript start site of the gene and they lie on the same chromosome.
   - Distal: same logic as cis, but with different default values for the window_base, upstream, and downstream parameters.
   - Trans: the gene expression id and methylation id lie on different chromosomes.
   - All: no region filtration.

P-value filtration filters results after calculating the regression, and it saves output time. Region filtration filters the input before the regression results are computed, and it saves both output time and computation time.

## Documentation

Currently, the README and the `tecpg ... --help` commands serve as documentation. Within the code, the function docstrings provide a lot of information about the function. The extensive type hints give added insight into the purpose of functions.

For an end-to-end walkthrough of how eCpGs are filtered, prioritized, tested
for enrichment, and visualized across the `pipeline.sh`/`pipelinePost.sh`
workflow (regions, p-values, qr stats, precise p-values, FDR, bootstrap
scores, network nodes/edges), see the living document
[`docs/ecpg-filtering-prioritization.md`](docs/ecpg-filtering-prioritization.md).

The `docs/` directory also carries topic documents that track the code:

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

## Running `tecpg run mlr` directly

If you need to invoke `tecpg run mlr` directly — for example to
prototype a non-default backend or to integrate `tecpg` into another
pipeline — the equivalent of the `pipeline.sh` mapping stage is:

```bash
tecpg -i data -a annot -o output run mlr \
    --mlr-method qr --cis --compute-ig --compute-influence
```

`--mlr-method` selects the backend (`qr` for mapping, `qr_bootstrap` for the
bootstrap stage, `qr_permute` for permutation testing). On the `qr` backend,
`--qr-impl {torch,householder}` selects the QR factorization; `torch`
(`torch.linalg.qr`) is the default, and `householder` is a CUDA-only batched
path. Chunk sizes are auto-selected by the CLI on server-class hosts. See
*Chunking* above and *Performance tuning* below for the available
overrides, and run `tecpg run mlr --help` for the up-to-date option
list (it is the authoritative source — the README intentionally no
longer reproduces it).

## Selecting a GPU when multiple are available

We have run into this issue when using a development system or a cluster (e.g., Sun Grid Engine) where the system has numerous GPUs and selection is necessary. 

Find the ID of the GPU you’d like to use:
```
nvidia-smi
Fri Dec 15 13:33:36 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                  Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A2                       On | 00000000:81:00.0 Off |                    0 |
|  0%   38C    P8                9W /  60W|      0MiB / 15356MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA L4                       On | 00000000:82:00.0 Off |                    0 |
| N/A   54C    P8               18W /  75W|      0MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

Here, we see GPU 0 is the A2 (previous one) and GPU 1 is the L4 (new one).

Selection of the GPU to use can be done through software (e.g., https://discuss.pytorch.org/t/selecting-the-gpu/20276) or using the shell. For software that we are not going to be editing directly (e.g., tecpg), we use the shell variable direction.
 
The environment variable CUDA_VISIBLE_DEVICES can be set when you call python.
 
To use the A2 GPU, the following re-mapping works:
```bash
CUDA_VISIBLE_DEVICES=1,0 python tecpg run mlr --all --p-thresh 0.000001 --gene-loci-per-chunk 100 --meth-loci-per-chunk 100000
```

To use the L4 GPU, the following re-mapping works:
```bash
CUDA_VISIBLE_DEVICES=0,1 python tecpg run mlr --all --p-thresh 0.000001 --gene-loci-per-chunk 100 --meth-loci-per-chunk 100000
```

## Performance tuning

The CLI exposes several knobs to overlap GPU compute with host I/O and BLAS
work. Most of them auto-resolve from the active `--host-profile` (see
*CUDA* above) and rarely need to be touched, but the following overrides
are available when a run is GPU-, save-, or CPU-bound:

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
* `--gene-loci-per-chunk` / `--meth-loci-per-chunk`: see *Chunking* above.
  Supplying exactly one pins that axis and lets the auto-sizer choose the
  other.

Rules of thumb: if VRAM is full but GPU SM% is low, try `--prefetch-chunks 2`;
if CPU is saturated by writers, lower `--save-threads`; if host BLAS is
fighting the GPU feeder, set `--blas-threads 2`.

For a deeper investigation, `profiling.sh` (added in `1.15.0-dev`) drives
`nvidia-smi`, `top`, `vmstat`, and `pidstat` alongside PyTorch debug output,
sweeps prefetching / chunk size / TF32 / BLAS thread configurations, and
emits an environment-annotated results tarball plus a `Verdict:` line that
classifies the bottleneck (GPU-, save-, or CPU-bound). See `docs/profiling.md`
for details.

The per-chunk startup banner reports the effective values
(`save_threads_effective`, `prefetch_chunks_effective`,
`blas_threads_effective`, logical and physical CPU counts) and per-chunk
metrics (`gpu_idle_between_chunks_ms`, `save_queue_depth`, `prefetch_fill`)
are emitted with an end-of-run statistical summary to help diagnose
bottlenecks.

# Part B — Demonstration: the GTP/MESA golden path

The remainder of this README is a self-contained demonstration of `tecpg` on two public datasets (GTP and MESA), driven by the `pipeline*.sh` scripts and the helper scripts under `tools/`. None of this is required to use `tecpg` itself (Part A) — it is one complete, reproducible worked example.

## Demo datasets (GTP and MESA)

Two real-world public datasets are bundled as turn-key demonstrations,
in addition to the synthetic `dummy` dataset used for smoke tests. Both
are downloaded directly from GEO by the `tecpg data` sub-commands and
are the same datasets used by Kennedy et al. *BMC Genomics* (2018)
**19:476** (`10.1186/s12864-018-4842-3`), whose published eCpG–transcript
pairs are automatically downloaded alongside the raw matrices for use as
a benchmark reference list.

* **GTP — Grady Trauma Project** (`tecpg data gtp`,
  `./pipeline.sh --dataset gtp`). A study of *n ≈ 340* primarily
  African-American adults recruited from urban primary-care clinics in
  Atlanta, GA, designed to characterize the genetic and epigenetic
  correlates of trauma exposure and PTSD. Whole-blood DNA methylation
  was assayed on the Illumina HumanMethylation450 BeadChip (GEO
  accession **GSE72680**, ~349k CpG loci) and gene expression on the
  Illumina HumanHT-12 v4 BeadChip (GEO accession **GSE58137**,
  ~39k expression probes). The matched Kennedy 2018 eCpG list is
  pulled from
  `MOESM1_ESM.txt` of the supplementary materials.

* **MESA — Multi-Ethnic Study of Atherosclerosis** (`tecpg data mesa`,
  `./pipeline.sh --dataset mesa`). A multi-site, multi-ethnic
  longitudinal cohort focused on the subclinical-to-clinical
  progression of cardiovascular disease. CD14+ monocyte DNA
  methylation was assayed on the Illumina HumanMethylation450 BeadChip
  (GEO accession **GSE56046**) and matching gene expression on the
  Illumina HumanHT-12 v4 BeadChip (GEO accession **GSE56045**), giving
  a several-hundred-sample paired methylation/expression cohort. The
  matched Kennedy 2018 eCpG list is pulled from `MOESM2_ESM.txt`.

Both datasets share the same downstream array combination
(HumanMethylation450 + HumanHT-12 v4), so the comprehensive BED6
annotation files shipped under `demo/` apply unchanged to either, and
`pipelinePre.sh` plus `pipeline.sh` wire up identical processing for
`--dataset gtp` and `--dataset mesa` (data prep → probe blacklist →
methylation-derived ancestry instruments → EpiDISH cell proportions →
categorical encoding → residualized PCA → MLR + IG + influence → merge →
region annotation → precise p-values → BH-FDR / diagnostics → influence flag →
bootstrap candidate list → bootstrap evaluation).

A third option, `gtpsub`, is a locus-subsampled GTP build (10,000 CpGs and
5,000 expression probes by default, seed 42) intended for fast wiring checks
on real data; like `dummy` it skips the ancestry and EpiDISH stages.

## Quick start (the golden path)

The recommended way to reproduce an end-to-end demo run is to first prepare the
dataset with `pipelinePre.sh` and then run `pipeline.sh`, which is the
authoritative mapping entry point and uses the same
`mlr --mlr-method qr --compute-ig` invocation, dataset defaults, and
downstream tools described in the *Pipeline stages* subsections below.

```bash
# CIS-only run on the GTP demo dataset
./pipelinePre.sh --dataset gtp
./pipeline.sh --dataset gtp --mapping cis

# Genome-wide run on the MESA demo dataset
./pipelinePre.sh --dataset mesa
./pipeline.sh --dataset mesa --mapping all

# Smoke-test the full pipeline on synthetic data
./pipelinePre.sh --dataset dummy
./pipeline.sh --dataset dummy --mapping all
```

`pipelinePre.sh` downloads the dataset (GTP/MESA only), populates
`data_<dataset>/`, copies the comprehensive BED6 annotations from `demo/` into
`annot_<dataset>/`, and produces the `M.csv`, `G.csv`, and `C.csv` matrices.
`pipeline.sh` then walks through the mapping and downstream stages, writing
artifacts and diagnostic plots into `output_<dataset>/`. Any individual stage
can be resumed with `--start-stage <name>` (see the stage lists below).

After `pipeline.sh` finishes, run `./pipelinePost.sh <dataset>` to
produce Circos / volcano / Manhattan / bipartite network
visualizations from `output_<dataset>/bootstrap_merged.parquet`.

## Pipeline stages

The demonstration is driven by four orchestration scripts, run in order. Each reuses the per-dataset working directories (`data_<dataset>/`, `annot_<dataset>/`, `output_<dataset>/`) and can resume from any stage via `--start-stage`.

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

The annotation files used in the `prep` stage default to the comprehensive BED6
annotations under `demo/` (`annoEPIC_comprehensive.hg19.bed6` and
`annoHT12_comprehensive.hg19.bed6`, originally generated in `1.27.4-dev` with a
validated multi-source HT-12 pipeline and regenerated in `2.0.0b2.dev77` so
that probes without positional evidence are emitted as unmapped rather than
given fabricated positions), with a graceful fallback to the original
`annoEPIC.hg19.bed6` / `annoHT12.hg19.bed6` files.

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
   --compute-influence`, with chunk sizes auto-selected by the CLI's
   `_auto_chunk_sizes` (overridable by exporting `TECPG_M_CHUNK` /
   `TECPG_G_CHUNK`). `MAP_P_THRESH` (default `0.001`, matching the CLI's
   own `-p` default) is the catalog's inclusion gate: pairs above it are
   never written. It is set explicitly in `pipeline.sh` so it appears in
   the run log. Logs are tee'd to `mlr_run_<dataset>.log` and
   `TOTAL_TESTS` is extracted from that log for downstream FDR.
2. **`merge` — Merge chunked output** *(stage `[4/9]`)*.
   `tools/mergeOutputs.py` combines per-chunk files into a single
   `output_<dataset>/merged.parquet`; intermediate chunk files are
   deleted.
3. **`annotate` — Region annotation** *(stage `[5/9]`)*. First derives a
   probe-gene map (`annot_<dataset>/probe_gene_model.tsv`) with
   `tools/build_probe_gene_model.py` from the GENCODE GTF at
   `$TECPG_GENCODE_GTF` (default
   `encode_beds/gencode.v49lift37.annotation.gtf.gz`; `dummy` gets a
   synthetic fixed-span map instead, whose labels carry no biological
   meaning), reusing an existing map only when both the GTF and the
   staged `G.bed6` still match its header. Then
   `tools/assignRegionToEcpg_parquet.py` annotates each pair with one of
   seven strand-aware regions — `PROMOTER`, `GENEBODY`, `CIS5`, `CIS3`,
   `DISTAL5`, `DISTAL3`, `TRANS` (5′/3′ relative to the gene's strand) —
   using the gene spans from that map and writes `annotated.parquet`;
   pairs whose probe or gene lacks an annotation are summarized downstream
   as `UNKNOWN`. Missing-annotation probe IDs
   are collected into a sidecar `annotation_missing_ids.txt` (since
   `1.27.6-dev`).
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

#### Integrated Gradients (IG) Covariates

The pipeline computes per-feature saliency (Integrated Gradients) to measure the relative contribution of methylation vs. covariates. Because computing this for every genome-wide eQTM pair inflates the intermediate output files, the feature is scoped by stage using two variables near the top of `pipeline.sh`:

*   `MLR_IG_COVARIATES`: controls Stage 3 (genome-wide mapping). `"all"`
    (the current default) emits per-covariate IG columns; `"none"` emits only
    the scalar `mt_ig`; a comma-separated list restricts IG to those
    covariates.
*   `BOOTSTRAP_IG_COVARIATES`: controls Stage 9 (bootstrap), default `"all"`.
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
   (`bootstrap_merged.parquet`). This analysis was previously part of
   `tools/summarizeOutput_parquet.py`. `tools/summarizeEnrichment.py` then
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

* `-d, --dataset {dummy,gtp,gtpsub,mesa}` — which dataset to use. Must match the dataset already prepared by `pipelinePre.sh`.
* `--master-parquet PATH` — existing mapping output to score. Also accepts a `sample_reservoir.csv` directly.
* `--reservoir` — score the reservoir universe from a prior `--reservoir-count` map.
* `--cis-enrich` (default) — build a unified gene-anchored master: run a cis write-all map (`--cis-window`, default 1 Mb) and assemble its near-gene pairs with the reservoir's trans/distal pairs via `tools/build_gene_anchored_master.py`, so the per-region evaluation has the near-gene coverage a flat reservoir lacks.
* `-m, --mapping {all}` — the only supported method is `all`. `cis` is accepted by the parser but rejected at runtime because `qr_permute`'s null is trans-global.
* `-s, --start-stage STAGE` — resume from one of `all` (default), `permute`, `eval`.
* `--permutations`, `--subsample-mt-count` (default 2000), `--subsample-g-count` (default 2000), `--seed` — pass-through arguments to `tecpg run mlr`.
* `--total-tests N` — BH denominator for `fdr_permute`; required when the mainline annotation stage runs, and must be the mapping-grid `TOTAL_TESTS` used for `fdr_est`.
* `--no-assign-regions`, `--no-qc-report`, `--no-annotate-mainline` — skip the region-annotation, QC-report, and mainline-annotation work respectively.

> **NOTE:** `--subsample-mt-count` / `--subsample-g-count` subsample the NULL population only. The reported set is always the full M x G cross product; these flags do NOT reduce output size. To get a tractable reported set, physically subset `data_<ds>/M.csv` and `data_<ds>/G.csv` into a smaller `data_<ds>` first. Subsample LOCI, never SAMPLES -- dropping samples changes DF.

> **NOTE:** The `dummy` dataset is a WIRING SMOKE TEST ONLY. Disbelieve its numbers. Dummy annotations are chrom=randrange(1,23) over random data, so cis and trans are exchangeable BY CONSTRUCTION and the stratify arm will return 'single_global_null_adequate' trivially. It says nothing about real data.

The script runs in five stages, reusing the per-dataset working directories `data_<dataset>/`, `annot_<dataset>/`, and `output_<dataset>/`:

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

There are times when we may want to define our own classifications
for a region (e.g., CIS) and apply different annotations to our
mapping data. The standard, supported path is the Parquet-based
classifier driven by `pipeline.sh` (stage `annotate`,
`[5/9]`).

To run it standalone against a merged Parquet produced by an
out-of-band `tecpg run mlr --all ...` invocation, first derive the
probe-gene map from a GENCODE GTF and then classify:

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
supplies the gene span (from the gene model, not the probe footprint)
that the region windows are measured against. See
[`docs/annotation.md`](docs/annotation.md) for details.

Pre-built comprehensive BED6 annotation files for the Illumina EPIC
and HT-12 v4 arrays are shipped under `demo/`:

* `demo/annoEPIC_comprehensive.hg19.bed6` and
  `demo/annoEPIC_comprehensive.hg38.bed6`
* `demo/annoHT12_comprehensive.hg19.bed6` and
  `demo/annoHT12_comprehensive.hg38.bed6`

These were generated in `1.27.4-dev` with
`tools/generate_annotations.py`, which uses a validated multi-source
HT-12 mapping pipeline (Re-Annotator → GEO → UCSC WG-6, with NA
fallback and provenance tracking) and correctly handles unmapped
probes, alternate/unplaced contigs, and pseudoautosomal labels, and were
regenerated in `2.0.0b2.dev77` so that probes without positional evidence
stay unmapped instead of receiving fabricated positions. The
defaults follow Kennedy et al. *BMC Genomics* (2018) **19:476**, split by
strand: `CIS5` / `CIS3` within 50 kb of the gene on its 5′ / 3′ side,
`DISTAL5` / `DISTAL3` beyond 50 kb on the same chromosome, `PROMOTER`
± 2.5 kb of the TSS, `GENEBODY` within the gene span, and `TRANS` on a
different chromosome. Override these in the script's defaults
block if you need different cutoffs. The script annotates every row it
is given; it applies no p-value filter of its own.

> **Legacy CSV path:** the original per-chunk CSV classifier
> `tools/assignRegionToEcpg.py` is retained for backwards
> compatibility with pre-Parquet outputs but is no longer the
> recommended entry point. New work should use the Parquet variant
> above, which is what `pipeline.sh` runs.

## Tools and helper scripts

The `tools/` directory contains the supporting scripts driven by
`pipelinePre.sh`, `pipeline.sh`, `pipelinePost.sh`, and
`pipelinePermute.sh`. They can also be invoked standalone.

Data preparation and QC:

* `tools/generateProbeBlacklist.sh` / `generateProbeBlacklist.R` —
  build a probe blacklist (SNP-affected, cross-reactive, sex-chromosome) from
  the DMRcatedata ExperimentHub lists, scoped to the array in use. Defaults to
  **450k**; pass `epic` or `both` as the second argument, or set `METH_ARRAY`
  for `pipelinePre.sh`. The DMRcate source lists span 450K and EPICv1; the
  array argument selects which manifest to scope the output to. Output is
  `probes_blacklist.csv` with columns `Probe_ID,Reason` (Reason is one or more
  of SNP / CROSSREACTIVE / SEXCHROM, `;`-joined).
  The superseded `generateEpicProbeBlacklist.sh` / `_v2.R` derived
  sex-chromosome probes from the EPIC manifest alone and so missed 450K probes
  with no EPIC counterpart.
* `tools/exclude_blacklisted_probes.py` — drop blacklisted CpGs from
  `M_orig.csv` to produce `M.csv`.
* `tools/exploreOmics.py` — QC metrics, plots, and a consolidated HTML
  report for the original and processed methylation/expression matrices.
* `tools/estimateCellProportions.R` / `estimateCellProportions.sh` — run
  EpiDISH for immune cell-proportion estimation; M-value aware
  (`1.23.1-dev`).
* `tools/residualize_pca.py` / `residualize_pca.sh` — residualize against
  covariates and emit principal-component covariates.
* `tools/preprocessPcaCovariates.py` — PCA preprocessing for covariates
  used by `pipelinePre.sh`.
* `tools/ancestry_probes_report.py` — evaluate methylation-derived ancestry
  instruments and emit scores, a probe table, and an HTML/JSON report.
* `tools/mergeCovariateColumns.py` — merge selected sidecar columns (e.g.
  ancestry components) into the covariate matrix.
* `tools/encodeCategorical.py` — expand integer-coded categorical covariates
  into indicator columns with a minimum cell-size guard.
* `tools/subsample_loci.py` — subsample rows (loci, never samples) of a
  matrix; used to build the `gtpsub` dataset.
* `tools/install_dependencies.R` — install all R packages required by
  the tools (`pheatmap`, `EpiDISH`, `sva`, `IlluminaHumanMethylationEPIC*`,
  `ExperimentHub`) via `BiocManager`.

Annotation:

* `tools/generate_annotations.py` — regenerate comprehensive HT-12 / EPIC
  BED6 annotations from Re-Annotator, GEO, and UCSC sources, with
  provenance tracking.
* `tools/annotation_io.py` — shared annotation readers (transparently reads
  gzipped files) and drop-if-ambiguous probe/symbol → gene-model resolvers.
* `tools/build_probe_gene_model.py` — derive the ILMN probe → gene-model map
  (`probe_gene_model.tsv`) from a GENCODE GTF, or a synthetic fixed-span map
  for `dummy`.
* `tools/assignRegionToEcpg_parquet.py` and `tools/assignRegionToEcpg.py` —
  Parquet- and CSV-based region assignment into the seven strand-aware
  labels (PROMOTER / GENEBODY / CIS5 / CIS3 / DISTAL5 / DISTAL3 / TRANS). The Parquet variant takes the `--gene-model` map and
  writes a sidecar `annotation_missing_ids.txt` of unmatched probes
  (`1.27.6-dev`).

Mapping post-processing:

* `tools/mergeOutputs.py` — merge per-chunk CSV/Parquet outputs into a
  single file (skips `sample_reservoir.csv`).
* `tools/recalculate_pvalues_parquet.py` / `recalculate_pvalues.py` —
  recompute p-values from t-statistics with high precision, replacing the
  normal-CDF approximation with Student's-t.
* `tools/summarizeOutput_parquet.py` / `summarizeOutput.py` — global
  BH-FDR, top-hits table, QQ / histogram / saliency plots, and regional FDR
  summaries. (Functional/ENCODE enrichment was moved to
  `tools/runEnrichment.py`.)
* `tools/runEnrichment.py` — standalone functional (Enrichr/`gseapy`/`mygene`)
  and optional ENCODE ChromHMM enrichment of significant genes. Reads the FDR
  summary (`--fdr-input summarized.parquet`) and/or the bootstrap IG ranking
  (`--ig-input bootstrap_merged.parquet`) selected via `--rank-by fdr ig`, and
  is run as the final stage of `pipelinePost.sh`.
* `tools/summaryParquetToCsv.py` — Parquet→CSV converter for summary
  files.
* `tools/summarizeEnrichment.py` — self-contained HTML summary of the
  enrichment results.
* `tools/chromatin_features.py` / `tools/chromatinEnrichment_parquet.py` —
  interval index over chromatin-feature tracks and the Kennedy Fig. 6
  chromatin enrichment (Fisher exact statistics, BH q-values, and a
  `--plot` two-panel heatmap coloured by log odds ratio).

Influence and permutation diagnostics:

* `tools/flagInfluence_parquet.py` — derive `mt_influence_flag` from the
  mapper's `mt_h_max` leverage column and emit an influence QC JSON.
* `tools/calibration_bridge.py`, `tools/fig_influence_dose_response.py`,
  `tools/influence_diagnostic_panels.py`, `tools/influence_pair_anatomy.py`,
  `tools/diagnose_se_ratio_trend.py`, `tools/se_ratio_trend_report.py` —
  influence calibration against bootstrap fragility, dose-response and
  SE-ratio figures and reports.
* `tools/influence_qc_report.py` — consolidated FastQC-style HTML report over
  the influence artifacts.
* `tools/eval_permute.py` — read-only audit of a `qr_permute` parquet,
  producing `eval_permute_report.json`.
* `tools/summarize_permute.py` / `tools/read_permute_diagnostics.py` /
  `tools/plot_permute_diagnostics.py` — permutation summaries, the 7-way
  region table, and diagnostic plots across cohorts.
* `tools/permute_qc_report.py` — self-contained HTML QC report for a
  permutation run.
* `tools/annotate_permute_p.py` / `tools/join_precise_p_permute.py` — write
  `p_permute` / `fdr_permute` onto the mainline catalogs.
* `tools/build_gene_anchored_master.py` — assemble the cis near-gene pairs and
  the reservoir trans/distal pairs into the master scored by `qr_permute`.
* `tools/reservoir_to_parquet.py` — convert `sample_reservoir.csv` into a
  master parquet.
* `tools/compare_perm_vs_analytic.py` — compare permutation and analytic
  p-values.

Bootstrapping:

* `tools/createBootstrapList.py` — pick the top hits (by p-value, with
  per-region floors and caps) to feed the `qr_bootstrap` MLR backend.
* `tools/annotate_bootstrap_concordance.py` — raw bootstrap / analytic
  concordance scores and a distribution summary.

Visualization and network analysis:

* `tools/plotCircos.py` — Circos plots of the eQTM architecture. Uses the
  hg19 UCSC `cytoBand.txt` (downloaded automatically by `pipelinePost.sh`)
  and reports detailed reasons for excluded CpG-Gene pairs.
* `tools/visualizeFindings.py` — volcano, Manhattan, and scatter plots;
  emits a full set of plots for each available p-value column
  (`p_boot`, `precise_mt_p`, `mt_p`) with prefixed filenames.
* `tools/evaluateSaliency.py` — integrated-gradients saliency diagnostics,
  with an optional `--frac-exclude` pass that removes expression-derived IG
  from the saliency denominator.
* `tools/ig_qc_report.py` — self-contained HTML QC report over the IG
  columns: coverage, whether `|mt_ig|` is an independent ranking axis
  (against `|t|`), what drives its magnitude, the methylation share of
  attribution — with an optional `--frac-exclude` (e.g. `'Exp_PC*_ig'`)
  second denominator reported alongside the raw one — and per-region IG.
* `tools/plotRegionProportions.py` — regional composition plots.
* `tools/exportBipartiteNetwork.py` — Cytoscape-formatted node and edge
  tables (with optional `--min-effect`, `--max-boot-p`, `--max-fdr`, and
  `--top-k` filtering and an explicit `--out-dir`).
* `tools/visualizeBipartiteNetwork.py` — ForceAtlas2-based energy-minimized
  bipartite network, UMAP of regulatory β-diversity, regulatory degree
  distribution, clustered bipartite adjacency heatmap, a signed `mt_est`
  heatmap, a hypergeometric gene–gene projection, `--per-region` stratified
  figures, and arc diagrams; handles duplicate edges by keeping the
  maximum-weight pair.

Benchmarking and profiling:

* `pipelineBenchmarkKennedy.sh` / `tools/benchmark_kennedy.py` — comparison against the Kennedy et al.
  benchmark, standardizing thresholds (1e-5 and 1e-11) across cohorts, with
  an eligibility decomposition (testable vs blacklisted vs otherwise absent
  Kennedy pairs), a probe-blacklist audit, effect-size / t-statistic / sign
  concordance on the shared pairs, influence-stratified recovery, and a
  region-composition crosswalk to Kennedy's four categories.
* `tools/diagnose_overlap.py` / `tools/check_catalog_grid.py` — overlap and
  catalog-grid consistency diagnostics for benchmark comparisons.
* `tools/io_microbench.py` — IO microbenchmarks for the save pool.
* `profiling.sh` and `docs/profiling.md` — bottleneck diagnostic harness.

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

This work was partially supported by an NIH NCI MERIT award (R37, CA233774, PI: Kober) and Cancer Center Support Grant (P30, CA082103, Co-I: Olshen).

