# tecpg

Torch-eCpG is a GPU enabled expression quantitative trait methylation (eQTM) mapper to identify expression associated CpG (eCpG) loci with python CLI using pytorch.

If you use Torch-eCpG in your research, please cite the following paper: Kober, K.M., Berger, L., Roy, R. et al. Torch-eCpG: a fast and scalable eQTM mapper for thousands of molecular phenotypes with graphical processing units. BMC Bioinformatics 25, 71 (2024). https://doi.org/10.1186/s12859-024-05670-4

The current development version on the `dev` branch is **1.27.6-dev**. Since the
`1.0.0` release on `main`, the project has grown an end-to-end analysis
pipeline (`pipeline.sh`), a downstream visualization/network pipeline
(`pipelinePost.sh`), Parquet output, anchored auto chunk-sizing, Integrated
Gradients (IG), an empirical bootstrap MLR backend, support for the MESA
dataset, comprehensive HT-12/EPIC BED6 annotations, and a host-profile aware
CLI. See `CHANGELOG.md` for the full per-version history.

## Docker Image

A docker image is now available for the Torch-eCpG (tecpg) tool to perform eQTM mapping analysis. The docker image provides a pre-configured environment for running tecpg.

The image can be created from the instructions in the docker-related/ directory.

Alternatively, a full image is available for download from docker hub:
https://hub.docker.com/r/kordk/torch-ecpg

## Installation

Pip install from github using `git+https://`.

```bash
pip install git+https://github.com/kordk/torch-ecpg.git
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

## CUDA

`tecpg` can calculate on the CPU or on a CUDA enabled GPU device. CUDA devices are generally faster than CPU computations for sufficiently large inputs.

The program will automatically determine whether there is a CUDA enabled device and use it if available. To force calculation on the CPU, set the `--threads` option to a nonzero integer. This will also set the number of CPU threads used.

The top-level CLI also accepts `--host-profile {auto,minimum,server}` (envvar
`TECPG_HOST_PROFILE`). `auto` (default) inspects the host (physical CPU count
and total RAM) and picks `minimum` for laptop-class hosts (`<12 cores` or
`<32 GB`) and `server` otherwise. The resolved profile drives defaults for the
save pool, output format (`parquet` on `server`, `csv` on `minimum`),
prefetch depth, and chunk auto-sizing. Explicit per-flag overrides
(`--save-threads`, `--output-format`, `--prefetch-chunks`,
`--gene-loci-per-chunk`, `--meth-loci-per-chunk`, `--blas-threads`) always win.

## Full analysis pipeline (`pipeline.sh`)

`pipeline.sh` is the recommended way to run the demo analysis end to end. It
wraps `tecpg` and the helper scripts in `tools/` into a nine-stage workflow,
with structured logging, dataset-aware defaults, and the ability to resume
from any stage.

```bash
./pipeline.sh --help
./pipeline.sh --dataset dummy --mapping all
./pipeline.sh --dataset gtp   --mapping cis
./pipeline.sh --dataset mesa  --mapping all
./pipeline.sh --dataset gtp   --mapping all --start-stage map
```

Options:

* `-d, --dataset {dummy,gtp,mesa}` — which dataset to use. `dummy` generates a
  small synthetic dataset for testing; `gtp` downloads and prepares the Grady
  Trauma Project data via `tecpg data gtp`; `mesa` does the same for MESA via
  `tecpg data mesa`.
* `-m, --mapping {all,cis}` — region filter passed through to
  `tecpg run mlr` (`--all` or `--cis`).
* `-s, --start-stage STAGE` — resume from one of `all` (default), `prep`,
  `cell_prop`, `pca`, `map`, `merge`, `annotate`, `precise_p`, `summarize`,
  `boot_list`, `bootstrap`. Context variables (`DF`, `TOTAL_TESTS`) are
  recomputed from the on-disk artifacts so any stage can run on its own.

The script creates per-dataset working directories `data_<dataset>/`,
`annot_<dataset>/`, and `output_<dataset>/`, and runs the following
stages. Each stage name (in `code`) matches the value accepted by
`--start-stage`, so any individual step can be re-run in isolation.

1. **`prep` — Data preparation** *(stage `[1/9]`)*. Downloads or
   generates the dataset (`tecpg data {dummy,gtp,mesa}`), copies the
   default comprehensive BED6 annotations into `annot_<dataset>/`
   (with a graceful fallback to the original `annoEPIC.hg19.bed6` /
   `annoHT12.hg19.bed6`), applies the EPIC probe blacklist
   (`tools/generateEpicProbeBlacklist.sh` +
   `tools/exclude_blacklisted_probes.py`) to produce `M.csv` from
   `M_orig.csv`, and runs `tools/exploreOmics.py` to write QC plots
   and an HTML report under `data_<dataset>/qc/`.
2. **`cell_prop` — Immune cell-proportion estimation** *(stage
   `[1.5/9]`)*. `tools/estimateCellProportions.sh` runs EpiDISH
   (M-value aware) on real datasets to produce
   `C_post_cellTypes.csv`. Skipped for `dummy` (random noise causes
   singular fits); `C_orig.csv` is copied through instead.
3. **`pca` — Residualization & PCA** *(stage `[2/9]`)*.
   `tools/residualize_pca.sh` generates expression and methylation
   principal components, which are merged with the cell-proportion
   covariates to produce the final `C.csv`. Degrees of freedom
   (`DF = SAMPLES − COVARS − 2`) are recomputed from `C.csv` for use
   by the precise-p-value stage.
4. **`map` — eQTM mapping** *(stage `[3/9]`)*. Runs `tecpg ... run
   mlr --mlr-method lstsq --<mapping> --compute-ig`, with chunk sizes
   auto-selected by the CLI's `_auto_chunk_sizes` (overridable by
   exporting `TECPG_M_CHUNK` / `TECPG_G_CHUNK`). Logs are tee'd to
   `mlr_run_<dataset>.log` and `TOTAL_TESTS` is extracted from that
   log for downstream FDR.
5. **`merge` — Merge chunked output** *(stage `[4/9]`)*.
   `tools/mergeOutputs.py` combines per-chunk files into a single
   `output_<dataset>/merged.parquet`; intermediate chunk files are
   deleted.
6. **`annotate` — Region annotation** *(stage `[5/9]`)*.
   `tools/assignRegionToEcpg_parquet.py` annotates each pair with
   `CIS`/`DISTAL`/`TRANS`/`PROMOTER`/`GENEBODY` and writes
   `annotated.parquet`. Missing-annotation probe IDs are collected
   into a sidecar `annotation_missing_ids.txt` (since `1.27.6-dev`).
7. **`precise_p` — High-precision p-values** *(stage `[6/9]`)*.
   `tools/recalculate_pvalues_parquet.py` replaces the normal-CDF
   approximation with Student's-t p-values using the degrees of
   freedom derived from `C.csv`, writing `annotated_pcalc.parquet`.
8. **`summarize` — FDR and summary** *(stage `[7/9]`)*.
   `tools/summarizeOutput_parquet.py` computes a global
   Benjamini–Hochberg FDR (using `TOTAL_TESTS` dynamically extracted
   from the `mlr` log), writes `summarized.parquet`, and emits QQ,
   histogram, and saliency diagnostic plots into `output_<dataset>/`.
9. **`boot_list` — Bootstrap candidate list** *(stage `[8/9]`)*.
   `tools/createBootstrapList.py` selects the top hits (ranked by
   p-value) for bootstrapping into `bootstrap_list.csv`.
10. **`bootstrap` — Bootstrap evaluation** *(stage `[9/9]`)*. Runs
    `tecpg ... run mlr --mlr-method lstsq_bootstrap --pairs-file ...
    --master-parquet ... --bootstrap-iterations 100
    --bootstrap-batch-size 10 --compute-ig` to attach empirical
    bootstrap p-values to the top candidates and write
    `bootstrap_merged.parquet`.

The annotation files used in stage 1 default to the comprehensive BED6
annotations under `demo/` (`annoEPIC_comprehensive.hg19.bed6` and
`annoHT12_comprehensive.hg19.bed6`, regenerated in `1.27.4-dev` with a
validated multi-source HT-12 pipeline), with a graceful fallback to the
original `annoEPIC.hg19.bed6` / `annoHT12.hg19.bed6` files.

## Post-processing pipeline (`pipelinePost.sh`)

`pipelinePost.sh` consumes `output_<dataset>/bootstrap_merged.parquet`
produced by `pipeline.sh` and runs the visualization and network-analysis
tools:

```bash
./pipelinePost.sh gtp
./pipelinePost.sh mesa
```

The script downloads the UCSC hg19 `cytoBand.txt` if missing and then runs,
in order:

1. `tools/plotCircos.py` — Circos plots of the eQTM architecture
   (`output_<dataset>/plots/`).
2. `tools/visualizeFindings.py` — volcano, Manhattan, scatter, and related
   plots. Generates a full set of figures for every available p-value
   column (bootstrap `p_boot`, `precise_mt_p`, `mt_p`) with prefixed
   filenames.
3. `tools/exportBipartiteNetwork.py` — Cytoscape-formatted node and edge
   tables under `output_<dataset>/network/`, filtered by `--top-k 500`
   and `--max-boot-p 0.05` by default.
4. `tools/visualizeBipartiteNetwork.py` — energy-minimized bipartite
   network, UMAP of regulatory β-diversity, regulatory degree distribution,
   clustered bipartite adjacency heatmap, and arc diagrams.

## Input data

Methylation values, gene expression values, and covariates are provided in CSV or TSV files in the `<working>/data` directory. For methylation and gene expression, columns are for individual samples and each row is for a loci. For the covariates, the columns are the type of covariate and the rows are the sample. Annotation files are used for region filtration and are stored in the `<working>/annot`. They use the `BED6` standard and store the positions of the methylation or gene expression loci.

Methylatlion CSV datafiles from the GTP dataset (see Demostration below):
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

Gene expression CSV datafiles from the GTP dataset (see Demostration below):
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

Covariate CSV datafiles from the GTP dataset (see Demostration below):
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

By default, the output format follows `--host-profile`: `parquet` on
server-class hosts and `csv` on minimum-class hosts. Use
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
saliency values are written alongside the regression results. After the
post-mapping stages of `pipeline.sh`, additional columns include the
high-precision p-value (`precise_mt_p`), the assigned region
(`region`/`Region`), the global BH-FDR q-value, and (after the bootstrap
stage) the empirical bootstrap p-value `p_boot`.


### Integrated Gradients (IG) Covariates

The pipeline computes per-feature saliency (Integrated Gradients) to measure the relative contribution of methylation vs. covariates. Because computing this for every genome-wide eQTM pair significantly bloats the intermediate output files (~12 float columns per row across 150M rows adds >5GB per file), the feature is scoped by stage using two variables near the top of `pipeline.sh`:

*   `MLR_IG_COVARIATES`: Defaults to `"none"` for Stage 3 (genome-wide mapping). Only scalar `mt_ig` is produced.
*   `BOOTSTRAP_IG_COVARIATES`: Defaults to `"all"` for Stage 9 (bootstrap). Because the bootstrap runs on a small, prioritized candidate list (e.g. 20,000 rows), enabling full per-feature IG costs very little space (~1MB) while enabling full fraction-based saliency analysis downstream.

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

## MLR approximate p-values

The p-values returned by `tecpg run mlr` are approximations using the normal distribution CDF. This approximation is more accurate for larger degrees of freedom. As the number of degrees of freedom approaches $+\infty$, the CDF of the normal distribution and the Student's T distribution approach. The approximation is done because pytorch does not support the Student CDF and does not have the needed funtions to implement it efficiently.

For example:

- For 336 degrees of freedom and test t-statistic of 1.877, the percent difference between the normal CDF and Student CDF is 0.04469%.
- For 50 degrees of freedom and test t-statistic of 1.877, the percent difference between the normal CDF and Student CDF is 0.30206%.

The user should determine whether this accuracy is suitable for the task and the degrees of freedom.

This image from https://en.wikipedia.org/wiki/Student%27s_t-distribution shows the deviation of the Student's T distribution CDF from the normal CDF represented as $v=+\infty$:

<details open>
<summary> Student T CDF comparison </summary>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Student_t_cdf.svg/325px-Student_t_cdf.svg.png">
</details>

## Documentation

Currently, the README and the `tecpg ... --help` commands serve as documentation. Within the code, the function docstrings provide a lot of information about the function. The extensive type hints give added insight into the purpose of functions.

For an end-to-end walkthrough of how eCpGs are filtered, prioritized, tested
for enrichment, and visualized across the `pipeline.sh`/`pipelinePost.sh`
workflow (regions, p-values, lstsq stats, precise p-values, FDR, bootstrap
scores, network nodes/edges), see the living document
[`docs/ecpg-filtering-prioritization.md`](docs/ecpg-filtering-prioritization.md).

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
`pipeline.sh` wires up identical processing for `--dataset gtp` and
`--dataset mesa` (data prep → EpiDISH cell proportions → residualized
PCA → MLR + IG → merge → region annotation → precise p-values →
BH-FDR / diagnostics → bootstrap candidate list → bootstrap
evaluation).

## Demonstration

The recommended way to reproduce an end-to-end demo run is via
`pipeline.sh`, which is the authoritative entry point and uses the
same `mlr --mlr-method lstsq --compute-ig` invocation, dataset
defaults, and downstream tools described in
*Full analysis pipeline* above.

```bash
# CIS-only run on the GTP demo dataset
./pipeline.sh --dataset gtp --mapping cis

# Genome-wide run on the MESA demo dataset
./pipeline.sh --dataset mesa --mapping all

# Smoke-test the full pipeline on synthetic data
./pipeline.sh --dataset dummy --mapping all
```

The first invocation will download the dataset (GTP/MESA only),
populate `data_<dataset>/`, copy the comprehensive BED6 annotations
from `demo/` into `annot_<dataset>/`, and walk through all nine
pipeline stages, writing artifacts and diagnostic plots into
`output_<dataset>/`. Any individual stage can be resumed with
`--start-stage <name>` (see the stage list above).

After `pipeline.sh` finishes, run `./pipelinePost.sh <dataset>` to
produce Circos / volcano / Manhattan / bipartite network
visualizations from `output_<dataset>/bootstrap_merged.parquet`.

### Manual CIS-only mapping (advanced)

If you need to invoke `tecpg run mlr` directly — for example to
prototype a non-default backend or to integrate `tecpg` into another
pipeline — the equivalent of the `pipeline.sh` mapping stage is:

```bash
tecpg -i data -a annot -o output run mlr \
    --mlr-method lstsq --cis --compute-ig
```

Chunk sizes are auto-selected by the CLI on server-class hosts. See
*Chunking* and *Performance tuning* below for the available
overrides, and run `tecpg run mlr --help` for the up-to-date option
list (it is the authoritative source — the README intentionally no
longer reproduces it).

## Alternative annotation and assignment of regions

There are times when we may want to define our own classifications
for a region (e.g., CIS) and apply different annotations to our
mapping data. The standard, supported path is the Parquet-based
classifier driven by `pipeline.sh` (stage `annotate`,
`[5/9]`).

To run it standalone against a merged Parquet produced by an
out-of-band `tecpg run mlr --all ...` invocation:

```bash
python3 tools/assignRegionToEcpg_parquet.py \
    -d output/merged.parquet \
    -g annot/G.bed6 \
    -m annot/M.bed6 \
    -o output/annotated.parquet
```

Pre-built comprehensive BED6 annotation files for the Illumina EPIC
and HT-12 v4 arrays are shipped under `demo/`:

* `demo/annoEPIC_comprehensive.hg19.bed6` and
  `demo/annoEPIC_comprehensive.hg38.bed6`
* `demo/annoHT12_comprehensive.hg19.bed6` and
  `demo/annoHT12_comprehensive.hg38.bed6`

These were regenerated in `1.27.4-dev` with
`tools/generate_annotations.py`, which uses a validated multi-source
HT-12 mapping pipeline (Re-Annotator → GEO → UCSC WG-6, with NA
fallback and provenance tracking) and correctly handles unmapped
probes, alternate/unplaced contigs, and pseudoautosomal labels. The
defaults follow Kennedy et al. *BMC Genomics* (2018) **19:476**:
`PVALCUTOFF = 1e-6` (exploratory), `CIS < 50 kb` upstream of TSS,
`DISTAL > 50 kb` from TSS, and `PROMOTER ± 2.5 kb` of TSS. Override
these in the script's defaults block if you need different cutoffs.

> **Legacy CSV path:** the original per-chunk CSV classifier
> `tools/assignRegionToEcpg.py` is retained for backwards
> compatibility with pre-Parquet outputs but is no longer the
> recommended entry point. New work should use the Parquet variant
> above, which is what `pipeline.sh` runs.

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
 
The the environment variable CUDA_VISIBLE_DEVICES can be set when you call python.
 
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
* `--output-format {auto,csv,parquet}`: `auto` resolves to `parquet` on
  server-class hosts and `csv` on minimum-class hosts.
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

## Tools and helper scripts

The `tools/` directory contains the supporting scripts driven by
`pipeline.sh` and `pipelinePost.sh`. They can also be invoked standalone.

Data preparation and QC:

* `tools/generateEpicProbeBlacklist.sh` / `generateEpicProbeBlacklist_v2.R` —
  build an EPIC probe blacklist from packaged Bioconductor annotations.
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
  used by `pipeline.sh`.
* `tools/install_dependencies.R` — install all R packages required by
  the tools (`pheatmap`, `EpiDISH`, `sva`, `IlluminaHumanMethylationEPIC*`,
  `ExperimentHub`) via `BiocManager`.

Annotation:

* `tools/generate_annotations.py` — regenerate comprehensive HT-12 / EPIC
  BED6 annotations from Re-Annotator, GEO, and UCSC sources, with
  provenance tracking.
* `tools/assignRegionToEcpg_parquet.py` and `tools/assignRegionToEcpg.py` —
  Parquet- and CSV-based region assignment (CIS / DISTAL / TRANS /
  PROMOTER / GENEBODY). The Parquet variant writes a sidecar
  `annotation_missing_ids.txt` of unmatched probes (`1.27.6-dev`).

Mapping post-processing:

* `tools/mergeOutputs.py` — merge per-chunk CSV/Parquet outputs into a
  single file (skips `sample_reservoir.csv`).
* `tools/recalculate_pvalues_parquet.py` / `recalculate_pvalues.py` —
  recompute p-values from t-statistics with high precision, replacing the
  normal-CDF approximation with Student's-t.
* `tools/summarizeOutput_parquet.py` / `summarizeOutput.py` — global
  BH-FDR, top-hits table, QQ / histogram / saliency plots, regional FDR
  summaries, and optional ENCODE/`gseapy`/`mygene` enrichment.
* `tools/summaryParquetToCsv.py` — Parquet→CSV converter for summary
  files.

Bootstrapping:

* `tools/createBootstrapList.py` — pick the top hits (by p-value) to feed
  the `lstsq_bootstrap` MLR backend.

Visualization and network analysis:

* `tools/plotCircos.py` — Circos plots of the eQTM architecture. Uses the
  hg19 UCSC `cytoBand.txt` (downloaded automatically by `pipelinePost.sh`)
  and reports detailed reasons for excluded CpG-Gene pairs.
* `tools/visualizeFindings.py` — volcano, Manhattan, and scatter plots;
  emits a full set of plots for each available p-value column
  (`p_boot`, `precise_mt_p`, `mt_p`) with prefixed filenames.
* `tools/exportBipartiteNetwork.py` — Cytoscape-formatted node and edge
  tables (with optional `--min-effect`, `--max-boot-p`, and `--top-k`
  filtering and an explicit `--out-dir`).
* `tools/visualizeBipartiteNetwork.py` — ForceAtlas2-based energy-minimized
  bipartite network, UMAP of regulatory β-diversity, regulatory degree
  distribution, clustered bipartite adjacency heatmap, and arc diagrams;
  handles duplicate edges by keeping the maximum-weight pair.

Benchmarking and profiling:

* `tools/benchmark_kennedy.py` — comparison against the Kennedy et al.
  benchmark.
* `tools/io_microbench.py` — IO microbenchmarks for the save pool.
* `profiling.sh` and `docs/profiling.md` — bottleneck diagnostic harness.

## Acknowledgements

This work was partially supported by an NIH NCI MERIT award (R37, CA233774, PI: Kober) and Cancer Center Support Grant (P30, CA082103, Co-I: Olshen).

