# Annotation

How tecpg assigns a genomic region and a gene identity to each CpG–transcript pair for the GTP and MESA exemplar datasets.

---

## 1. Terminology

Two artifacts are involved, and keeping them distinct matters — conflating them is what produced
the defects this design replaced.

| Term | Artifact | Keyed by | Cardinality (GTP) |
|---|---|---|---|
| **probe→gene map (TSV)** | `annot_<ds>/probe_gene_model.tsv` | ILMN expression probe | ~45–52k rows |
| **pair-level results (mapping parquets)** | the chain in §5 | `(mt_id, gt_id)` | ~17M rows |

The **probe→gene map** is derived once per dataset from a GENCODE GTF. It is loaded transiently,
used to place a CpG against a gene span, and discarded. Nothing downstream of region assignment
reads it. Because it is keyed by probe, a one-to-many probe→gene relation costs nothing here.

The **pair-level results** are where cardinality is load-bearing: FDR denominators,
`tecpg_perm_n_reported`, and bootstrap joins all depend on the row count. A one-to-many relation
therefore cannot be expressed by adding rows; it is compressed to a single row carrying
comma-joined gene identifiers.

---

## 2. Inputs

| File | Supplies | Source |
|---|---|---|
| `annot_<ds>/M.bed6` | CpG position (`mt_chrom`, `mt_chromStart`, `mt_strand`) | `tools/generate_annotations.py` |
| `annot_<ds>/G.bed6` | expression **probe** position (`gt_chrom`, `gt_chromStart`, `gt_strand`) and the cis/trans mapping test space | `tools/generate_annotations.py`, multi-source: Re-Annotator + GEO + UCSC |
| `annot_<ds>/probe_gene_model.tsv` | the **gene span** used for the region windows, plus gene id and symbol | `tools/build_probe_gene_model.py` |
| GENCODE GTF | gene and exon features | `$TECPG_GENCODE_GTF`, default `encode_beds/gencode.v49lift37.annotation.gtf.gz`; `pipeline.sh` downloads the default from `$TECPG_GENCODE_GTF_URL` (GENCODE release 49 GRCh37 mapping) when it is absent |

All coordinates are **1-based inclusive**. The GFF/GTF parser in `tools/annotation_io.py` reads
0-based starts and adds +1 on the way out.

`G.bed6` carries **one position per probe**. Probes the Re-Annotator declined to place are filled
from GEO and UCSC where possible and written as `NA` otherwise; `NA` rows are skipped by every
consumer.

---

## 3. Region assignment

`region` is a property of the **pair**, not of the gene. It answers: *where does this CpG sit
relative to this gene?* The CpG position is what is being classified; the gene supplies only the
reference frame — its start, end, and strand.

Two inputs, two jobs, deliberately separated:

- **coordinates** (`mt_*`, `gt_*`) come from the probe BEDs (`-m`, `-g`) and are written whenever
  that probe id resolves, independent of whether a gene model did;
- **the region label** is computed against the gene span from the probe→gene map
  (`--gene-model`), which is never written verbatim.

A probe whose gene does not resolve therefore loses its region label, never its coordinates.

### 3.1 Decision order

```
if mt_chrom is NULL or gt_chrom is NULL      ->  region = NULL
elif mt_chrom != gt_chrom                    ->  region = TRANS
elif probe not in the probe->gene map        ->  region = NULL
else                                         ->  window arithmetic below
```

`TRANS` is decided on **probe** chromosomes. The cis/trans test space that produced the pairs is
built from `G.bed6`, so deciding otherwise would let a pair tecpg computed as cis be labelled
trans. It follows that an unresolved probe still receives its `TRANS` labels — chromosome
identity needs no gene model.

### 3.2 Windows

`s` = gene `chromStart`, `e` = gene `chromEnd`, `mt` = `mt_chromStart`. Boundaries are 2,500 bp
(promoter) and 50,000 bp (cis/distal).

| Region | `+` strand | `−` strand |
|---|---|---|
| `DISTAL5` | `mt < s − 50000` | `mt > e + 50000` |
| `CIS5` | `s − 50000 ≤ mt < s − 2500` | `e + 2500 < mt ≤ e + 50000` |
| `PROMOTER` | `s − 2500 ≤ mt ≤ s + 2500` | `e − 2500 ≤ mt ≤ e + 2500` |
| `GENEBODY` | `s + 2500 < mt < e` | `s < mt < e − 2500` |
| `CIS3` | `e ≤ mt ≤ e + 50000` | `s − 50000 ≤ mt ≤ s` |
| `DISTAL3` | `mt > e + 50000` | `mt < s − 50000` |
| `TRANS` | different chromosome | different chromosome |

5′/3′ are relative to the gene's strand, so a `−`-strand gene anchors `PROMOTER` and `CIS5` at
`chromEnd`. A pair falling in no window (possible only for a gene shorter than 2,500 bp, or a
strand field that is neither `+` nor `−`) yields `region = NULL` with coordinates intact.

---

## 4. Building the probe→gene map

`tools/build_probe_gene_model.py` takes the GENCODE GTF and `G.bed6` and emits
`probe_gene_model.tsv`. It is stdlib-only and follows the annotation method of Kennedy et al.
2018 (see §6).

### 4.1 Gene assignment is by exon overlap

A probe is assigned to a gene when the probe interval overlaps that gene's merged **exon**
intervals by **more than 25 bp**. Overlapping exons of one gene are collapsed to maximal
intervals first.

Exons, not gene spans: an expression probe is designed against mature transcript, so intersecting
gene spans would attach probes to genes they merely sit inside — introns and nested loci.

The **span** used for the region windows is then that gene's `gene` feature. GENCODE's `gene`
feature is already minimum TSS to maximum TES across transcripts, so no transcript collapse is
needed.

### 4.2 Read-through transcripts

GENCODE marks two different things, and they are handled differently:

| Tag | Meaning | Treatment | Count (v49lift37) |
|---|---|---|---|
| `readthrough_transcript` | one isoform of an otherwise ordinary gene splices through into the neighbour | **exclude the exon** from the overlap test; keep the gene | 38,304 transcripts across 970 genes |
| `readthrough_gene` | the whole locus is a fusion annotation | **drop the gene** when resolving duplicates | 661 genes |

A read-through isoform extends a gene's apparent exon footprint into its neighbour. Since gene
assignment is by exon overlap, those exons would attach a probe to a gene it does not belong to.
Promoting the transcript tag to the gene, by contrast, would discard ~970 ordinary genes for
having one odd isoform.

### 4.3 Resolving several genes at one locus

Where more than one gene qualifies, poorly characterised names (`KIAA*`, `FLJ*`, `LOC*`, anything
containing `orf`) and read-through genes are removed from the duplicates. The filter is applied
**only** to resolve duplicates and **never empties the set**: a probe whose sole candidate is
`LOC999` keeps it.

All surviving genes are retained. The span becomes the **union** of their gene-feature spans, and
`gtf_gene_ids` / `gtf_gene_symbols` are comma-joined in candidate order.

Consequence worth knowing: a union span is wider than any single constituent gene, so `PROMOTER`
and `GENEBODY` are anchored to a boundary that may belong to no one gene.

### 4.4 File format

```
#tecpg_probe_gene_model	v1
#method	exon_overlap_gt_25bp_union_span
#gtf	<absolute path to the GENCODE GTF>
#probe_bed	<absolute path to G.bed6>
#coords	1-based inclusive
#counts	RESOLVED=… NO_OVERLAP=… MULTI_CHROM=…
probe_id	status	chrom	start	end	strand	n_genes	gtf_gene_ids	gtf_gene_symbols	gtf_gene_types
```

Every probe with a position gets a row, so coverage is a census rather than a log line. Only
`RESOLVED` rows enter the lookup; an unresolved probe is simply absent, and the caller's
"probe not in map" branch handles it.

| `status` | Meaning |
|---|---|
| `RESOLVED` | at least one gene passed the exon-overlap test |
| `NO_OVERLAP` | no gene's exons overlap the probe by more than 25 bp |
| `MULTI_CHROM` | surviving genes span different contigs, so a union span is undefined (should not occur) |

`gtf_gene_types` records GENCODE biotype but is **not acted on** — no biotype filter is applied,
so the choice of whether to restrict to protein-coding genes stays with the consumer.

Read it with `annotation_io.readProbeGeneModel()`, which returns the map plus the header
key–value pairs. Inner keys match `readAnnotationFileToDict` (`chrom`, `chromStart`, `chromEnd`,
`strand`), so the region arithmetic is agnostic to which source supplied the span.

---

## 5. Where this runs, and what it writes

The map is derived in `pipeline.sh` stage `[5/9]`, immediately before its only consumer. It is
**not** built in `pipelinePre.sh`: staging inputs and deriving artifacts are different jobs, and
`--start-stage annotate` must still produce it.

The reuse guard matches the GTF basename against the map's header block, so a map built from a
different GENCODE release is re-derived rather than silently reused.

`pipelinePermute.sh` **gates** on the map rather than deriving it — permute always follows a
`pipeline.sh` run. Under `--no-assign-regions` it is not required.

Region assignment writes into two independent lineages:

| File | Written by | Region assigned |
|---|---|---|
| `merged.parquet` | `pipeline.sh [4/9]` | |
| `annotated.parquet` | `pipeline.sh [5/9]` | **yes** |
| `recalculated.parquet` → `summarized.parquet` | `pipeline.sh [6/9]`, `[7/9]` | |
| `reservoir_master.parquet` / `gene_anchored_master.parquet` | `pipelinePermute.sh` | |
| `*.region.parquet` | `pipelinePermute.sh [1/5]` | **yes** |

These are different row sets — the mainline catalogue versus a reservoir sample or reassembled
master — so each needs its own pass. Both read the same materialised map, which is what makes
them resolve identically; permute licensing is read per region.

### 5.1 Columns added

| Column | Content | NULL when |
|---|---|---|
| `mt_chrom`, `mt_chromStart`, `mt_strand` | CpG position | the CpG is absent from `M.bed6` |
| `gt_chrom`, `gt_chromStart`, `gt_strand` | expression **probe** position | the probe is absent from `G.bed6` |
| `region` | one of the seven labels in §3.2 | either side lacks coordinates, or the probe has no gene model |
| `gtf_gene_model` | versioned GENCODE gene id(s), comma-joined | the probe has no gene model |
| `gtf_gene_symbol` | GENCODE `gene_name`(s), comma-joined | the probe has no gene model |

`gtf_gene_model` and `gtf_gene_symbol` are properties of the **probe**, so they are populated on
every row whose probe resolved — including `TRANS` rows, where no gene model was consulted to
produce the label. The `region` column already records whether the model was used.

The GTF provenance from the map header is stamped into the Parquet key–value metadata under
`tecpg_pgm_*`, so the annotation source travels with the data.

Note that lift37 gene ids carry a mapping suffix (`ENSG00000156508.21_19`). This is written as it
appears in the GTF; consumers joining against standard Ensembl identifiers must strip both the
`_NN` suffix and the version.

### 5.2 Reading coverage from a run

```
grep '\[assignRegion\] coverage'            <pipeline log>
grep '\[assignRegion\] eCpgs Counts by Region' <pipeline log>
head -6 annot_<ds>/probe_gene_model.tsv
```

The two coverage lines are reported separately and mean different things: **coordinate coverage**
is the fraction of pairs with usable positions on both sides; **gene-annotation coverage** is the
fraction whose expression probe resolved to a gene model. The second is always the lower of the
two, and the gap is the population that gets `TRANS` or `NULL` but no fine label.

---

## 6. Comparison with Kennedy et al. 2018

Kennedy et al. (BMC Genomics 19:476) performed eQTM mapping on the same GTP and MESA cohorts and
the same array platforms. Their annotation method is adopted here wherever it is applicable, so
that the comparison is against a peer-reviewed procedure rather than choices invented for this
project.

| | Kennedy et al. 2018 | tecpg |
|---|---|---|
| Probe alignment source | Barbosa-Morais re-annotation (`illuminaHumanv3/v4.db`) | Re-Annotator + GEO + UCSC |
| Gene assignment | probe ∩ exon > 25 bp | **same** |
| Exon collapse | overlapping exons → maximal interval | **same** |
| Duplicate resolution | drop `KIAA`/`FLJ`/`LOC`/`*orf*` names and read-through transcripts | **same**, plus GENCODE biotype recorded but not applied |
| Multiple surviving genes | all retained | **same** |
| Gene span | largest interval formed by the overlapping transcripts | **same** (union of `gene` features) |
| Transcript collapse | min TSS, max TES | **same** (GENCODE `gene` feature) |
| Reference annotation | RefSeq and Ensembl, kept parallel | GENCODE v49lift37 (Ensembl-derived) |
| cis / distal boundary | 50 kb | **same** |
| Promoter window | 2,500 bp upstream of TSS | **same** |
| Region categories | cis, gene body, distal, trans | seven labels; ours subdivide theirs by 5′/3′ orientation |
| **Probe genomic location** | **selected per pair**, preferring the location that places the CpG inside or near the gene | **one fixed position per probe** |

### 6.1 The probe-location difference

This is the one substantive divergence and it is structural.

Kennedy report that 29.2% of GTP probes and 32.6% of MESA probes had more than one possible
genomic location. For each eCpG–transcript pair they compared all of that probe's locations
against the CpG and selected one by priority: the CpG inside the gene or within 2,500 bp upstream
of its TSS; failing that, the CpG within 1 Mb of the TSS; failing that, the same chromosome.
Ties were broken toward gene-annotated locations, then by re-annotation field precedence, then by
proximity to the TSS.

They quantify the effect: in GTP the cis fraction moved from 39.4% to 47.3% and trans from 49.4%
to 38.9%.

tecpg cannot reproduce this. `G.bed6` holds a single position per probe, and the cis/trans test
space is built from it, so a pair-dependent probe location would change which pairs are tested at
all — pair counts, FDR denominators, and the permutation universe. The alignment source also
differs and is less permissive: `ILMN_1343295`, which Kennedy place at three loci, is reported as
a unique alignment by the Re-Annotator.

**Consequence:** the tecpg trans fraction runs higher than Kennedy's, and the difference is a
method difference rather than a discrepancy in findings. Any concordance figure should be read
with this in mind.

### 6.2 Category mapping

tecpg labels roll up into Kennedy's categories without remainder:

| Kennedy | tecpg |
|---|---|
| cis | `PROMOTER`, `CIS5`, `CIS3` |
| gene body | `GENEBODY` |
| distal | `DISTAL5`, `DISTAL3` |
| trans | `TRANS` |

Kennedy's percentages are computed over their significant eCpG set. Comparisons should be made at
a matched significance threshold, not against the full test space.

---

## 7. Limitations

**One position per probe.** See §6.1. Probes with several genomic alignments are represented at a
single location, which is decided upstream in `generate_annotations.py` and shared by the mapping
test space.

**Union spans.** A probe resolving to several overlapping genes gets a span wider than any one of
them, so `PROMOTER` and `GENEBODY` are anchored to a merged boundary. `n_genes` in the map records
where this applies.

**No biotype filter.** All GENCODE `gene` features are eligible, so a probe may resolve to a
lncRNA that overlaps a protein-coding gene, and both enter the union. `gtf_gene_types` records
what was used.

**Unresolved probes.** Probes whose exons do not overlap any GENCODE gene by more than 25 bp
receive `TRANS` labels where applicable but no fine region label. This includes probes for
deprecated or predicted transcripts on an array designed against an older annotation.

**lift37 identifiers.** See §5.1 — ids carry a `_NN` mapping suffix that must be stripped before
joining against standard Ensembl identifiers.

**Alt contigs and junction-spanning probes.** Probes aligning only to alternate haplotype contigs,
or whose alignment is non-contiguous across a splice junction, have no single interval in
`G.bed6` and are excluded from both the map and the coordinate columns.

---

## 8. Key files

| Path | Role |
|---|---|
| `tools/build_probe_gene_model.py` | derives the probe→gene map from the GTF |
| `tools/annotation_io.py` | `readProbeGeneModel`, `readAnnotationFileToDict` |
| `tools/assignRegionToEcpg_parquet.py` | writes coordinates and region labels |
| `tools/generate_annotations.py` | builds `M.bed6` and `G.bed6` |
| `pipeline.sh` `[5/9]` | derives the map, then annotates |
| `pipelinePermute.sh` | gates on the map, then annotates the permutation master |
| `tests/test_build_probe_gene_model.py` | exon overlap, read-through handling, duplicate resolution |
| `tests/test_assign_region_split.py` | coordinate/region split, window arithmetic, TRANS on probe chromosome |
