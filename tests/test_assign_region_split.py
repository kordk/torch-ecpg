import os
import sys
import subprocess
import pytest
import pandas as pd
import pyarrow.parquet as pq

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "assignRegionToEcpg_parquet.py")

# probe BED: gt probes are ~50bp footprints. GENE1 probe sits inside a big gene.
G_BED = """\
1\t100000\t100049\tILMN_A\t0\t+
1\t100000\t100049\tILMN_B\t0\t+
2\t500000\t500049\tILMN_C\t0\t+
1\t100000\t100049\tILMN_NOMODEL\t0\t+
NA\tNA\tNA\tILMN_NOCOORD\t0\t+
"""

# probe->gene map: ILMN_A spans 90000-150000 (+). ILMN_B spans 90000-150000 (-).
# ILMN_C on chr2. ILMN_NOMODEL is NO_OVERLAP -> absent from the dict.
PGM = """\
#tecpg_probe_gene_model\tv1
#method\texon_overlap_gt_25bp_union_span
#gtf\t/fake/gencode.v49lift37.annotation.gtf.gz
#counts\tRESOLVED=3 NO_OVERLAP=1 MULTI_CHROM=0
probe_id\tstatus\tchrom\tstart\tend\tstrand\tn_genes\tgtf_gene_ids\tgtf_gene_symbols\tgtf_gene_types
ILMN_A\tRESOLVED\t1\t90000\t150000\t+\t1\tENSG_A.1\tAAA\tprotein_coding
ILMN_B\tRESOLVED\t1\t90000\t150000\t-\t2\tENSG_B.1,ENSG_B2.1\tBBB,BB2\tprotein_coding,lncRNA
ILMN_C\tRESOLVED\t2\t490000\t520000\t+\t1\tENSG_C.1\tCCC\tprotein_coding
ILMN_NOMODEL\tNO_OVERLAP\t\t\t\t\t0\t\t\t
"""

# CpG positions chosen against the GENE span (90000-150000), not the probe (100000).
M_BED = """\
1\t120000\t120001\tcg_body\t0\t+
1\t91000\t91001\tcg_prom\t0\t+
1\t30000\t30001\tcg_distal5\t0\t+
2\t495000\t495001\tcg_chr2\t0\t+
1\t120000\t120001\tcg_forNoModel\t0\t+
"""

PAIRS = [
    ("ILMN_A", "cg_body",     "GENEBODY"),   # inside the GENE span; would be DISTAL under probe span
    ("ILMN_A", "cg_prom",     "PROMOTER"),   # near gene TSS 90000
    ("ILMN_A", "cg_distal5",  "DISTAL5"),
    ("ILMN_B", "cg_body",     "GENEBODY"),   # negative strand
    ("ILMN_C", "cg_body",     "TRANS"),      # chr2 gene vs chr1 CpG
    ("ILMN_A", "cg_chr2",     "TRANS"),
    ("ILMN_NOMODEL", "cg_forNoModel", None), # coords present, no gene model
    ("ILMN_NOCOORD", "cg_body", None),       # gt probe has no coordinates
    ("ILMN_NOMODEL", "cg_chr2", "TRANS"),    # unresolved probe must STILL get TRANS
]

@pytest.fixture(scope="module")
def run_result(tmp_path_factory):
    d = tmp_path_factory.mktemp("assign_region_split")

    with open(os.path.join(d, "G.bed6"), "w") as f:
        f.write(G_BED)
    with open(os.path.join(d, "M.bed6"), "w") as f:
        f.write(M_BED)
    with open(os.path.join(d, "pgm.tsv"), "w") as f:
        f.write(PGM)

    df = pd.DataFrame({
        "gt_id": [p[0] for p in PAIRS],
        "mt_id": [p[1] for p in PAIRS],
        "mt_p":  [1e-6] * len(PAIRS),
        "mt_est": [0.1] * len(PAIRS),
    })
    src = os.path.join(d, "in.parquet")
    df.to_parquet(src, index=False)

    out = os.path.join(d, "out.parquet")
    r = subprocess.run([sys.executable, TOOL, "-d", src,
                        "-g", os.path.join(d, "G.bed6"),
                        "--gene-model", os.path.join(d, "pgm.tsv"),
                        "-m", os.path.join(d, "M.bed6"),
                        "-o", out], capture_output=True, text=True)

    if r.returncode != 0:
        raise SystemExit(f"tool failed\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")

    t = pq.read_table(out)
    res_df = t.to_pandas()
    key = {(row.gt_id, row.mt_id): row for row in res_df.itertuples()}

    return key, t.schema, r.stderr + r.stdout

def test_schema_has_new_fields(run_result):
    key, schema, log = run_result
    fields = schema.names
    assert "gtf_gene_model" in fields and "gtf_gene_symbol" in fields, "schema has 9 new fields incl. gtf_*"

def test_genebody_measured_against_gene_span(run_result):
    key, schema, log = run_result
    r = key[("ILMN_A", "cg_body")]
    assert r.region == "GENEBODY", f"GENEBODY measured against the GENE span, not the probe footprint   {r.region}"

def test_promoter_anchored_at_tss(run_result):
    key, schema, log = run_result
    r = key[("ILMN_A", "cg_prom")]
    assert r.region == "PROMOTER", f"PROMOTER anchored at gene TSS   {r.region}"

def test_distal5_beyond_50kb_of_gene_start(run_result):
    key, schema, log = run_result
    r = key[("ILMN_A", "cg_distal5")]
    assert r.region == "DISTAL5", f"DISTAL5 beyond 50kb of gene start   {r.region}"

def test_negative_strand_genebody(run_result):
    key, schema, log = run_result
    r = key[("ILMN_B", "cg_body")]
    assert r.region == "GENEBODY", f"negative-strand GENEBODY   {r.region}"

def test_gt_chromstart_is_probe_position(run_result):
    key, schema, log = run_result
    r = key[("ILMN_A", "cg_body")]
    assert r.gt_chromStart == 100000, f"gt_chromStart is the PROBE position, not the gene start   {r.gt_chromStart}"

def test_gtf_gene_model_carries_versioned_id(run_result):
    key, schema, log = run_result
    r = key[("ILMN_A", "cg_body")]
    assert r.gtf_gene_model == "ENSG_A.1" and r.gtf_gene_symbol == "AAA", f"gtf_gene_model carries the versioned GENCODE id   {r.gtf_gene_model}/{r.gtf_gene_symbol}"

def test_union_of_two_genes_comma_joined(run_result):
    key, schema, log = run_result
    r = key[("ILMN_B", "cg_body")]
    assert r.gtf_gene_model == "ENSG_B.1,ENSG_B2.1" and r.gtf_gene_symbol == "BBB,BB2", f"union of two genes -> comma-joined ids and symbols   {r.gtf_gene_model}/{r.gtf_gene_symbol}"

def test_trans_decided_on_probe_chromosome(run_result):
    key, schema, log = run_result
    r = key[("ILMN_C", "cg_body")]
    assert r.region == "TRANS", f"TRANS decided on PROBE chromosome   {r.region}"

def test_gtf_populated_on_trans_rows(run_result):
    key, schema, log = run_result
    r = key[("ILMN_C", "cg_body")]
    assert r.gtf_gene_model == "ENSG_C.1", f"gtf_* populated on TRANS rows   {r.gtf_gene_model}"

def test_no_gene_model_region_null_coords_intact(run_result):
    key, schema, log = run_result
    r = key[("ILMN_NOMODEL", "cg_forNoModel")]
    assert (r.region is None or pd.isna(r.region)) and r.gt_chromStart == 100000 and r.mt_chromStart == 120000, f"no gene model -> region NULL but coordinates INTACT   region={r.region} gt={r.gt_chromStart} mt={r.mt_chromStart}"

def test_no_gene_model_gtf_null(run_result):
    key, schema, log = run_result
    r = key[("ILMN_NOMODEL", "cg_forNoModel")]
    assert (r.gtf_gene_model is None or pd.isna(r.gtf_gene_model)), f"no gene model -> gtf_* NULL   {r.gtf_gene_model}"

def test_no_coord_probe_gt_null_mt_written(run_result):
    key, schema, log = run_result
    r = key[("ILMN_NOCOORD", "cg_body")]
    assert pd.isna(r.gt_chromStart) and r.mt_chromStart == 120000, f"gt probe absent from BED -> gt_* NULL but mt_* still written   gt={r.gt_chromStart} mt={r.mt_chromStart}"

def test_unresolved_probe_still_gets_trans(run_result):
    key, schema, log = run_result
    r = key[("ILMN_NOMODEL", "cg_chr2")]
    assert r.region == "TRANS" and r.gt_chromStart == 100000, f"unresolved probe still gets TRANS (no gene model needed)   region={r.region} gt={r.gt_chromStart}"

def test_gtf_provenance_stamped_into_metadata(run_result):
    key, schema, log = run_result
    md = schema.metadata or {}
    k_decoded = [k.decode() for k in md if k.startswith(b"tecpg_pgm_")]
    assert any(k.startswith(b"tecpg_pgm_") for k in md), f"GTF provenance stamped into Parquet metadata   {k_decoded}"

def test_coverage_lines_logged(run_result):
    key, schema, log = run_result
    assert "coverage: coordinate" in log and "coverage: gene-annotation" in log, "coverage lines logged"
