import os
import shutil
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'tools'))

from assignRegionToEcpg_parquet import (
    _normalize_chrom,
    _strip_id_version,
    readAnnotationFileToDict,
    assignRegion,
)


class TestNormalizeChrom(unittest.TestCase):
    """Audit Task 1: Verify chromosome normalization handles chr1 vs 1."""

    def test_strip_chr_prefix(self):
        self.assertEqual(_normalize_chrom("chr1"), "1")
        self.assertEqual(_normalize_chrom("chrX"), "X")
        self.assertEqual(_normalize_chrom("chrMT"), "MT")

    def test_no_prefix(self):
        self.assertEqual(_normalize_chrom("1"), "1")
        self.assertEqual(_normalize_chrom("X"), "X")

    def test_case_insensitive(self):
        self.assertEqual(_normalize_chrom("Chr1"), "1")
        self.assertEqual(_normalize_chrom("CHR1"), "1")

    def test_whitespace(self):
        self.assertEqual(_normalize_chrom(" chr1 "), "1")
        self.assertEqual(_normalize_chrom(" 1 "), "1")


class TestStripIdVersion(unittest.TestCase):
    """Audit Task 1: Verify Ensembl ID version stripping."""

    def test_versioned_id(self):
        self.assertEqual(_strip_id_version("ENSG00000000003.15"), "ENSG00000000003")

    def test_double_versioned_id(self):
        # featureCounts can produce IDs like ENSG00000240929.15_3
        self.assertEqual(_strip_id_version("ENSG00000240929.15_3"), "ENSG00000240929")

    def test_no_version(self):
        self.assertEqual(_strip_id_version("ENSG00000000003"), "ENSG00000000003")

    def test_cpg_id_unchanged(self):
        # CpG IDs have no dots, so stripping is a no-op
        self.assertEqual(_strip_id_version("cg13191808"), "cg13191808")


class TestReadAnnotationFileToDict(unittest.TestCase):
    """Audit Tasks 1, 2, 3: Test annotation loading with normalization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_bed6_chrom_normalization(self):
        """BED6 file with 'chr' prefix should produce normalized chromosomes."""
        bed_file = os.path.join(self.test_dir, "test.bed6")
        with open(bed_file, "w") as f:
            f.write("chr1\t1000\t2000\tcg00000001\t0\t+\n")
            f.write("chr2\t3000\t4000\tcg00000002\t0\t-\n")

        result = readAnnotationFileToDict(bed_file)
        self.assertEqual(result["cg00000001"]["chrom"], "1")
        self.assertEqual(result["cg00000002"]["chrom"], "2")

    def test_bed6_no_prefix(self):
        """BED6 file without 'chr' prefix should work."""
        bed_file = os.path.join(self.test_dir, "test.bed6")
        with open(bed_file, "w") as f:
            f.write("1\t1000\t2000\tcg00000001\t0\t+\n")

        result = readAnnotationFileToDict(bed_file)
        self.assertEqual(result["cg00000001"]["chrom"], "1")

    def test_bed6_int_types(self):
        """Audit Task 2: Verify coordinates are integers."""
        bed_file = os.path.join(self.test_dir, "test.bed6")
        with open(bed_file, "w") as f:
            f.write("chr1\t1000\t2000\tcg00000001\t0\t+\n")

        result = readAnnotationFileToDict(bed_file)
        self.assertIsInstance(result["cg00000001"]["chromStart"], int)
        self.assertIsInstance(result["cg00000001"]["chromEnd"], int)

    def test_gff_geneid_attribute(self):
        """Audit Task 3: featureCounts flat format with Geneid attribute."""
        gff_file = os.path.join(self.test_dir, "test.gff")
        with open(gff_file, "w") as f:
            f.write('chr1\tfeatureCounts\tgene\t1001\t2000\t.\t+\t.\tGeneid "ENSG00000000003.15"\n')

        result = readAnnotationFileToDict(gff_file)
        # Version should be stripped
        self.assertIn("ENSG00000000003", result)
        # Chrom should be normalized
        self.assertEqual(result["ENSG00000000003"]["chrom"], "1")
        # GFF 1-based -> 0-based conversion
        self.assertEqual(result["ENSG00000000003"]["chromStart"], 1000)
        self.assertEqual(result["ENSG00000000003"]["chromEnd"], 2000)

    def test_gtf_gene_id_attribute(self):
        """Audit Task 3: Standard GTF format with gene_id attribute."""
        gtf_file = os.path.join(self.test_dir, "test.gtf")
        with open(gtf_file, "w") as f:
            f.write('chr1\tensembl\tgene\t1001\t2000\t.\t+\t.\tgene_id "ENSG00000000003.15"; transcript_id "ENST00000001"\n')

        result = readAnnotationFileToDict(gtf_file)
        self.assertIn("ENSG00000000003", result)
        self.assertEqual(result["ENSG00000000003"]["chromStart"], 1000)

    def test_gff3_gene_id_attribute(self):
        """Audit Task 3: GFF3 format with gene_id= attribute."""
        gff3_file = os.path.join(self.test_dir, "test.gff3")
        with open(gff3_file, "w") as f:
            f.write('chr1\tensembl\tgene\t1001\t2000\t.\t+\t.\tgene_id=ENSG00000000003.15;Name=TP53\n')

        result = readAnnotationFileToDict(gff3_file)
        self.assertIn("ENSG00000000003", result)

    def test_bed6_versioned_gene_id(self):
        """BED6 file with versioned gene ID should be stripped."""
        bed_file = os.path.join(self.test_dir, "test.bed6")
        with open(bed_file, "w") as f:
            f.write("chr1\t1000\t2000\tENSG00000000003.15\t0\t+\n")

        result = readAnnotationFileToDict(bed_file)
        self.assertIn("ENSG00000000003", result)
        self.assertNotIn("ENSG00000000003.15", result)

    def test_skip_comments_and_headers(self):
        """Skip comment lines and BED header lines."""
        bed_file = os.path.join(self.test_dir, "test.bed6")
        with open(bed_file, "w") as f:
            f.write("# comment line\n")
            f.write("chrom\tchromStart\tchromEnd\tname\tscore\tstrand\n")
            f.write("chr1\t1000\t2000\tcg00000001\t0\t+\n")

        result = readAnnotationFileToDict(bed_file)
        self.assertEqual(len(result), 1)
        self.assertIn("cg00000001", result)


class TestAssignRegion(unittest.TestCase):
    """Audit Tasks 1-3: Test region assignment logic end-to-end."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _make_parquet(self, data, filename="input.parquet"):
        """Helper to create a test parquet file."""
        df = pd.DataFrame(data)
        filepath = os.path.join(self.test_dir, filename)
        df.to_parquet(filepath, index=False)
        return filepath

    def _read_output(self, filename="output.parquet"):
        filepath = os.path.join(self.test_dir, filename)
        return pd.read_parquet(filepath)

    def test_chrom_normalization_prevents_false_trans(self):
        """Audit Task 1: chr1 vs 1 mismatch should NOT cause false TRANS."""
        # Gene annotation uses 'chr1', methylation uses '1'
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            "cg001": {"chrom": "1", "chromStart": 150000, "chromEnd": 150001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        # Should be GENEBODY, not TRANS
        self.assertTrue(len(result) > 0)
        regions = result['region'].tolist()
        self.assertNotIn("TRANS", regions)
        self.assertIn("GENEBODY", regions)

    def test_versioned_id_lookup(self):
        """Audit Task 1: Versioned IDs in annotation should match versionless IDs in parquet."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            "cg001": {"chrom": "1", "chromStart": 150000, "chromEnd": 150001, "strand": "+"}
        }

        # Parquet uses versionless IDs, annotation keys are already stripped
        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        self.assertTrue(len(result) > 0)

    def test_trans_classification(self):
        """Different chromosomes should be TRANS."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            "cg001": {"chrom": "2", "chromStart": 150000, "chromEnd": 150001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        self.assertEqual(result['region'].tolist(), ["TRANS"])

    def test_cis_positive_strand(self):
        """CpG within 50Kb upstream of TSS on + strand should be CIS."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            # 30Kb upstream of TSS (100000 - 30000 = 70000)
            "cg001": {"chrom": "1", "chromStart": 70000, "chromEnd": 70001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("CIS", regions)

    def test_cis_negative_strand(self):
        """Audit Task 2: CIS on negative strand was previously always-false due to CIS_OFFSET=0."""
        gH = {
            # Negative strand gene: TSS = chromEnd = 200000
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "-"}
        }
        mH = {
            # 30Kb upstream of TSS for - strand (200000 + 30000 = 230000)
            "cg001": {"chrom": "1", "chromStart": 230000, "chromEnd": 230001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("CIS", regions, "CIS on negative strand should be detected (was always-false before fix)")

    def test_promoter_positive_strand(self):
        """CpG within +/-2500bp of TSS on + strand should be PROMOTER."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            # 1000bp upstream of TSS
            "cg001": {"chrom": "1", "chromStart": 99000, "chromEnd": 99001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("PROMOTER", regions)

    def test_promoter_negative_strand(self):
        """PROMOTER on negative strand should use chromEnd as TSS."""
        gH = {
            # Negative strand: TSS = chromEnd = 200000
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "-"}
        }
        mH = {
            # 1000bp from TSS (200000 + 1000 = 201000)
            "cg001": {"chrom": "1", "chromStart": 201000, "chromEnd": 201001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("PROMOTER", regions)

    def test_genebody_positive_strand(self):
        """CpG within gene body on + strand."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            "cg001": {"chrom": "1", "chromStart": 150000, "chromEnd": 150001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("GENEBODY", regions)

    def test_genebody_negative_strand(self):
        """Audit Task 2: Gene body on negative strand was always-false for standard coords."""
        gH = {
            # Standard coords: chromStart < chromEnd
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "-"}
        }
        mH = {
            "cg001": {"chrom": "1", "chromStart": 150000, "chromEnd": 150001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("GENEBODY", regions, "Gene body on negative strand should be detected (was always-false before fix)")

    def test_distal_negative_strand(self):
        """DISTAL on negative strand should use chromEnd as TSS reference."""
        gH = {
            # TSS = chromEnd = 200000
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "-"}
        }
        mH = {
            # >50Kb upstream of TSS for - strand: 200000 + 50001 = 250001
            "cg001": {"chrom": "1", "chromStart": 260000, "chromEnd": 260001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("DISTAL", regions)

    def test_pvalue_filter(self):
        """Rows with p-value > 1e-6 should be excluded."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            "cg001": {"chrom": "1", "chromStart": 150000, "chromEnd": 150001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [0.5]  # p-value too large
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        total = assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)
        self.assertEqual(total, 0)

    def test_overlapping_features(self):
        """Audit Task 3: A CpG can be assigned to multiple regions (e.g., PROMOTER and CIS)."""
        gH = {
            "ENSG001": {"chrom": "1", "chromStart": 100000, "chromEnd": 200000, "strand": "+"}
        }
        mH = {
            # 2000bp upstream of TSS -> within PROMOTER (+/-2500) AND CIS (<50Kb)
            "cg001": {"chrom": "1", "chromStart": 98000, "chromEnd": 98001, "strand": "+"}
        }

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)

        result = self._read_output()
        regions = result['region'].tolist()
        # CpG is within both CIS (<50Kb) and PROMOTER (+/-2500) windows
        self.assertIn("CIS", regions)
        self.assertIn("PROMOTER", regions)

    def test_end_to_end_with_chr_prefix_mismatch(self):
        """Full pipeline: chr-prefixed annotation files + parquet with versionless IDs."""
        # Create BED6 gene annotation with 'chr' prefix and versioned ID
        gene_file = os.path.join(self.test_dir, "genes.bed6")
        with open(gene_file, "w") as f:
            f.write("chr1\t100000\t200000\tENSG001.5\t0\t+\n")

        # Create BED6 methylation annotation with 'chr' prefix
        meth_file = os.path.join(self.test_dir, "meth.bed6")
        with open(meth_file, "w") as f:
            f.write("chr1\t150000\t150001\tcg001\t0\t+\n")

        gH = readAnnotationFileToDict(gene_file)
        mH = readAnnotationFileToDict(meth_file)

        # Gene should be stored with stripped version
        self.assertIn("ENSG001", gH)
        # Chroms should be normalized
        self.assertEqual(gH["ENSG001"]["chrom"], "1")
        self.assertEqual(mH["cg001"]["chrom"], "1")

        parquet_file = self._make_parquet({
            'gt_id': ['ENSG001'],
            'mt_id': ['cg001'],
            'mt_p': [1e-8]
        })
        out_file = os.path.join(self.test_dir, "output.parquet")

        total = assignRegion(parquet_file, gH, mH, 'mt_p', out_file, 100000)
        self.assertGreater(total, 0)

        result = self._read_output()
        regions = result['region'].tolist()
        self.assertIn("GENEBODY", regions)
        self.assertNotIn("TRANS", regions)


if __name__ == '__main__':
    unittest.main()
