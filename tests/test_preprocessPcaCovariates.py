import os
import subprocess
import tempfile
import pandas as pd

def test_preprocess_pca():
    with tempfile.TemporaryDirectory() as tmpdir:
        g_file = os.path.join(tmpdir, "G.csv")
        c_file = os.path.join(tmpdir, "C.csv")
        out_file = os.path.join(tmpdir, "out.csv")

        with open(g_file, "w") as f:
            f.write(",sample1,sample2,sample3,sample4\n")
            f.write("gene1,1.0,2.0,3.0,4.0\n")
            f.write("gene2,2.0,3.0,4.0,\n")
            f.write("gene3,4.0,5.0,6.0,7.0\n")

        with open(c_file, "w") as f:
            f.write(",Age,Sex\n")
            f.write("sample1,44,1\n")
            f.write("sample2,50,1\n")
            f.write("sample3,52,0\n")
            f.write("sample4,56,1\n")

        result = subprocess.run(
            ["python3", "tools/preprocessPcaCovariates.py", "-g", g_file, "-c", c_file, "-o", out_file, "-n", "2"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert os.path.exists(out_file)

        df = pd.read_csv(out_file, index_col=0)
        assert df.shape == (4, 4)
        assert list(df.columns) == ["Age", "Sex", "PC1", "PC2"]

def test_preprocess_pca_transposed():
    with tempfile.TemporaryDirectory() as tmpdir:
        g_file = os.path.join(tmpdir, "G.csv")
        c_file = os.path.join(tmpdir, "C.csv")
        out_file = os.path.join(tmpdir, "out.csv")

        with open(g_file, "w") as f:
            f.write(",gene1,gene2,gene3\n")
            f.write("sample1,1.0,2.0,4.0\n")
            f.write("sample2,2.0,3.0,5.0\n")
            f.write("sample3,3.0,4.0,6.0\n")
            f.write("sample4,4.0,,7.0\n")

        with open(c_file, "w") as f:
            f.write(",Age,Sex\n")
            f.write("sample1,44,1\n")
            f.write("sample2,50,1\n")
            f.write("sample3,52,0\n")
            f.write("sample4,56,1\n")

        result = subprocess.run(
            ["python3", "tools/preprocessPcaCovariates.py", "-g", g_file, "-c", c_file, "-o", out_file, "-n", "2", "--transpose"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert os.path.exists(out_file)

        df = pd.read_csv(out_file, index_col=0)
        assert df.shape == (4, 4)
        assert list(df.columns) == ["Age", "Sex", "PC1", "PC2"]
