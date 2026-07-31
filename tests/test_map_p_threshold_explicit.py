"""Guards for D2: the mapper p-threshold is explicit in pipeline.sh.

`tecpg run mlr` gates the catalog at `-p` (default 0.001) before anything else
runs. pipeline.sh previously relied on that CLI default implicitly, so the
catalog's inclusion rule never appeared in the run log and could shift silently
if the CLI default changed. These tests pin the value, its wiring into the Stage
3 invocation, and its agreement with the CLI default.
"""
import ast
import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_SH = os.path.join(REPO_ROOT, "pipeline.sh")
CLI_PY = os.path.join(REPO_ROOT, "tecpg", "cli.py")


def _pipeline_text():
    with open(PIPELINE_SH, encoding="utf-8") as fh:
        return fh.read()


def _shell_var(name, text):
    """Return the value of a top-level `NAME="value"` assignment."""
    m = re.search(rf'^{name}="([^"]*)"\s*$', text, re.MULTILINE)
    assert m, f"{name} is not assigned in pipeline.sh"
    return m.group(1)


def _cli_mlr_p_default():
    """The -p/--p-thresh default on `run mlr`, read from source without importing torch.

    tecpg/cli.py declares -p twice: once on `mlr` and once on `mlr-single`
    (a different default). Walking to the FunctionDef named `mlr` disambiguates.
    """
    tree = ast.parse(open(CLI_PY, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "mlr":
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call)
                    and getattr(dec.func, "attr", None) == "option"):
                continue
            flags = [a.value for a in dec.args if isinstance(a, ast.Constant)]
            if "-p" not in flags and "--p-thresh" not in flags:
                continue
            for kw in dec.keywords:
                if kw.arg == "default":
                    return ast.literal_eval(kw.value)
    raise AssertionError("could not locate the -p default on the `run mlr` command")


def test_map_p_thresh_is_defined():
    """pipeline.sh must define the threshold as a named variable."""
    assert _shell_var("MAP_P_THRESH", _pipeline_text()) == "0.001"


def test_map_p_thresh_matches_cli_default():
    """Load-bearing guard: making the gate explicit must not change the gate.

    If the CLI default and the pipeline value ever diverge, the catalog silently
    changes size. This forces the divergence to be a deliberate, reviewed edit.
    """
    shell_value = float(_shell_var("MAP_P_THRESH", _pipeline_text()))
    assert shell_value == _cli_mlr_p_default(), (
        "pipeline.sh MAP_P_THRESH and the `run mlr` -p default disagree. "
        "Making the mapper gate explicit was meant to be behaviour-preserving; "
        "changing either value changes the catalog and every downstream count."
    )


def test_stage3_invocation_passes_the_threshold():
    """The qr mapping call must actually receive -p, not merely define it."""
    text = _pipeline_text()
    m = re.search(r"^.*run mlr --mlr-method qr .*$", text, re.MULTILINE)
    assert m, "could not find the Stage 3 `run mlr --mlr-method qr` invocation"
    invocation = m.group(0)
    assert '-p "$MAP_P_THRESH"' in invocation, (
        f"Stage 3 does not pass -p; invocation was:\n{invocation}"
    )


def test_threshold_is_logged():
    """The value must reach the run log so it is citable in methods."""
    text = _pipeline_text()
    log_lines = [ln for ln in text.splitlines()
                 if ln.strip().startswith("log ") and "MAP_P_THRESH" in ln]
    assert log_lines, "no log line emits MAP_P_THRESH"


def test_bootstrap_invocation_does_not_pass_the_threshold():
    """qr_bootstrap ignores p_thresh; passing it there would imply otherwise."""
    text = _pipeline_text()
    m = re.search(r"^.*run mlr --mlr-method qr_bootstrap .*$", text, re.MULTILINE)
    assert m, "could not find the Stage 9 `run mlr --mlr-method qr_bootstrap` invocation"
    assert "MAP_P_THRESH" not in m.group(0)


def test_pipeline_sh_is_valid_bash():
    """Quoting errors in the edited invocation must not reach a production run."""
    proc = subprocess.run(["bash", "-n", PIPELINE_SH],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"bash -n failed:\n{proc.stderr}"
