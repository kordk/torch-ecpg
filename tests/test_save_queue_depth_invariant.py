"""
Regression test for the `save_queue_depth` shadowing bug in
`tecpg/processing.py:_tecpg_mlr_lstsq_inner`.

History: a profile-log helper inside the gene-chunk loop was assigned to a
variable named `save_queue_depth`, which shadowed the configured back-pressure
cap of the same name set up before the loop. After the loop's prune step
(`while futures and futures[0].done(): futures.popleft().result()`), the
shadowed value was almost always 0, so the back-pressure loop
`while len(futures) >= save_queue_depth: futures.popleft().result()` then
popped from an empty deque and crashed with
`IndexError: pop from an empty deque`.

This test uses a static AST check rather than a full pipeline run because the
crash only reproduces under TECPG_PROFILE=1 with real GPU work, which is not
available in CI. The invariant we assert is structural: inside
`_tecpg_mlr_lstsq_inner`, the name `save_queue_depth` is bound exactly once
(at the top of the function, from carry_data), and is never reassigned later.
The per-iteration profile fill metric uses a different name
(`save_queue_fill`).
"""
import ast
import inspect

from tecpg import processing


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in tecpg.processing")


def _assignment_targets(func: ast.FunctionDef):
    """Yield (lineno, target_name) for every Name target of an Assign in the
    function body."""
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    yield node.lineno, target.id
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            yield node.lineno, node.target.id


def test_save_queue_depth_cap_not_shadowed_in_loop():
    """The back-pressure cap variable must be assigned exactly once and not
    overwritten inside the gene-chunk loop."""
    source = inspect.getsource(processing)
    tree = ast.parse(source)
    func = _find_function(tree, '_tecpg_mlr_lstsq_inner')

    cap_assignments = [
        lineno for lineno, name in _assignment_targets(func)
        if name == 'save_queue_depth'
    ]
    assert len(cap_assignments) == 1, (
        f"`save_queue_depth` (back-pressure cap) must be assigned exactly "
        f"once in _tecpg_mlr_lstsq_inner, found {len(cap_assignments)} "
        f"assignment(s) at line(s) {cap_assignments}. A second assignment "
        f"shadows the cap and reintroduces the empty-deque bug."
    )


def test_save_queue_depth_cap_used_in_backpressure_loops():
    """The back-pressure `while len(futures) >= save_queue_depth:` loops must
    still exist and reference the cap variable (not a shadowed copy)."""
    source = inspect.getsource(processing)
    # At least two back-pressure loops (chunked + non-chunked save sites).
    occurrences = source.count('while len(futures) >= save_queue_depth')
    assert occurrences >= 2, (
        f"Expected >=2 back-pressure loops bounded by save_queue_depth in "
        f"tecpg/processing.py, found {occurrences}."
    )


def test_save_queue_fill_is_used_for_profile_metric():
    """The per-iteration profile log uses `save_queue_fill`, not the cap."""
    source = inspect.getsource(processing)
    assert 'save_queue_fill = len(futures)' in source, (
        "Expected `save_queue_fill = len(futures)` in _tecpg_mlr_lstsq_inner "
        "to expose the queue fill level for PROFILE logging without "
        "shadowing the back-pressure cap."
    )
    assert 'save_q={save_queue_fill}' in source, (
        "PROFILE log line should reference {save_queue_fill}, not the cap."
    )
