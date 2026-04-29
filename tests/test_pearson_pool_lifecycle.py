import ast
from pathlib import Path
import unittest


def _find_function(module, name):
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


class PearsonPoolLifecycleTests(unittest.TestCase):
    def test_chunk_save_inner_does_not_shutdown_pool_inside_loop(self):
        source = Path("tecpg/pearson_full.py").read_text()
        module = ast.parse(source)
        function = _find_function(module, "_pearson_chunk_save_tensor_inner")

        for node in ast.walk(function):
            if not isinstance(node, ast.For):
                continue

            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "shutdown"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "pool"
                ):
                    self.fail("pool.shutdown() must not run inside the chunk loop")


if __name__ == "__main__":
    unittest.main()
