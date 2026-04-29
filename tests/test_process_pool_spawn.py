import multiprocessing
import os
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def _save_dataframe_for_spawn_smoke(dataframe, path):
    dataframe.to_csv(path, index=False)
    return os.path.getsize(path)


class ProcessPoolSpawnTests(unittest.TestCase):
    def test_spawn_pool_can_save_dataframe(self):
        context = multiprocessing.get_context("spawn")
        dataframe = pd.DataFrame({"a": [1, 2]})

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "out.csv")
            with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
                list(pool.map(int, range(2)))
                result = pool.submit(
                    _save_dataframe_for_spawn_smoke,
                    dataframe,
                    output_path,
                ).result(timeout=10)

            self.assertGreater(result, 0)
            self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
