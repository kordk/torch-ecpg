import sys
import unittest
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['colorama'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['captum'] = MagicMock()
sys.modules['captum.attr'] = MagicMock()
sys.modules['mygene'] = MagicMock()
sys.modules['requests'] = MagicMock()

try:
    from tecpg.regression_full import regression_full
    from tecpg.pearson_full import pearson_chunk_save_tensor
    from tecpg.processing import tecpg_mlr_qr
    from tecpg.regression_single import regression_single
    print("Imports passed with mocks.")
except Exception as e:
    print(f"Error during import: {e}")
