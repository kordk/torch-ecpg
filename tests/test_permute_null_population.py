import pandas as pd
import numpy as np
import pytest
from tecpg import permute

# Using a mock logger since _select_null_population expects one
class MockLogger:
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass

@pytest.fixture
def mock_logger():
    return MockLogger()

@pytest.fixture
def small_M():
    return pd.DataFrame(index=['cg1', 'cg2'])

@pytest.fixture
def small_G():
    return pd.DataFrame(index=['geneA', 'geneB'])

@pytest.fixture
def large_M(monkeypatch):
    size = 1000
    df = pd.DataFrame(index=[f'cg{i}' for i in range(size)])
    # Set ceiling lower to avoid creating a real 50M frame
    monkeypatch.setattr(permute, 'MAX_NULL_PAIR_PRODUCT', 500)
    return df

@pytest.fixture
def large_G():
    size = 1000
    return pd.DataFrame(index=[f'gene{i}' for i in range(size)])


def test_small_unsubsampled_returns_full_matrices(small_M, small_G, mock_logger):
    # Product is 4, well below ceiling
    null_M, null_G = permute._select_null_population(
        small_M, small_G, None, None, None, 'all', None, None, None,
        None, None, 42, mock_logger
    )
    assert null_M is small_M
    assert null_G is small_G

def test_oversized_unsubsampled_raises(large_M, large_G, mock_logger):
    # Ceiling patched to 500. M=1000, G=1000, product=1,000,000 > 500
    with pytest.raises(ValueError) as excinfo:
        permute._select_null_population(
            large_M, large_G, None, None, None, 'all', None, None, None,
            None, None, 42, mock_logger
        )
    assert 'Host-memory ceiling exceeded' in str(excinfo.value)

def test_oversized_message_names_both_flags(large_M, large_G, mock_logger):
    with pytest.raises(ValueError) as excinfo:
        permute._select_null_population(
            large_M, large_G, None, None, None, 'all', None, None, None,
            None, None, 42, mock_logger
        )
    msg = str(excinfo.value)
    assert '--subsample-mt-count' in msg
    assert '--subsample-g-count' in msg

def test_oversized_message_names_dimensions_and_ceiling(large_M, large_G, mock_logger):
    with pytest.raises(ValueError) as excinfo:
        permute._select_null_population(
            large_M, large_G, None, None, None, 'all', None, None, None,
            None, None, 42, mock_logger
        )
    msg = str(excinfo.value)
    assert str(len(large_M)) in msg
    assert str(len(large_G)) in msg
    product = len(large_M) * len(large_G)
    assert str(product) in msg
    assert str(permute.MAX_NULL_PAIR_PRODUCT) in msg

def test_explicit_counts_below_ceiling_subsample(large_M, large_G, mock_logger):
    # Explicit subsample to 20x20 = 400 < 500 (ceiling)
    null_M, null_G = permute._select_null_population(
        large_M, large_G, None, None, None, 'all', None, None, None,
        20, 20, 42, mock_logger
    )
    assert len(null_M) == 20
    assert len(null_G) == 20

def test_explicit_counts_above_ceiling_raises(large_M, large_G, mock_logger):
    # Explicit subsample to 30x30 = 900 > 500 (ceiling)
    with pytest.raises(ValueError) as excinfo:
        permute._select_null_population(
            large_M, large_G, None, None, None, 'all', None, None, None,
            30, 30, 42, mock_logger
        )
    assert 'Host-memory ceiling exceeded' in str(excinfo.value)

def test_product_exactly_at_ceiling_is_allowed(large_M, large_G, mock_logger):
    # Ceiling is 500. Explicitly set to 25x20 = 500
    null_M, null_G = permute._select_null_population(
        large_M, large_G, None, None, None, 'all', None, None, None,
        25, 20, 42, mock_logger
    )
    assert len(null_M) == 25
    assert len(null_G) == 20

def test_seeded_subsample_is_reproducible(large_M, large_G, mock_logger):
    # Below ceiling subsample 10x10 = 100 < 500
    null_M_1, null_G_1 = permute._select_null_population(
        large_M, large_G, None, None, None, 'all', None, None, None,
        10, 10, 42, mock_logger
    )

    null_M_2, null_G_2 = permute._select_null_population(
        large_M, large_G, None, None, None, 'all', None, None, None,
        10, 10, 42, mock_logger
    )

    # Different seed
    null_M_3, null_G_3 = permute._select_null_population(
        large_M, large_G, None, None, None, 'all', None, None, None,
        10, 10, 99, mock_logger
    )

    pd.testing.assert_frame_equal(null_M_1, null_M_2)
    pd.testing.assert_frame_equal(null_G_1, null_G_2)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(null_M_1, null_M_3)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(null_G_1, null_G_3)
