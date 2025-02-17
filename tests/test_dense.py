from pathlib import Path

import numpy as np
import pytest

from cellarr_array import DenseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_1d_array_creation(temp_dir):
    """Test creation of 1D dense array."""
    uri = str(Path(temp_dir) / "test_dense_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=False)

    assert isinstance(array, DenseCellArray)
    assert array.shape == (100,)
    assert array.ndim == 1
    assert array.dim_names == ["dim_0"]
    assert "data" in array.attr_names


def test_2d_array_creation(temp_dir):
    """Test creation of 2D dense array."""
    uri = str(Path(temp_dir) / "test_dense_2d")
    array = create_cellarray(uri=uri, shape=(100, 50), attr_dtype=np.float32, sparse=False, dim_names=["rows", "cols"])

    assert isinstance(array, DenseCellArray)
    assert array.shape == (100, 50)
    assert array.ndim == 2
    assert array.dim_names == ["rows", "cols"]
    assert "data" in array.attr_names


def test_1d_write_batch(sample_dense_array_1d):
    """Test batch writing to 1D dense array."""
    data = np.random.random(10).astype(np.float32)
    sample_dense_array_1d.write_batch(data, start_row=0)

    result = sample_dense_array_1d[0:10]
    np.testing.assert_array_almost_equal(result, data)


def test_2d_write_batch(sample_dense_array_2d):
    """Test batch writing to 2D dense array."""
    data = np.random.random((10, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

    result = sample_dense_array_2d[0:10, :]
    np.testing.assert_array_almost_equal(result, data)


def test_1d_bounds_check(sample_dense_array_1d):
    """Test bounds checking in 1D array."""
    data = np.random.random(150).astype(np.float32)
    with pytest.raises(ValueError, match="would exceed array bounds"):
        sample_dense_array_1d.write_batch(data, start_row=0)


def test_2d_bounds_check(sample_dense_array_2d):
    """Test bounds checking in 2D array."""
    # Test row bounds
    data = np.random.random((150, 50)).astype(np.float32)
    with pytest.raises(ValueError, match="would exceed array bounds"):
        sample_dense_array_2d.write_batch(data, start_row=0)

    # Test column bounds
    data = np.random.random((10, 60)).astype(np.float32)
    with pytest.raises(ValueError, match="Data columns"):
        sample_dense_array_2d.write_batch(data, start_row=0)


def test_1d_slicing(sample_dense_array_1d):
    """Test slicing operations on 1D array."""
    data = np.random.random(100).astype(np.float32)
    sample_dense_array_1d.write_batch(data, start_row=0)

    # Full slice
    result = sample_dense_array_1d[:]
    np.testing.assert_array_almost_equal(result, data)

    # Partial slice
    result = sample_dense_array_1d[10:20]
    np.testing.assert_array_almost_equal(result, data[10:20])

    # Single index
    result = sample_dense_array_1d[5]
    np.testing.assert_array_almost_equal(result, data[5])

    # Negative indices
    result = sample_dense_array_1d[-10:]
    np.testing.assert_array_almost_equal(result, data[-10:])


def test_2d_slicing(sample_dense_array_2d):
    """Test slicing operations on 2D array."""
    data = np.random.random((100, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

    # Full slice
    result = sample_dense_array_2d[:]
    np.testing.assert_array_almost_equal(result, data)

    # Partial slice
    result = sample_dense_array_2d[10:20, 5:15]
    np.testing.assert_array_almost_equal(result, data[10:20, 5:15])

    # Single row
    result = sample_dense_array_2d[5]
    np.testing.assert_array_almost_equal(result.flatten(), data[5])

    # Negative indices
    result = sample_dense_array_2d[-10:, -5:]
    np.testing.assert_array_almost_equal(result, data[-10:, -5:])


def test_multi_index_access(sample_dense_array_2d):
    """Test multi-index access patterns."""
    data = np.random.random((100, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

    # List indices
    rows = [1, 3, 5]
    cols = [2, 4, 6]
    result = sample_dense_array_2d[rows, cols]
    expected = data[rows][:, cols]
    np.testing.assert_array_almost_equal(result, expected)

    # Mixed slice and list
    result = sample_dense_array_2d[10:20, cols]
    expected = data[10:20][:, cols]
    np.testing.assert_array_almost_equal(result, expected)


def test_mixed_slice_list_bounds(sample_dense_array_2d):
    """Test boundary handling in mixed slice-list queries."""
    data = np.random.random((100, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

    # Test mixed slice and list with various bounds
    cols = [2, 4, 6]

    # Simple slice
    result = sample_dense_array_2d[10:20, cols]
    expected = data[10:20][:, cols]
    np.testing.assert_array_almost_equal(result, expected)

    # Slice at array bounds
    result = sample_dense_array_2d[90:100, cols]
    expected = data[90:100][:, cols]
    np.testing.assert_array_almost_equal(result, expected)

    # Slice with step
    with pytest.raises(Exception):
        # stepped slicer are not supported by multi_index
        result = sample_dense_array_2d[10:20:2, cols]

    # Reversed indices
    cols_reversed = cols[::-1]
    result = sample_dense_array_2d[10:20, cols_reversed]
    expected = data[10:20][:, cols_reversed]
    np.testing.assert_array_almost_equal(result, expected)


def test_invalid_operations(sample_dense_array_2d):
    """Test invalid operations raise appropriate errors."""
    # Invalid mode
    with pytest.raises(ValueError, match="Mode must be one of"):
        sample_dense_array_2d.mode = "invalid"

    # Invalid attribute
    with pytest.raises(ValueError, match="Attribute .* does not exist"):
        DenseCellArray(sample_dense_array_2d.uri, attr="invalid_attr")

    # Invalid dimensions in slice
    with pytest.raises(IndexError, match="Invalid number of dimensions"):
        _ = sample_dense_array_2d[0:10, 0:10, 0:10]

    # Out of bounds slice
    with pytest.raises(IndexError, match="out of bounds"):
        _ = sample_dense_array_2d[200:300]
