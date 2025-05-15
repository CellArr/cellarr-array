from pathlib import Path

import numpy as np
import pytest
import tiledb

from cellarr_array import DenseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_1d_array_creation(temp_dir):
    uri = str(Path(temp_dir) / "test_dense_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=False)

    assert isinstance(array, DenseCellArray)
    assert array.shape == (100,)
    assert array.ndim == 1
    assert array.dim_names == ["dim_0"]
    assert "data" in array.attr_names


def test_2d_array_creation(temp_dir):
    uri = str(Path(temp_dir) / "test_dense_2d")
    array = create_cellarray(uri=uri, shape=(100, 50), attr_dtype=np.float32, sparse=False, dim_names=["rows", "cols"])

    assert isinstance(array, DenseCellArray)
    assert array.shape == (100, 50)
    assert array.ndim == 2
    assert array.dim_names == ["rows", "cols"]
    assert "data" in array.attr_names


def test_1d_write_batch(sample_dense_array_1d):
    data = np.random.random(10).astype(np.float32)
    sample_dense_array_1d.write_batch(data, start_row=0)

    result = sample_dense_array_1d[0:10]
    np.testing.assert_array_almost_equal(result, data)


def test_2d_write_batch(sample_dense_array_2d):
    data = np.random.random((10, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

    result = sample_dense_array_2d[0:10, :]
    np.testing.assert_array_almost_equal(result, data)


def test_1d_bounds_check(sample_dense_array_1d):
    data = np.random.random(150).astype(np.float32)
    with pytest.raises(ValueError, match="would exceed array bounds"):
        sample_dense_array_1d.write_batch(data, start_row=0)


def test_2d_bounds_check(sample_dense_array_2d):
    data = np.random.random((150, 50)).astype(np.float32)
    with pytest.raises(ValueError, match="would exceed array bounds"):
        sample_dense_array_2d.write_batch(data, start_row=0)

    data = np.random.random((10, 60)).astype(np.float32)
    with pytest.raises(ValueError, match="Data columns"):
        sample_dense_array_2d.write_batch(data, start_row=0)


def test_1d_slicing(sample_dense_array_1d):
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

    # Ellipsis
    result = sample_dense_array_1d[...]
    actual = data[...]
    np.testing.assert_array_almost_equal(result, actual), f"{actual} != {result}"


def test_2d_slicing(sample_dense_array_2d):
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

    # Ellipsis
    result = sample_dense_array_2d[..., :1]
    np.testing.assert_array_almost_equal(result, data[..., :1])
    result = sample_dense_array_2d[..., :]
    np.testing.assert_array_almost_equal(result, data[..., :])
    result = sample_dense_array_2d[-1:, ...]
    np.testing.assert_array_almost_equal(result, data[-1:, ...])
    with pytest.raises(IndexError):
        _ = sample_dense_array_2d[..., ...]
    with pytest.raises(IndexError):
        _ = sample_dense_array_2d[[0, 3], ...]


def test_multi_index_access(sample_dense_array_2d):
    data = np.random.random((100, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

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
    data = np.random.random((100, 50)).astype(np.float32)
    sample_dense_array_2d.write_batch(data, start_row=0)

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
    with pytest.raises(ValueError, match="Mode must be one of"):
        sample_dense_array_2d.mode = "invalid"

    with pytest.raises(ValueError, match="Attribute .* does not exist"):
        DenseCellArray(uri=sample_dense_array_2d.uri, attr="invalid_attr")

    with pytest.raises(IndexError, match="Invalid number of dimensions"):
        _ = sample_dense_array_2d[0:10, 0:10, 0:10]

    with pytest.raises(IndexError, match="out of bounds"):
        _ = sample_dense_array_2d[200:300]


def test_array_object(temp_dir):
    uri = str(Path(temp_dir) / "test_dense_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=False)
    tdb_obj = tiledb.open(uri, "r")
    alt_array = DenseCellArray(tiledb_array_obj=tdb_obj)

    assert isinstance(array, DenseCellArray)
    assert array.shape == alt_array.shape
    assert array.ndim == alt_array.ndim
    assert array.dim_names == alt_array.dim_names
    assert "data" in array.attr_names
    assert "data" in alt_array.attr_names
