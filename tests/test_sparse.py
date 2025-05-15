from pathlib import Path

import numpy as np
import pytest
import tiledb
from scipy import sparse

from cellarr_array import SparseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_1d_array_creation(temp_dir):
    uri = str(Path(temp_dir) / "test_sparse_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=True)

    assert isinstance(array, SparseCellArray)
    assert array.shape == (100,)
    assert array.ndim == 1
    assert array.dim_names == ["dim_0"]
    assert "data" in array.attr_names


def test_2d_array_creation(temp_dir):
    uri = str(Path(temp_dir) / "test_sparse_2d")
    array = create_cellarray(uri=uri, shape=(100, 50), attr_dtype=np.float32, sparse=True, dim_names=["rows", "cols"])

    assert isinstance(array, SparseCellArray)
    assert array.shape == (100, 50)
    assert array.ndim == 2
    assert array.dim_names == ["rows", "cols"]
    assert "data" in array.attr_names


def test_1d_write_batch(sample_sparse_array_1d):
    indices = np.array([1, 3, 5, 7, 9])
    data = np.random.random(len(indices)).astype(np.float32)
    sparse_data = sparse.coo_matrix((data, (indices, np.zeros(len(indices)))), shape=(10, 1))

    sample_sparse_array_1d.write_batch(sparse_data, start_row=0)

    result = sample_sparse_array_1d[0:10]
    expected = sparse_data.toarray().flatten()
    np.testing.assert_array_almost_equal(result.toarray().flatten(), expected)

    read_arr = SparseCellArray(uri=sample_sparse_array_1d.uri, return_sparse=False)

    # Full slice
    result = read_arr[0:10]
    np.testing.assert_array_almost_equal(result["data"], data)

    # Partial slice
    result = read_arr[2:7]
    expected = sparse_data.tocsr()[2:7]
    np.testing.assert_array_almost_equal(result["data"], expected.data)

    # Single index
    result = read_arr[3]
    expected = sparse_data.tocsr()[3]
    np.testing.assert_array_almost_equal(result["data"], expected.data)


def test_1d_empty_regions(sample_sparse_array_1d):
    indices = np.array([1, 3, 5])
    data = np.random.random(len(indices)).astype(np.float32)
    sparse_data = sparse.coo_matrix((data, (indices, np.zeros(len(indices)))), shape=(10, 1))

    sample_sparse_array_1d.write_batch(sparse_data, start_row=0)

    read_arr = SparseCellArray(uri=sample_sparse_array_1d.uri, return_sparse=True)

    # Query empty region
    result = read_arr[7:10]
    expected = sparse_data.toarray()[7:10].flatten()
    np.testing.assert_array_almost_equal(result.toarray().flatten(), expected)
    assert result.shape[1] == 3  # Should return full slice length
    assert np.all(result.data == 0)  # All values should be zero


def test_2d_formats(sample_sparse_array_2d):
    data = sparse.random(10, 50, density=0.1, format="coo", dtype=np.float32)

    sample_sparse_array_2d.write_batch(data, start_row=0)
    array_coo = SparseCellArray(uri=sample_sparse_array_2d.uri, return_sparse=True)
    result = array_coo[0:10, :]
    np.testing.assert_array_almost_equal(result.toarray(), data.toarray())


def test_coo_output(sample_sparse_array_2d):
    data = sparse.random(10, 50, density=0.1, format="coo", dtype=np.float32)
    sample_sparse_array_2d.write_batch(data, start_row=0)

    array_coo = SparseCellArray(uri=sample_sparse_array_2d.uri, return_sparse=True)

    # Test full slice
    result = array_coo[0:10, :]
    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_almost_equal(result.toarray(), data.toarray())

    # Test full slice with ellipsis
    result = array_coo[0:10, ...]
    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_almost_equal(result.toarray(), data.toarray())

    # Test partial slice
    data_csr = data.tocsr()
    result = array_coo[2:5, 10:20]
    assert sparse.isspmatrix_csr(result)
    np.testing.assert_array_almost_equal(result.toarray(), data_csr[2:5, 10:20].toarray())


def test_mixed_slice_list_bounds(sample_sparse_array_2d):
    data = sparse.random(100, 50, density=0.2, format="csr", dtype=np.float32)
    sample_sparse_array_2d.write_batch(data, start_row=0)

    array_coo = SparseCellArray(uri=sample_sparse_array_2d.uri, return_sparse=True)

    cols = [2, 4, 6]

    # Simple slice
    result = array_coo[10:20, cols]
    expected = data[10:20][:, cols]
    np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())

    # Slice at array bounds
    result = array_coo[90:100, cols]
    expected = data[90:100][:, cols]
    np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())

    # Slice with step
    with pytest.raises(Exception):
        result = array_coo[10:20:2, cols]

    # Empty region
    result = array_coo[50:60, [45, 46, 47]]
    expected = data[50:60][:, [45, 46, 47]]
    np.testing.assert_array_almost_equal(result.toarray(), expected.toarray())


def test_empty_regions(sample_sparse_array_2d):
    data = sparse.random(10, 50, density=0.1, format="coo", dtype=np.float32)
    sample_sparse_array_2d.write_batch(data, start_row=0)

    array_coo = SparseCellArray(uri=sample_sparse_array_2d.uri, return_sparse=True)

    # Query empty region
    result = array_coo[50:60, :]
    assert sparse.isspmatrix_csr(result)
    assert result.getnnz() == 0
    assert result.shape == (10, 50)


def test_bounds_checking(sample_sparse_array_2d):
    data = sparse.random(150, 50, density=0.1, format="coo", dtype=np.float32)
    with pytest.raises(ValueError, match="would exceed array bounds"):
        sample_sparse_array_2d.write_batch(data, start_row=0)

    data = sparse.random(10, 60, density=0.1, format="coo", dtype=np.float32)
    with pytest.raises(ValueError, match="Data columns"):
        sample_sparse_array_2d.write_batch(data, start_row=0)


def test_invalid_inputs(sample_sparse_array_2d):
    with pytest.raises(TypeError, match="must be a scipy sparse matrix"):
        sample_sparse_array_2d.write_batch(np.random.random((10, 50)), start_row=0)

    with pytest.raises(ValueError, match="Mode must be one of"):
        sample_sparse_array_2d.mode = "invalid"

    with pytest.raises(ValueError, match="Attribute .* does not exist"):
        SparseCellArray(sample_sparse_array_2d.uri, attr="invalid_attr")

    with pytest.raises(ValueError, match="Only 1D and 2D arrays are supported"):
        create_cellarray(
            uri=str(Path(sample_sparse_array_2d.uri).parent / "invalid"),
            shape=(10, 10, 10),
            attr_dtype=np.float32,
            sparse=True,
        )


def test_array_object(temp_dir):
    uri = str(Path(temp_dir) / "test_sparse_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=True)
    tdb_obj = tiledb.open(uri, "r")
    alt_array = SparseCellArray(tiledb_array_obj=tdb_obj)

    assert isinstance(array, SparseCellArray)
    assert array.shape == alt_array.shape
    assert array.ndim == alt_array.ndim
    assert array.dim_names == alt_array.dim_names
    assert "data" in array.attr_names
    assert "data" in alt_array.attr_names
