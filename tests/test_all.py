from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from cellarr_array import CellArrConfig, ConsolidationConfig, DenseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_dimension_validation(temp_dir):
    uri = str(Path(temp_dir) / "invalid_dims")

    with pytest.raises(ValueError, match="Only 1D and 2D arrays are supported"):
        create_cellarray(uri=uri, shape=(10, 10, 10), attr_dtype=np.float32)


def test_attribute_validation(temp_dir):
    uri = str(Path(temp_dir) / "attr_test")

    create_cellarray(uri=uri, shape=(10, 10), attr_dtype=np.float32, attr_name="values")

    with pytest.raises(ValueError, match="Attribute 'invalid' does not exist"):
        DenseCellArray(uri=uri, attr="invalid")


def test_1d_integration(temp_dir):
    dense_uri = str(Path(temp_dir) / "dense_1d")
    sparse_uri = str(Path(temp_dir) / "sparse_1d")

    dense_array = create_cellarray(uri=dense_uri, shape=(100,), attr_dtype=np.float32, sparse=False)
    sparse_array = create_cellarray(uri=sparse_uri, shape=(100,), attr_dtype=np.float32, sparse=True)

    dense_data = np.random.random(30).astype(np.float32)
    dense_array.write_batch(dense_data, start_row=0)

    result = dense_array[5:15]
    np.testing.assert_array_almost_equal(result, dense_data[5:15])

    sparse_data = sparse.random(30, 1, density=0.3, format="coo", dtype=np.float32)
    sparse_array.write_batch(sparse_data, start_row=0)

    result = sparse_array[5:15].toarray().flatten()
    expected = sparse_data.toarray()[5:15, :].flatten()
    np.testing.assert_array_almost_equal(result, expected)


def test_2d_integration(temp_dir):
    dense_uri = str(Path(temp_dir) / "dense_2d")
    sparse_uri = str(Path(temp_dir) / "sparse_2d")

    config = CellArrConfig(tile_capacity=1000, attrs_filters={"data": [{"name": "gzip", "level": 5}]})

    dense_array = create_cellarray(uri=dense_uri, shape=(100, 50), attr_dtype=np.float32, sparse=False, config=config)

    sparse_array = create_cellarray(
        uri=sparse_uri, shape=(100, 50), attr_dtype=np.float32, sparse=True, return_coo=True
    )

    dense_data = np.random.random((30, 50)).astype(np.float32)
    dense_array.write_batch(dense_data, start_row=0)

    sparse_data = sparse.random(30, 50, density=0.1, format="csr", dtype=np.float32)
    sparse_array.write_batch(sparse_data, start_row=0)

    # Direct slice
    result1 = dense_array[5:15, 10:20]
    np.testing.assert_array_almost_equal(result1, dense_data[5:15, 10:20])

    # Multi-index
    rows = [1, 3, 5]
    cols = [2, 4, 6]
    result2 = dense_array[rows, cols]
    np.testing.assert_array_almost_equal(result2, dense_data[rows][:, cols])

    result3 = sparse_array[5:15, 10:20]
    np.testing.assert_array_almost_equal(result3.toarray(), sparse_data[5:15, 10:20].toarray())

    # Test maintenance operations
    dense_array.consolidate(ConsolidationConfig(vacuum_after=True))
    sparse_array.vacuum()


# def test_multi_attribute_handling(sample_multi_attr_array):
#     data = np.random.random((10, 50)).astype(np.float32)

#     # Test writing to specific attribute
#     sample_multi_attr_array.write_batch(data, start_row=0)

#     result = sample_multi_attr_array[0:10, :]
#     np.testing.assert_array_almost_equal(result, data)

#     # Switch attribute
#     array_values = DenseCellArray(sample_multi_attr_array.uri, attr="values")

#     result = array_values[0:10, :]
#     assert result.shape == data.shape
