import os
import shutil

import numpy as np
import pytest
import scipy.sparse as sp

from cellarr_array import DenseCellArray, SparseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def dense_array_uri():
    uri = "test_dense_array_query"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    create_cellarray(uri, shape=(10, 5), sparse=False)

    arr = DenseCellArray(uri, mode="w")
    data = np.arange(50).reshape(10, 5)
    arr.write_batch(data, start_row=0)

    return uri


@pytest.fixture
def sparse_array_uri():
    uri = "test_sparse_array_query"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    arr = create_cellarray(uri, shape=(10, 5), sparse=True)
    data = sp.csr_matrix(np.arange(50).reshape(10, 5))
    arr.write_batch(data, start_row=0)
    return uri


def test_dense_array_query(dense_array_uri):
    arr = DenseCellArray(dense_array_uri)
    with pytest.raises(Exception):
        result = arr["dim_0 > 5"]

    result = arr["data > 5"]
    assert isinstance(result["data"], np.ndarray)


def test_sparse_array_query(sparse_array_uri):
    arr = SparseCellArray(sparse_array_uri, return_sparse=False)
    result = arr["dim_0 > 5"]
    # Even if empty, it should return a dictionary with the correct keys
    assert "data" in result
    assert "dim_0" in result
    assert "dim_1" in result


def test_get_unique_dim_values(sparse_array_uri):
    arr = SparseCellArray(sparse_array_uri)
    unique_rows = arr.get_unique_dim_values("dim_0")
    assert np.array_equal(unique_rows, np.arange(10))
