from pathlib import Path

import numpy as np
import pytest
import scipy as sp

from cellarr_array import DenseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_inmem_uri():
    shape = (10_000, 10_000)
    arr = np.arange(100_000_000).reshape(shape)
    uri = "mem://dense"

    dense_inmem = create_cellarray(uri, shape=(shape))
    dense_inmem.write_batch(arr, start_row=0)

    assert np.allclose(dense_inmem[:10, :10], arr[:10, :10])


def test_inmem_uri_sparse():
    shape = (1000, 1000)

    s = sp.sparse.random(1000, 1000, density=0.25)
    uri = "mem://sparse"

    dense_inmem = create_cellarray(uri, shape=(shape), sparse=True)
    dense_inmem.write_batch(s, start_row=0)

    assert np.allclose(dense_inmem[:10, :10].toarray(), s.tocsr()[:10, :10].toarray())
