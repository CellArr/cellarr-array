import numpy as np
import scipy as sp
import tiledb

from cellarr_array import DenseCellArray, SparseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_inmem_uri():
    shape = (10_000, 10_000)
    arr = np.arange(100_000_000).reshape(shape)
    uri = "mem://dense"

    dense_inmem = create_cellarray(uri=uri, shape=(shape))
    dense_inmem.write_batch(arr, start_row=0)

    assert np.allclose(dense_inmem[:10, :10], arr[:10, :10])

    tdb_obj = tiledb.open(uri, "r")
    alt_array = DenseCellArray(tiledb_array_obj=tdb_obj)
    assert np.allclose(alt_array[:10, :10], arr[:10, :10])


def test_inmem_uri_sparse():
    shape = (1000, 1000)

    s = sp.sparse.random(1000, 1000, density=0.25)
    uri = "mem://sparse"

    dense_inmem = create_cellarray(uri, shape=(shape), sparse=True)
    dense_inmem.write_batch(s, start_row=0)

    assert np.allclose(dense_inmem[:10, :10].toarray(), s.tocsr()[:10, :10].toarray())

    tdb_obj = tiledb.open(uri, "r")
    alt_array = SparseCellArray(tiledb_array_obj=tdb_obj)
    assert np.allclose(alt_array[:10, :10].toarray(), s.tocsr()[:10, :10].toarray())
