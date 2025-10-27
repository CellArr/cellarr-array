import os
import shutil

import numpy as np
import pandas as pd
import pytest
import tiledb

from cellarr_array import SparseCellArray, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def string_dim_array_uri():
    uri = "test_string_dim_array"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    create_cellarray(uri, sparse=True, dim_dtypes=[str, str], attr_dtype=np.float64, attr_name="value")

    yield uri

    shutil.rmtree(uri)


def test_create_string_dim_schema(string_dim_array_uri):
    with tiledb.open(string_dim_array_uri, "r") as A:
        schema = A.schema
        assert schema.domain.dim(0).dtype == np.dtype("S")
        assert schema.domain.dim(1).dtype == np.dtype("S")
        assert schema.attr("value").dtype == np.float64


def test_string_dim_write_read(string_dim_array_uri):
    sca = SparseCellArray(string_dim_array_uri, attr="value", mode="w", return_sparse=False)

    rows = np.array(["cell_A", "cell_B", "cell_C"])
    cols = np.array(["gene_X", "gene_Y", "gene_Z"])
    values = np.array([1.1, 2.2, 3.3])

    with sca.open_array() as A:
        A[rows, cols] = values

    sca_read = SparseCellArray(string_dim_array_uri, attr="value", return_sparse=False)
    data = sca_read[:]
    data["dim_0"] = [x.decode("ascii") for x in data["dim_0"]]
    data["dim_1"] = [x.decode("ascii") for x in data["dim_1"]]

    assert len(data["value"]) == 3
    pd.testing.assert_frame_equal(
        pd.DataFrame({"value": values, "dim_0": rows, "dim_1": cols})
        .sort_values(by=["dim_0", "dim_1"])
        .reset_index(drop=True),
        pd.DataFrame(data).sort_values(by=["dim_0", "dim_1"]).reset_index(drop=True),
    )


def test_string_dim_slicing(string_dim_array_uri):
    sca = SparseCellArray(string_dim_array_uri, attr="value", mode="w", return_sparse=False)

    with sca.open_array() as A:
        A[["cell_A", "cell_A", "cell_B"], ["gene_X", "gene_Y", "gene_Y"]] = np.array([1.0, 2.0, 3.0])

    sca_read = SparseCellArray(string_dim_array_uri, attr="value", return_sparse=False)

    subset = sca_read[["cell_A"], :]

    assert len(subset["value"]) == 2
    assert all(r.decode("ascii") == "cell_A" for r in subset["dim_0"])
    assert set([x.decode("ascii") for x in subset["dim_1"]]) == {"gene_X", "gene_Y"}
