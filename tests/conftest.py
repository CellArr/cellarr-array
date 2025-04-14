"""
Dummy conftest.py for cellarr_array.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cellarr_array import create_cellarray


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for TileDB arrays."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_dense_array_1d(temp_dir):
    """Create a sample 1D dense array for testing."""
    uri = str(Path(temp_dir) / "dense_array_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=False, attr_name="data")
    return array


@pytest.fixture
def sample_dense_array_2d(temp_dir):
    """Create a sample 2D dense array for testing."""
    uri = str(Path(temp_dir) / "dense_array_2d")
    array = create_cellarray(uri=uri, shape=(100, 50), attr_dtype=np.float32, sparse=False, attr_name="data")
    return array


@pytest.fixture
def sample_sparse_array_1d(temp_dir):
    """Create a sample 1D sparse array for testing."""
    uri = str(Path(temp_dir) / "sparse_array_1d")
    array = create_cellarray(uri=uri, shape=(100,), attr_dtype=np.float32, sparse=True, attr_name="data")
    return array


@pytest.fixture
def sample_sparse_array_2d(temp_dir):
    """Create a sample 2D sparse array for testing."""
    uri = str(Path(temp_dir) / "sparse_array_2d")
    array = create_cellarray(uri=uri, shape=(100, 50), attr_dtype=np.float32, sparse=True, attr_name="data")
    return array


# @pytest.fixture
# def sample_multi_attr_array(temp_dir):
#     """Create a sample array with multiple attributes."""
#     uri = str(Path(temp_dir) / "multi_attr_array")
#     config = CellArrConfig()

#     # Create custom schema with multiple attributes
#     import tiledb

#     dom = tiledb.Domain(
#         tiledb.Dim(name="rows", domain=(0, 99), tile=10, dtype=np.uint32),
#         tiledb.Dim(name="cols", domain=(0, 49), tile=10, dtype=np.uint32),
#     )

#     schema = tiledb.ArraySchema(
#         domain=dom,
#         attrs=[
#             tiledb.Attr(name="data", dtype=np.float32),
#             tiledb.Attr(name="values", dtype=np.float32),
#             tiledb.Attr(name="counts", dtype=np.int32),
#         ],
#         sparse=False,
#     )

#     tiledb.Array.create(uri, schema)

#     # Return array opened with default attribute
#     from cellarr_array import DenseCellArray

#     return DenseCellArray(uri, attr="data")
