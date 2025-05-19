"""
Dummy conftest.py for cellarr_array.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

# import pytest

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cellarr_array import create_cellarray
from cellarr_array.utils.mock import generate_tiledb_dense_array, generate_tiledb_sparse_array


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


##########
### FOR dataloaders
##########


@pytest.fixture(scope="session")
def tiledb_ctx_config():
    """Configuration for TileDB context for tests."""
    return {
        "sm.tile_cache_size": str(10 * 1024**2),  # 10MB
        "sm.num_reader_threads": "2",
    }


@pytest.fixture
def temp_tiledb_uri(tmp_path: Path) -> str:
    """Create a temporary URI for a TileDB array."""
    return str(tmp_path / "test_array.tdb")


@pytest.fixture
def dense_array_params():
    """Parameters for a standard dense test array."""
    return {
        "rows": 100,
        "cols": 10,
        "attr_name": "data",
        "attr_dtype": np.float32,
        "chunk_size": 10,
    }


@pytest.fixture
def sparse_array_params():
    """Parameters for a standard sparse test array."""
    return {
        "rows": 150,
        "cols": 20,
        "density": 0.1,
        "attr_name": "data",
        "attr_dtype": np.float64,
        "chunk_size": 20,
        "sparse_format_to_write": "coo",
    }


@pytest.fixture
def created_dense_array_uri(temp_tiledb_uri: Path, dense_array_params, tiledb_ctx_config):
    """Creates a dense TileDB array for testing and returns its URI.

    Cleans up after the test.
    """
    uri = str(temp_tiledb_uri)
    if generate_tiledb_dense_array is None:
        pytest.skip("generate_tiledb_dense_array mock function not available.")

    generate_tiledb_dense_array(
        uri=uri,
        rows=dense_array_params["rows"],
        cols=dense_array_params["cols"],
        attr_name=dense_array_params["attr_name"],
        attr_dtype=dense_array_params["attr_dtype"],
        chunk_size=dense_array_params["chunk_size"],
        tiledb_config=tiledb_ctx_config,
    )
    yield uri
    if os.path.exists(uri):
        shutil.rmtree(uri)


@pytest.fixture
def created_sparse_array_uri(temp_tiledb_uri: Path, sparse_array_params, tiledb_ctx_config):
    """Creates a sparse TileDB array for testing and returns its URI.

    Cleans up after the test.
    """
    uri = str(temp_tiledb_uri)  # Use a different name or ensure temp_tiledb_uri gives unique paths if used in same test
    if generate_tiledb_sparse_array is None:
        pytest.skip("generate_tiledb_sparse_array mock function not available.")

    generate_tiledb_sparse_array(
        uri=uri,
        rows=sparse_array_params["rows"],
        cols=sparse_array_params["cols"],
        density=sparse_array_params["density"],
        attr_name=sparse_array_params["attr_name"],
        attr_dtype=sparse_array_params["attr_dtype"],
        chunk_size=sparse_array_params["chunk_size"],
        tiledb_config=tiledb_ctx_config,
        sparse_format_to_write=sparse_array_params["sparse_format_to_write"],
    )
    yield uri
    if os.path.exists(uri):
        shutil.rmtree(uri)


@pytest.fixture
def empty_dense_array_uri(tmp_path: Path, dense_array_params, tiledb_ctx_config):
    """Creates an empty dense TileDB array."""
    uri = str(tmp_path / "empty_dense_array.tdb")
    if generate_tiledb_dense_array is None:
        pytest.skip("generate_tiledb_dense_array mock function not available.")

    # Create array schema but write 0 rows
    generate_tiledb_dense_array(
        uri=uri,
        rows=0,  # 0 rows
        cols=dense_array_params["cols"],
        attr_name=dense_array_params["attr_name"],
        attr_dtype=dense_array_params["attr_dtype"],
        chunk_size=dense_array_params["chunk_size"],
        tiledb_config=tiledb_ctx_config,
    )
    yield uri
    if os.path.exists(uri):
        shutil.rmtree(uri)


@pytest.fixture
def empty_sparse_array_uri(tmp_path: Path, sparse_array_params, tiledb_ctx_config):
    """Creates an empty sparse TileDB array."""
    uri = str(tmp_path / "empty_sparse_array.tdb")
    if generate_tiledb_sparse_array is None:
        pytest.skip("generate_tiledb_sparse_array mock function not available.")

    generate_tiledb_sparse_array(
        uri=uri,
        rows=0,  # 0 rows
        cols=sparse_array_params["cols"],
        density=sparse_array_params["density"],
        attr_name=sparse_array_params["attr_name"],
        attr_dtype=sparse_array_params["attr_dtype"],
        chunk_size=sparse_array_params["chunk_size"],
        tiledb_config=tiledb_ctx_config,
    )
    yield uri
    if os.path.exists(uri):
        shutil.rmtree(uri)
