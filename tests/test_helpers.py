from pathlib import Path

import numpy as np
import pytest
import tiledb

from cellarr_array import CellArrConfig, ConsolidationConfig, SliceHelper, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def test_slice_is_contiguous_indices():
    """Test detection of contiguous indices."""
    # Test contiguous indices
    assert SliceHelper.is_contiguous_indices([1, 2, 3, 4]) == slice(1, 5, None)

    # Test non-contiguous indices
    assert SliceHelper.is_contiguous_indices([1, 3, 5]) is None

    # Test empty list
    assert SliceHelper.is_contiguous_indices([]) is None

    # Test single element
    assert SliceHelper.is_contiguous_indices([1]) == slice(1, 2, None)


def test_slice_normalize_index():
    """Test index normalization."""
    dim_size = 10

    # Test positive slice
    assert SliceHelper.normalize_index(slice(1, 5), dim_size) == slice(1, 5, None)

    # Test negative slice
    assert SliceHelper.normalize_index(slice(-3, -1), dim_size) == slice(7, 9, None)

    # Test None values in slice
    assert SliceHelper.normalize_index(slice(None, None), dim_size) == slice(0, 10, None)

    # Test list of indices
    assert SliceHelper.normalize_index([1, -1], dim_size) == [1, 9]

    # Test single integer
    assert SliceHelper.normalize_index(5, dim_size) == slice(5, 6, None)
    assert SliceHelper.normalize_index(-1, dim_size) == slice(9, 10, None)


def test_slice_bounds_validation():
    """Test slice bounds validation."""
    dim_size = 10

    # Test out of bounds positive indices
    with pytest.raises(IndexError, match="out of bounds"):
        SliceHelper.normalize_index(15, dim_size)

    with pytest.raises(IndexError, match="out of bounds"):
        SliceHelper.normalize_index(slice(0, 15), dim_size)

    # Test out of bounds negative indices
    with pytest.raises(IndexError, match="out of bounds"):
        SliceHelper.normalize_index(-15, dim_size)

    # Test out of bounds list indices
    with pytest.raises(IndexError, match="out of bounds"):
        SliceHelper.normalize_index([5, 12], dim_size)


def test_cellarr_config():
    """Test CellArrConfig initialization and validation."""
    # Test default configuration
    config = CellArrConfig()
    assert isinstance(config.coords_filters[0], tiledb.Filter)
    assert isinstance(config.offsets_filters[0], tiledb.Filter)
    assert isinstance(config.attrs_filters[""][0], tiledb.Filter)

    # Test custom configuration
    config = CellArrConfig(
        tile_capacity=50000, cell_order="col-major", attrs_filters={"data": [{"name": "gzip", "level": 5}]}
    )
    assert config.tile_capacity == 50000
    assert config.cell_order == "col-major"
    assert isinstance(config.attrs_filters["data"][0], tiledb.GzipFilter)

    # Test invalid filter
    with pytest.raises(ValueError, match="Unsupported filter type"):
        CellArrConfig(attrs_filters={"data": [{"name": "invalid"}]})


def test_consolidation_config():
    """Test ConsolidationConfig initialization and validation."""
    # Test default configuration
    config = ConsolidationConfig()
    assert "fragment" in config.steps
    assert "fragment_meta" in config.steps
    assert config.vacuum_after is True

    # Test custom configuration
    config = ConsolidationConfig(steps=["fragment"], num_threads=2, vacuum_after=False)
    assert config.steps == ["fragment"]
    assert config.num_threads == 2
    assert config.vacuum_after is False


def test_create_cellarray_validation(temp_dir):
    """Test validation in create_cellarray function."""
    base_uri = str(Path(temp_dir) / "test_array")

    # Test missing shape and dim_dtypes
    with pytest.raises(ValueError, match="Either 'shape' or 'dim_dtypes' must be provided"):
        create_cellarray(uri=base_uri + "_1")

    # Test mismatched dimensions
    with pytest.raises(ValueError, match="Lengths .* must match"):
        create_cellarray(uri=base_uri + "_2", shape=(10, 10), dim_dtypes=[np.uint32], dim_names=["dim1"])

    # Test 3D array creation
    with pytest.raises(ValueError, match="Only 1D and 2D arrays are supported"):
        create_cellarray(uri=base_uri + "_3", shape=(10, 10, 10))


def test_create_cellarray_dtypes(temp_dir):
    """Test dtype handling in create_cellarray."""
    base_uri = str(Path(temp_dir) / "dtype_test")

    # Test string dtype conversion
    array = create_cellarray(uri=base_uri + "_1", shape=(10, 10), attr_dtype="float32")
    assert array._attr == "data"

    # Test custom dim_dtypes
    array = create_cellarray(uri=base_uri + "_2", shape=(10, 10), dim_dtypes=["uint16", "uint16"])

    # Verify dimension dtypes through TileDB schema
    with array.open_array("r") as A:
        assert A.schema.domain.dim(0).dtype == np.uint16
        assert A.schema.domain.dim(1).dtype == np.uint16

    with pytest.raises(Exception):
        create_cellarray(uri=base_uri + "_2", shape=(10, 10), dim_dtypes=["uint16", "uint16"])


def test_create_cellarray_automatic_shape(temp_dir):
    """Test automatic shape calculation from dtypes."""
    uri = str(Path(temp_dir) / "auto_shape")

    # Create array with None shape
    array = create_cellarray(uri=uri, shape=(None, None), dim_dtypes=[np.uint8, np.uint8])

    # Verify shapes were set to dtype maximums
    expected_shape = (np.iinfo(np.uint8).max, np.iinfo(np.uint8).max)
    assert array.shape == expected_shape
