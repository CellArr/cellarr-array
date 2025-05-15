try:
    from types import EllipsisType
except ImportError:
    # TODO: This is required for Python <3.10. Remove once Python 3.9 reaches EOL in October 2025
    EllipsisType = type(...)
from typing import List, Optional, Tuple, Union

import numpy as np
import tiledb

from .config import CellArrConfig

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_cellarray(
    uri: str,
    shape: Optional[Tuple[Optional[int], ...]] = None,
    attr_dtype: Optional[Union[str, np.dtype]] = None,
    sparse: bool = False,
    mode: str = None,
    config: Optional[CellArrConfig] = None,
    dim_names: Optional[List[str]] = None,
    dim_dtypes: Optional[List[Union[str, np.dtype]]] = None,
    attr_name: str = "data",
    **kwargs,
):
    """Factory function to create a new TileDB cell array.

    Args:
        uri:
            Array URI.

        shape:
            Optional array shape. If None or contains None, uses dtype max.

        attr_dtype:
            Data type for the attribute. Defaults to float32.

        sparse:
            Whether to create a sparse array.

        mode:
            Array open mode. Defaults to None for automatic switching.

        config:
            Optional configuration.

        dim_names:
            Optional list of dimension names.

        dim_dtypes:
            Optional list of dimension dtypes.

        attr_name:
            Name of the data attribute.

        **kwargs:
            Additional arguments for array creation.

    Returns:
        CellArray instance.

    Raises:
        ValueError: If dimensions are invalid or inputs are inconsistent.
    """
    config = config or CellArrConfig()

    if attr_dtype is None:
        attr_dtype = np.float32
    if isinstance(attr_dtype, str):
        attr_dtype = np.dtype(attr_dtype)

    # Require either shape or dim_dtypes
    if shape is None and dim_dtypes is None:
        raise ValueError("Either 'shape' or 'dim_dtypes' must be provided.")

    if shape is not None:
        if len(shape) not in (1, 2):
            raise ValueError("Only 1D and 2D arrays are supported.")

    # Set dimension dtypes, defaults to numpy uint32
    if dim_dtypes is None:
        dim_dtypes = [np.uint32] * len(shape)
    else:
        if len(dim_dtypes) not in (1, 2):
            raise ValueError("Only 1D and 2D arrays are supported.")
        dim_dtypes = [np.dtype(dt) if isinstance(dt, str) else dt for dt in dim_dtypes]

    # Calculate shape from dtypes if needed
    if shape is None:
        shape = tuple(np.iinfo(dt).max if np.issubdtype(dt, np.integer) else None for dt in dim_dtypes)
    if None in shape:
        shape = tuple(
            np.iinfo(dt).max if s is None and np.issubdtype(dt, np.integer) else s for s, dt in zip(shape, dim_dtypes)
        )

    # Set dimension names
    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(len(shape))]

    # Validate all input lengths
    if not (len(shape) == len(dim_dtypes) == len(dim_names)):
        raise ValueError("Lengths of 'shape', 'dim_dtypes', and 'dim_names' must match.")

    dom = tiledb.Domain(
        *[
            tiledb.Dim(name=name, domain=(0, s - 1), tile=min(s, config.tile_capacity), dtype=dt)
            for name, s, dt in zip(dim_names, shape, dim_dtypes)
        ],
        ctx=tiledb.Ctx(config.ctx_config),
    )

    attr = tiledb.Attr(
        name=attr_name,
        dtype=attr_dtype,
        filters=config.attrs_filters.get(attr_name, config.attrs_filters.get("", None)),
    )

    schema = tiledb.ArraySchema(
        domain=dom,
        attrs=[attr],
        cell_order=config.cell_order,
        tile_order=config.tile_order,
        sparse=sparse,
        coords_filters=config.coords_filters,
        offsets_filters=config.offsets_filters,
        ctx=tiledb.Ctx(config.ctx_config),
    )

    tiledb.Array.create(uri, schema)

    # Import here to avoid circular imports
    from .cellarray_dense import DenseCellArray
    from .cellarray_sparse import SparseCellArray

    # Return appropriate array type
    return (
        SparseCellArray(uri=uri, attr=attr_name, mode=mode)
        if sparse
        else DenseCellArray(uri=uri, attr=attr_name, mode=mode)
    )


class SliceHelper:
    """Helper class for handling array slicing operations."""

    @staticmethod
    def is_contiguous_indices(indices: List[int]) -> Optional[slice]:
        """Check if indices can be represented as a contiguous slice."""
        if not indices:
            return None

        diffs = np.diff(indices)
        if np.all(diffs == 1):
            return slice(indices[0], indices[-1] + 1, None)
        return None

    @staticmethod
    def normalize_index(idx: Union[int, slice, List[int]], dim_size: int) -> Union[slice, List[int], EllipsisType]:
        """Normalize index to handle negative indices and ensure consistency."""

        if isinstance(idx, EllipsisType):
            return idx

        # Convert ranges to slices
        if isinstance(idx, range):
            idx = slice(idx.start, idx.stop, idx.step)

        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else dim_size
            step = idx.step

            # Handle negative indices
            if start < 0:
                start = dim_size + start

            if stop < 0:
                stop = dim_size + stop

            if start < 0 or start > dim_size:
                raise IndexError(f"Start index {start} out of bounds for dimension size {dim_size}")
            if stop < 0 or stop > dim_size:
                raise IndexError(f"Stop index {stop} out of bounds for dimension size {dim_size}")

            return slice(start, stop, step)

        elif isinstance(idx, list):
            norm_idx = [i if i >= 0 else dim_size + i for i in idx]
            if any(i < 0 or i >= dim_size for i in norm_idx):
                raise IndexError(f"List indices {idx} out of bounds for dimension size {dim_size}")
            return norm_idx

        else:  # Single integer index
            norm_idx = idx if idx >= 0 else dim_size + idx

            if norm_idx < 0 or norm_idx >= dim_size:
                raise IndexError(f"Index {idx} out of bounds for dimension size {dim_size}")
            return slice(norm_idx, norm_idx + 1, None)


def create_group(output_path, group_name):
    tiledb.group_create(f"{output_path}/{group_name}")
