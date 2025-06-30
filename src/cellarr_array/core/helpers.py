try:
    from types import EllipsisType
except ImportError:
    # TODO: This is required for Python <3.10. Remove once Python 3.9 reaches EOL in October 2025
    EllipsisType = type(...)
from typing import List, Optional, Tuple, Union

import numpy as np
import tiledb

from ..utils.config import CellArrConfig

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
            Optional list of dimension dtypes. Defaults to numpy's uint32.

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
    tiledb_ctx = tiledb.Config(config.ctx_config) if config.ctx_config else None

    if attr_dtype is None:
        attr_dtype = np.float32
    if isinstance(attr_dtype, str):
        attr_dtype = np.dtype(attr_dtype)

    if shape is None and dim_dtypes is None:
        raise ValueError("Either 'shape' or 'dim_dtypes' must be provided.")

    if shape is not None:
        if len(shape) not in (1, 2):
            raise ValueError("Shape must have 1 or 2 dimensions.")

    # Set dimension dtypes, defaults to numpy uint32
    if dim_dtypes is None:
        dim_dtypes = [np.uint32] * len(shape)
    else:
        if len(dim_dtypes) not in (1, 2):
            raise ValueError("Array must have 1 or 2 dimensions.")
        dim_dtypes = [np.dtype(dt) if isinstance(dt, str) else dt for dt in dim_dtypes]

    if shape is None:
        shape = tuple(np.iinfo(dt).max if np.issubdtype(dt, np.integer) else None for dt in dim_dtypes)
    if None in shape:
        shape = tuple(
            np.iinfo(dt).max if s is None and np.issubdtype(dt, np.integer) else s for s, dt in zip(shape, dim_dtypes)
        )

    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(len(shape))]

    # Validate all input lengths
    if not (len(shape) == len(dim_dtypes) == len(dim_names)):
        raise ValueError("Lengths of 'shape', 'dim_dtypes', and 'dim_names' must match.")

    dom = tiledb.Domain(
        *[
            tiledb.Dim(
                name=name,
                # supporting empty dimensions
                domain=(0, 0 if s == 0 else s - 1),
                tile=min(1 if s == 0 else s // 2, config.tile_capacity // 2),
                dtype=dt,
            )
            for name, s, dt in zip(dim_names, shape, dim_dtypes)
        ],
        ctx=tiledb_ctx,
    )
    attr_obj = tiledb.Attr(
        name=attr_name,
        dtype=attr_dtype,
        filters=config.attrs_filters.get(attr_name, config.attrs_filters.get("", None)),
        ctx=tiledb_ctx,
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        attrs=[attr_obj],
        cell_order=config.cell_order,
        tile_order=config.tile_order,
        sparse=sparse,
        coords_filters=config.coords_filters,
        offsets_filters=config.offsets_filters,
        ctx=tiledb_ctx,
    )
    tiledb.Array.create(uri, schema, ctx=tiledb_ctx)

    # Import here to avoid circular imports
    from .dense import DenseCellArray
    from .sparse import SparseCellArray

    return (
        SparseCellArray(uri=uri, attr=attr_name, mode=mode, config_or_context=tiledb_ctx)
        if sparse
        else DenseCellArray(uri=uri, attr=attr_name, mode=mode, config_or_context=tiledb_ctx)
    )


class SliceHelper:
    """Helper class for handling array slicing operations."""

    @staticmethod
    def is_contiguous_indices(indices: List[int]) -> Optional[slice]:
        if not indices:
            return None

        sorted_indices = sorted(list(set(indices)))
        if not sorted_indices:
            return None

        if len(sorted_indices) == 1:
            return slice(sorted_indices[0], sorted_indices[0] + 1, None)

        diffs = np.diff(sorted_indices)
        if np.all(diffs == 1):
            return slice(sorted_indices[0], sorted_indices[-1] + 1, None)

        return None

    @staticmethod
    def normalize_index(
        idx: Union[int, range, slice, List[int], EllipsisType], dim_size: int
    ) -> Union[slice, List[int], EllipsisType]:
        """Normalize index to handle negative indices and ensure consistency."""
        if isinstance(idx, EllipsisType):
            return idx

        # Convert ranges to slices
        if isinstance(idx, range):
            idx = slice(idx.start, idx.stop, idx.step)

        if isinstance(idx, slice):
            start = idx.start
            stop = idx.stop
            step = idx.step

            # Resolve None to full dimension slice parts
            if start is None:
                start = 0

            if stop is None:
                stop = dim_size

            # Handle negative indices
            if start < 0:
                start += dim_size
            if stop < 0:
                stop += dim_size

            # slice allows start > dim_size or stop < 0 to result in empty slices.
            # Note: start == dim_size is OK for empty slice like arr[dim_size:]
            if start < 0 or (start >= dim_size and dim_size > 0):
                if not (start == dim_size and (step is None or step > 0)):
                    if start >= dim_size:
                        raise IndexError(
                            f"Start index {idx.start if idx.start is not None else 'None'} results in {start}, which is out of bounds for dimension size {dim_size}."
                        )

            # Clamping slice arguments to dimensions
            stop = min(stop, dim_size)
            start = max(0, start)

            return slice(start, stop, step)
        elif isinstance(idx, list):
            if not idx:
                return []

            norm_idx = [i if i >= 0 else dim_size + i for i in idx]
            if any(i < 0 or i >= dim_size for i in norm_idx):
                oob_indices = [orig_i for orig_i, norm_i in zip(idx, norm_idx) if not (0 <= norm_i < dim_size)]
                raise IndexError(
                    f"List indices {oob_indices} (original values) are out of bounds for dimension size {dim_size}."
                )

            # TileDB multi_index usually returns data sorted by coordinates
            return sorted(list(set(norm_idx)))
        elif isinstance(idx, (int, np.integer)):
            norm_idx = int(idx)
            if norm_idx < 0:
                norm_idx += dim_size

            if not (0 <= norm_idx < dim_size):
                raise IndexError(f"Index {idx} out of bounds for dimension size {dim_size}")

            return slice(norm_idx, norm_idx + 1, None)
        else:
            raise TypeError(f"Index type {type(idx)} not supported for normalization.")


def create_group(output_path, group_name):
    tiledb.group_create(f"{output_path}/{group_name}")
