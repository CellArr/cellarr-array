try:
    from types import EllipsisType
except ImportError:
    # TODO: This is required for Python <3.10. Remove once Python 3.9 reaches EOL in October 2025
    EllipsisType = type(...)
from typing import List, Tuple, Union

import numpy as np

from .cellarray_base import CellArray
from .helpers import SliceHelper

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class DenseCellArray(CellArray):
    """Implementation for dense TileDB arrays."""

    def _direct_slice(self, key: Tuple[Union[slice, EllipsisType], ...]) -> np.ndarray:
        """Implementation for direct slicing of dense arrays.

        Args:
            key:
                Tuple of slice objects.

        Returns:
            Sliced data.
        """
        with self.open_array(mode="r") as array:
            res = array[key]
            return res[self._attr] if self._attr is not None else res

    def _multi_index(self, key: Tuple[Union[slice, List[int]], ...]) -> np.ndarray:
        """Implementation for multi-index access of dense arrays.

        Args:
            key:
                Tuple of slice objects or index lists.

        Returns:
            Sliced data.
        """
        # Try to optimize contiguous indices to slices
        optimized_key = []
        for idx in key:
            if isinstance(idx, list):
                slice_idx = SliceHelper.is_contiguous_indices(idx)
                optimized_key.append(slice_idx if slice_idx is not None else idx)
            else:
                optimized_key.append(idx)

        # If all indices are now slices, use direct slicing
        if all(isinstance(idx, slice) for idx in optimized_key):
            return self._direct_slice(tuple(optimized_key))

        # For mixed slice-list queries, adjust slice bounds to exclude upper bound
        tiledb_key = []
        for idx in key:
            if isinstance(idx, slice):
                # Adjust stop to be exclusive by subtracting 1 if stop is not None
                stop = None if idx.stop is None else idx.stop - 1
                tiledb_key.append(slice(idx.start, stop, idx.step))
            else:
                tiledb_key.append(idx)

        with self.open_array(mode="r") as array:
            res = array.multi_index[tuple(tiledb_key)]
            return res[self._attr] if self._attr is not None else res

    def write_batch(self, data: np.ndarray, start_row: int, **kwargs) -> None:
        """Write a batch of data to the dense array.

        Args:
            data:
                Numpy array to write.

            start_row:
                Starting row index for writing.

            **kwargs:
                Additional arguments passed to TileDB write operation.

        Raises:
            TypeError: If input is not a numpy array.
            ValueError: If dimensions don't match or bounds are exceeded.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if len(data.shape) != self.ndim:
            raise ValueError(f"Data dimensions {data.shape} don't match array dimensions {self.shape}.")

        # Check bounds
        end_row = start_row + data.shape[0]
        if end_row > self.shape[0]:
            raise ValueError(
                f"Write operation would exceed array bounds. End row {end_row} > array rows {self.shape[0]}."
            )

        if self.ndim == 2 and data.shape[1] != self.shape[1]:
            raise ValueError(f"Data columns {data.shape[1]} don't match array columns {self.shape[1]}.")

        # Construct write region
        if self.ndim == 1:
            write_region = slice(start_row, end_row)
        else:  # 2D
            write_region = (slice(start_row, end_row), slice(0, self.shape[1]))

        # write_data = {self._attr: data} if len(self.attr_names) > 1 else data
        with self.open_array(mode="w") as array:
            array[write_region] = data
