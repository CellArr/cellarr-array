try:
    from types import EllipsisType
except ImportError:
    # TODO: This is required for Python <3.10. Remove once Python 3.9 reaches EOL in October 2025
    EllipsisType = type(...)
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import tiledb
from scipy import sparse

from .base import CellArray
from .helpers import SliceHelper

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class DenseSparseCellArray(CellArray):
    """Dense TileDB array optimized for storing sparse data objects.

    This class provides a hybrid approach where sparse data (like scipy sparse matrices)
    is stored in a dense TileDB array. This combines the performance benefits of dense
    arrays (no coordinate materialization, lightweight indexing) with the ability to
    efficiently handle sparse data patterns.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        tiledb_array_obj: Optional[tiledb.Array] = None,
        attr: str = "data",
        mode: Optional[Literal["r", "w", "d", "m"]] = None,
        config_or_context: Optional[Union[tiledb.Config, tiledb.Ctx]] = None,
        return_sparse: bool = True,
        sparse_format: Union[sparse.csr_matrix, sparse.csc_matrix] = sparse.csr_matrix,
        fill_value: Union[int, float] = 0,
        auto_detect_sparsity: bool = True,
        validate: bool = True,
        **kwargs,
    ):
        """Initialize the `DenseSparseCellArray`.

        Args:
            uri:
                URI to the array.
                Required if 'tiledb_array_obj' is not provided.

            tiledb_array_obj:
                Optional, an already opened ``tiledb.Array`` instance.

            attr:
                Attribute to access.
                Defaults to "data".

            mode:
                Open mode ('r', 'w', 'd', 'm').
                Defaults to None.

            config_or_context:
                Optional config or context object.

            return_sparse:
                Whether to return sparse matrices on read.
                Defaults to True.

            sparse_format:
                Sparse matrix format to return.
                Defaults to csr_matrix.

            fill_value:
                Value to use for "empty" cells in dense storage.
                Defaults to 0.

            auto_detect_sparsity:
                Whether to automatically detect sparsity on write.
                Defaults to True.

            validate:
                Whether to validate the attributes.
                Defaults to True.

            **kwargs:
                Additional arguments.
        """
        super().__init__(
            uri=uri,
            tiledb_array_obj=tiledb_array_obj,
            attr=attr,
            mode=mode,
            config_or_context=config_or_context,
            validate=validate,
        )

        self.return_sparse = return_sparse
        self.sparse_format = sparse_format if sparse_format is not None else sparse.csr_matrix
        self.fill_value = fill_value
        self.auto_detect_sparsity = auto_detect_sparsity

    def _calculate_sparsity(self, data: np.ndarray) -> float:
        """Calculate the sparsity ratio of an array.

        Args:
            data:
                Input array.

        Returns:
            Sparsity ratio (0.0 = completely dense, 1.0 = completely sparse).
        """
        if data.size == 0:
            return 1.0

        # Count non-fill values
        non_fill_count = np.count_nonzero(data != self.fill_value)
        return 1.0 - (non_fill_count / data.size)

    def _validate_sparse_input(self, data: sparse.spmatrix) -> sparse.coo_matrix:
        """Validate and convert sparse matrix input.

        Args:
            data:
                Input sparse matrix.

        Returns:
            COO format matrix for processing.

        Raises:
            ValueError: If dimensions are incompatible.
        """
        if not sparse.issparse(data):
            raise TypeError("Input must be a scipy sparse matrix.")

        coo_data = data.tocoo() if not isinstance(data, sparse.coo_matrix) else data

        # Handle 1D arrays
        if self.ndim == 1:
            if coo_data.shape[0] == 1:
                # Convert (1,N) to (N,) conceptually, but store as (N,1) in TileDB
                expected_shape = (self.shape[0], 1)
            elif coo_data.shape[1] == 1:
                expected_shape = (self.shape[0], 1)
            else:
                raise ValueError(f"1D array expects (N,1) or (1,N) matrix, got {coo_data.shape}")
        else:
            expected_shape = self.shape

        if coo_data.shape != expected_shape and coo_data.shape != self.shape:
            raise ValueError(f"Matrix shape {coo_data.shape} doesn't match array shape {self.shape}")

        return coo_data

    def _sparse_to_dense(self, sparse_data: sparse.spmatrix) -> np.ndarray:
        """Convert sparse matrix to dense array for TileDB storage.

        Args:
            sparse_data:
                Input sparse matrix.

        Returns:
            Dense numpy array with fill_value for empty cells.
        """
        coo_data = self._validate_sparse_input(sparse_data)

        # Create dense array filled with fill_value
        if self.ndim == 1:
            dense_array = np.full((self.shape[0],), self.fill_value, dtype=coo_data.dtype)
            # For 1D, use row indices from COO matrix
            if coo_data.shape[0] == 1:
                dense_array[coo_data.col] = coo_data.data
            else:
                dense_array[coo_data.row] = coo_data.data
        else:
            dense_array = np.full(self.shape, self.fill_value, dtype=coo_data.dtype)
            dense_array[coo_data.row, coo_data.col] = coo_data.data

        return dense_array

    def _dense_to_sparse(
        self, dense_data: np.ndarray, target_shape: Optional[Tuple[int, ...]] = None
    ) -> sparse.spmatrix:
        """Convert dense array to sparse matrix format.

        Args:
            dense_data:
                Input dense array.

            target_shape:
                Optional target shape for the sparse matrix.

        Returns:
            Sparse matrix in the configured format.
        """
        # Find non-fill value indices
        if self.ndim == 1:
            non_fill_indices = np.where(dense_data != self.fill_value)[0]
            values = dense_data[non_fill_indices]

            # Create COO matrix in (N,1) format
            shape = target_shape if target_shape else (len(dense_data), 1)
            sparse_matrix = sparse.coo_matrix(
                (values, (non_fill_indices, np.zeros_like(non_fill_indices))), shape=shape
            )
        else:
            non_fill_indices = np.where(dense_data != self.fill_value)
            values = dense_data[non_fill_indices]

            shape = target_shape if target_shape else dense_data.shape
            sparse_matrix = sparse.coo_matrix((values, non_fill_indices), shape=shape)

        # Convert to sparse format
        if self.sparse_format in (sparse.csr_matrix, sparse.csr_array):
            return sparse_matrix.tocsr()
        elif self.sparse_format in (sparse.csc_matrix, sparse.csc_array):
            return sparse_matrix.tocsc()
        else:
            return sparse_matrix

    def _get_slice_shape(self, key: Tuple[Union[slice, List[int]], ...]) -> Tuple[int, ...]:
        """Calculate the shape of a sliced result.

        Args:
            key:
                Slice specification.

        Returns:
            Shape tuple of the result.
        """
        shape = []
        for i, idx in enumerate(key):
            if isinstance(idx, slice):
                start = idx.start or 0
                stop = idx.stop if idx.stop is not None else self.shape[i]
                shape.append(stop - start)
            elif isinstance(idx, list):
                shape.append(len(set(idx)))
            else:
                shape.append(1)

        return tuple(shape)

    def _direct_slice(self, key: Tuple[Union[slice, EllipsisType], ...]) -> Union[np.ndarray, sparse.spmatrix]:
        """Implementation for direct slicing using dense TileDB array.

        Args:
            key:
                Tuple of slice objects.

        Returns:
            Sliced data as dense array or sparse matrix based on return_sparse setting.
        """
        with self.open_array(mode="r") as array:
            # Get dense data from TileDB
            result = array[key]
            dense_data = result[self._attr] if self._attr is not None else result

            if not self.return_sparse:
                return dense_data

            # Convert to sparse format
            slice_shape = self._get_slice_shape(key)
            return self._dense_to_sparse(dense_data, target_shape=slice_shape)

    def _multi_index(self, key: Tuple[Union[slice, List[int]], ...]) -> Union[np.ndarray, sparse.spmatrix]:
        """Implementation for multi-index access using dense TileDB array.

        Args:
            key:
                Tuple of slice objects or index lists.

        Returns:
            Sliced data as dense array or sparse matrix based on return_sparse setting.
        """
        # Optimize contiguous indices to slices where possible
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

        # For mixed slice-list queries, we need to use TileDB's dense array indexing
        # Since we're using dense arrays, we can directly index without multi_index
        with self.open_array(mode="r") as array:
            result = array[key]
            dense_data = result[self._attr] if self._attr is not None else result

            if not self.return_sparse:
                return dense_data

            # Convert to sparse format
            slice_shape = self._get_slice_shape(key)
            return self._dense_to_sparse(dense_data, target_shape=slice_shape)

    def write_batch(
        self,
        data: Union[np.ndarray, sparse.spmatrix],
        start_row: int,
        auto_convert: bool = True,
        **kwargs,
    ) -> None:
        """Write a batch of data to the dense array.

        Args:
            data:
                Data to write (numpy array or scipy sparse matrix).

            start_row:
                Starting row index for writing.

            auto_convert:
                Whether to automatically convert sparse matrices to dense.

            **kwargs:
                Additional arguments passed to TileDB write operation.

        Raises:
            TypeError: If input type is not supported.
            ValueError: If dimensions don't match or bounds are exceeded.
        """
        if sparse.issparse(data):
            if auto_convert:
                dense_data = self._sparse_to_dense(data)
            else:
                raise ValueError("Sparse matrix input not supported with auto_convert=False")
        elif isinstance(data, np.ndarray):
            dense_data = data
        else:
            raise TypeError("Input must be a numpy array or scipy sparse matrix.")

        # Validate dimensions
        if len(dense_data.shape) != self.ndim:
            raise ValueError(f"Data dimensions {dense_data.shape} don't match array dimensions {self.shape}.")

        end_row = start_row + dense_data.shape[0]
        if end_row > self.shape[0]:
            raise ValueError(
                f"Write operation would exceed array bounds. End row {end_row} > array rows {self.shape[0]}."
            )

        if self.ndim == 2 and dense_data.shape[1] != self.shape[1]:
            raise ValueError(f"Data columns {dense_data.shape[1]} don't match array columns {self.shape[1]}.")

        # Determine write region
        if self.ndim == 1:
            write_region = slice(start_row, end_row)
        else:  # 2D
            write_region = (slice(start_row, end_row), slice(0, self.shape[1]))

        # Write to dense TileDB array
        with self.open_array(mode="w") as array:
            array[write_region] = dense_data

    def convert_to_pure_sparse(self, output_uri: str) -> "SparseCellArray":
        """Convert this dense-sparse array to a pure sparse array.

        Args:
            output_uri:
                URI for the new sparse array.

        Returns:
            New SparseCellArray instance.
        """
        from .helpers import create_cellarray

        # Create new sparse array with same dimensions
        sparse_array = create_cellarray(
            uri=output_uri,
            shape=self.shape,
            attr_dtype=None,  # Will be inferred
            sparse=True,
            dim_names=self.dim_names,
            attr_name=self._attr,
        )

        # Copy data in batches
        batch_size = 1000  # Configurable
        for start_row in range(0, self.shape[0], batch_size):
            end_row = min(start_row + batch_size, self.shape[0])

            # Read dense data
            if self.ndim == 1:
                key = slice(start_row, end_row)
            else:
                key = (slice(start_row, end_row), slice(0, self.shape[1]))

            dense_data = self._direct_slice((key,)) if isinstance(key, slice) else self._direct_slice(key)

            # Convert to sparse and write
            if isinstance(dense_data, np.ndarray):
                sparse_data = self._dense_to_sparse(dense_data)
            else:
                sparse_data = dense_data

            sparse_array.write_batch(sparse_data, start_row)

        return sparse_array
