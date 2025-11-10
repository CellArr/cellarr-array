try:
    from types import EllipsisType
except ImportError:
    # TODO: This is required for Python <3.10. Remove once Python 3.9 reaches EOL in October 2025
    EllipsisType = type(...)
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import tiledb
from scipy import sparse

from .base import CellArray
from .helpers import SliceHelper

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class SparseCellArray(CellArray):
    """Implementation for sparse TileDB arrays."""

    def __init__(
        self,
        uri: Optional[str] = None,
        tiledb_array_obj: Optional[tiledb.Array] = None,
        attr: str = "data",
        mode: Optional[Literal["r", "w", "d", "m"]] = None,
        config_or_context: Optional[Union[tiledb.Config, tiledb.Ctx]] = None,
        return_sparse: bool = True,
        sparse_format: Union[sparse.csr_matrix, sparse.csc_matrix] = sparse.csr_matrix,
        validate: bool = True,
        **kwargs,
    ):
        """Initialize the object.

        Args:
            uri:
                URI to the array.
                Required if 'tiledb_array_obj' is not provided.

            tiledb_array_obj:
                Optional, an already opened ``tiledb.Array`` instance.
                If provided, 'uri' can be None, and 'config_or_context' is ignored.

            attr:
                Attribute to access.
                Defaults to "data".

            mode:
                Open the array object in read 'r', write 'w', modify
                'm' mode, or delete 'd' mode.

                Defaults to None for automatic mode switching.

                If 'tiledb_array_obj' is provided, this mode should ideally match
                the mode of the provided array or be None.

            config_or_context:
                Optional config or context object. Ignored if 'tiledb_array_obj' is provided,
                as context will be derived from the object.

                Defaults to None.

            return_sparse:
                Whether to return a sparse representation of the data when object is sliced.
                Default is to return a dictionary that contains coordinates and values.

            sparse_format:
                Format to return, defaults to csr_matrix.

            validate:
                Whether to validate the attributes.
                Defaults to True.

            kwargs:
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
        self.sparse_format = sparse.csr_matrix if sparse_format is None else sparse_format
        self._list_remaps = {}

    def _validate_matrix_dims(self, data: sparse.spmatrix) -> Tuple[sparse.coo_matrix, bool]:
        """Validate and adjust matrix dimensions if needed.

        Args:
            data:
                Input sparse matrix.

        Returns:
            Tuple of (adjusted matrix, is_1d flag).

        Raises:
            ValueError: If dimensions are incompatible.
        """
        coo_data = data.tocoo() if not isinstance(data, sparse.coo_matrix) else data

        is_1d = self.ndim == 1
        if is_1d:
            if coo_data.shape[0] == 1:
                # Convert (1,N) to (N,1)
                coo_data = sparse.coo_matrix(
                    (coo_data.data, (coo_data.col, np.zeros_like(coo_data.col))), shape=(coo_data.shape[1], 1)
                )
            elif coo_data.shape[1] != 1:
                raise ValueError(f"1D array expects (N, 1) matrix, got {coo_data.shape}")

        return coo_data, is_1d

    def _get_slice_details(self, key: Tuple[Union[slice, List], ...]) -> ...:
        """Calculates the shape, remapping info, and if a remap is needed for a slice."""
        shape = []
        origins_or_maps = []
        is_list_remap = []

        for i, idx in enumerate(key):
            dim_dtype = self.dim_dtypes[i]

            if isinstance(idx, slice):
                if np.issubdtype(dim_dtype, np.integer) or np.issubdtype(dim_dtype, np.datetime64):
                    start = idx.start if idx.start is not None else 0
                    stop = idx.stop if idx.stop is not None else self.shape[i]
                    shape.append(stop - start)
                    origins_or_maps.append(start)
                    is_list_remap.append(False)
                else:  # String dim
                    shape.append(self.shape[i])
                    origins_or_maps.append(None)
                    is_list_remap.append(False)

            elif isinstance(idx, list):
                shape.append(len(idx))
                remap_dict = {val: i for i, val in enumerate(idx)}
                origins_or_maps.append(remap_dict)
                self._list_remaps[i] = remap_dict
                is_list_remap.append(True)

            else:
                raise TypeError(f"Unsupported index type in key: {type(idx)}")

        # Handle 1D array case
        if self.ndim == 1:
            shape = (shape[0], 1)

        return tuple(shape), origins_or_maps, is_list_remap

    def _to_sparse_format(
        self, result: Dict[str, np.ndarray], key: Tuple[Union[slice, List[int]], ...]
    ) -> Union[np.ndarray, sparse.spmatrix]:
        """Convert TileDB result to CSR format or dense array."""
        data = result[self._attr]

        slice_shape, origins_or_maps, is_list_remap = self._get_slice_details(key)

        if len(data) == 0:
            return self.sparse_format(slice_shape)

        # Remap coordinates similar to cellarrdataset
        new_coords = []
        dim_names = self.dim_names

        for i in range(self.ndim):
            dim_name = dim_names[i]
            global_coords = result[dim_name]

            if is_list_remap[i]:
                remap_dict = self._list_remaps.get(i)
                if remap_dict is None:
                    raise RuntimeError("Internal error: Coordinate remap dictionary not found.")

                new_coords.append(np.array([remap_dict[val] for val in global_coords]))
            elif origins_or_maps[i] is not None:
                new_coords.append(global_coords - origins_or_maps[i])
            else:
                new_coords.append(global_coords)

        if self.ndim == 1:
            final_coords = (new_coords[0], np.zeros_like(new_coords[0]))
        else:
            final_coords = tuple(new_coords)

        matrix = sparse.coo_matrix((data, final_coords), shape=slice_shape)

        if self.sparse_format in (sparse.csr_matrix, sparse.csr_array):
            return matrix.tocsr()
        elif self.sparse_format in (sparse.csc_matrix, sparse.csc_array):
            return matrix.tocsc()

        return matrix

    def _direct_slice(self, key: Tuple[Union[slice, EllipsisType], ...]) -> Union[np.ndarray, sparse.coo_matrix]:
        """Implementation for direct slicing of sparse arrays."""
        self._list_remaps.clear()

        with self.open_array(mode="r") as array:
            result = array[key]

            if not self.return_sparse:
                return result

            return self._to_sparse_format(result, key)

    def _multi_index(self, key: Tuple[Union[slice, List[int]], ...]) -> Union[np.ndarray, sparse.coo_matrix]:
        """Implementation for multi-index access of sparse arrays."""
        self._list_remaps.clear()

        optimized_key = []
        for i, idx in enumerate(key):
            if isinstance(idx, list) and np.issubdtype(self.dim_dtypes[i], np.integer):
                slice_idx = SliceHelper.is_contiguous_indices(idx)
                optimized_key.append(slice_idx if slice_idx is not None else idx)
            else:
                optimized_key.append(idx)

        if all(isinstance(idx, slice) for idx in optimized_key):
            return self._direct_slice(tuple(optimized_key))

        tiledb_key = []
        for idx in key:
            if isinstance(idx, slice):
                stop = None if idx.stop is None else idx.stop - 1
                tiledb_key.append(slice(idx.start, stop, idx.step))
            else:
                tiledb_key.append(idx)

        with self.open_array(mode="r") as array:
            result = array.multi_index[tuple(tiledb_key)]

            if not self.return_sparse:
                return result

            return self._to_sparse_format(result, key)

    def write_batch(
        self, data: Union[sparse.spmatrix, sparse.csc_matrix, sparse.coo_matrix], start_row: int, **kwargs
    ) -> None:
        """Write a batch of sparse data to the array.

        Args:
            data:
                Scipy sparse matrix (CSR, CSC, or COO format).

            start_row:
                Starting row index for writing.

            **kwargs:
                Additional arguments passed to TileDB write operation.

        Raises:
            TypeError: If input is not a sparse matrix.
            ValueError: If dimensions don't match or bounds are exceeded.
        """
        if not sparse.issparse(data):
            raise TypeError("Input must be a scipy sparse matrix.")

        coo_data, is_1d = self._validate_matrix_dims(data)

        end_row = start_row + coo_data.shape[0]
        if end_row > self.shape[0]:
            raise ValueError(
                f"Write operation would exceed array bounds. End row {end_row} > array rows {self.shape[0]}."
            )

        if not is_1d and coo_data.shape[1] != self.shape[1]:
            raise ValueError(f"Data columns {coo_data.shape[1]} don't match array columns {self.shape[1]}.")

        adjusted_rows = coo_data.row + start_row
        with self.open_array(mode="w") as array:
            if is_1d:
                array[adjusted_rows] = coo_data.data
            else:
                array[adjusted_rows, coo_data.col] = coo_data.data
