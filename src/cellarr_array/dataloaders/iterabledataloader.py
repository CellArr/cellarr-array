from typing import Callable, Dict, Iterator, Optional, Union

import numpy as np
import scipy.sparse as sp
import tiledb
import torch
from torch.utils.data import DataLoader, IterableDataset

from .utils import seed_worker

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrayIterableDataset(IterableDataset):
    """A PyTorch IterableDataset that yields batches of randomly sampled rows
    from a TileDB array (dense or sparse) using cellarr-array.

    An `IterableDataset` dataset is responsible for yielding entire batches of data,
    giving us full control over how a batch is formed, including
    performing a single bulk read from TileDB.
    """

    def __init__(
        self,
        array_uri: str,
        attribute_name: str,
        num_rows: int,
        num_columns: int,
        is_sparse: bool,
        batch_size: int = 1000,
        num_yields_per_epoch_per_worker: Optional[int] = None,
        cellarr_ctx_config: Optional[Dict] = None,
        transform: Optional[Callable] = None,
    ):
        """Initializes the `CellArrayIterableDataset`.

        Args:
            array_uri:
                URI of the TileDB array.

            attribute_name:
                Name of the TileDB attribute to read.

            num_rows:
                The total number of rows in the TileDB array.

            num_columns:
                The total number of columns in the TileDB array.

            is_sparse:
                True if the TileDB array is sparse, False if dense.

            batch_size:
                The number of random samples to include in each yielded batch. Defaults to 1000.

            num_yields_per_epoch_per_worker:
                The number of batches this dataset's iterator (per worker) will yield in one epoch.
                If None, it defaults to roughly covering all samples once across all workers (approx).
                The total batches seen by the training loop will be `num_workers * num_yields_per_epoch_per_worker`.
                Defaults to None.

            cellarr_ctx_config:
                Configuration dictionary for the TileDB context used by cellarr-array. Defaults to None.

            transform:
                A function/transform that takes the entire fetched batch (NumPy array for dense,
                SciPy sparse matrix for sparse) and returns a transformed version. Defaults to None.
        """
        super().__init__()

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        if not isinstance(num_rows, int) or num_rows < 0:
            raise ValueError("num_rows must be a non-negative integer.")

        if not isinstance(num_columns, int) or num_columns <= 0:
            raise ValueError("num_columns must be a positive integer.")

        self.array_uri = array_uri
        self.attribute_name = attribute_name
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.is_sparse = is_sparse
        self.batch_size = batch_size
        self.cellarr_ctx_config = cellarr_ctx_config
        self.transform = transform

        if num_yields_per_epoch_per_worker is None:
            # Default to roughly one pass over the data across all workers
            self.num_yields_per_epoch_per_worker = self.num_rows // self.batch_size
            if self.num_yields_per_epoch_per_worker == 0 and self.num_rows > 0:
                self.num_yields_per_epoch_per_worker = 1
        else:
            self.num_yields_per_epoch_per_worker = num_yields_per_epoch_per_worker

        self.cell_array_instance = None

    def _init_worker_state(self) -> None:
        """Initializes the cellarr-array instance (DenseCellArray or SparseCellArray)
        for the current worker process.

        This makes sure that each worker has its own TileDB array object.
        """
        if self.cell_array_instance is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else "Main"
            print(f"Worker {worker_id}: Initializing CellArray instance.")

            ctx = tiledb.Ctx(self.cellarr_ctx_config) if self.cellarr_ctx_config else None
            if self.is_sparse:
                from cellarr_array import SparseCellArray

                self.cell_array_instance = SparseCellArray(
                    uri=self.array_uri,
                    attr=self.attribute_name,
                    mode="r",
                    config_or_context=ctx,
                    return_sparse=True,
                    sparse_coerce=sp.coo_matrix,
                )
            else:
                from cellarr_array import DenseCellArray

                self.cell_array_instance = DenseCellArray(
                    uri=self.array_uri, attr=self.attribute_name, mode="r", config_or_context=ctx
                )

    def _fetch_one_random_batch(self) -> Union[np.ndarray, sp.spmatrix]:
        """Randomly selects `self.batch_size` row indices and fetches them from
        the TileDB array in a single multi-index read operation.

        Returns:
            A NumPy array (for dense) or SciPy sparse matrix (for sparse)
            representing the fetched batch of data. The shape will be
            (N, self.num_columns), where N is the number of successfully fetched rows
            (usually self.batch_size, but could be less if num_rows < self.batch_size).
        """
        if self.num_rows == 0:
            if self.is_sparse:
                return sp.coo_matrix((0, self.num_columns), dtype=np.float32)
            else:
                return np.empty((0, self.num_columns), dtype=np.float32)

        actual_batch_size = min(self.batch_size, self.num_rows)
        if actual_batch_size == 0:
            if self.is_sparse:
                return sp.coo_matrix((0, self.num_columns), dtype=np.float32)
            else:
                return np.empty((0, self.num_columns), dtype=np.float32)

        random_indices = np.random.choice(
            self.num_rows,
            size=actual_batch_size,
            replace=False,
        )

        random_indices.sort()
        batch_slice_key = (list(random_indices), slice(None))
        data_chunk = self.cell_array_instance[batch_slice_key]

        return data_chunk

    def __iter__(self) -> Iterator[Union[np.ndarray, sp.spmatrix]]:
        """Yields batches of randomly sampled data.

        This method is called by the DataLoader for each worker.
        """
        self._init_worker_state()

        for _ in range(self.num_yields_per_epoch_per_worker):
            if self.num_rows == 0:
                if self.is_sparse:
                    yield sp.coo_matrix((0, self.num_columns), dtype=np.float32)
                else:
                    yield np.empty((0, self.num_columns), dtype=np.float32)

                break

            batch_data = self._fetch_one_random_batch()

            if self.transform:
                batch_data = self.transform(batch_data)
            yield batch_data


def dense_batch_collate_fn(numpy_batch: np.ndarray) -> torch.Tensor:
    """Collate function for a dense batch from CellArrayIterableDataset.

    Receives the numpy_batch directly from the dataset's iterator.
    """
    if numpy_batch is None or (hasattr(numpy_batch, "shape") and numpy_batch.shape[0] == 0):
        print("CollateFn (Dense): Received batch_item that is None or has 0 rows.")
        if numpy_batch is not None:
            return torch.from_numpy(numpy_batch)
        return torch.empty(0)

    return torch.from_numpy(numpy_batch)


def sparse_batch_collate_fn(scipy_sparse_batch: sp.spmatrix) -> torch.Tensor:
    """Collate function for a sparse batch from CellArrayIterableDataset.

    Receives the scipy_sparse_batch directly from the dataset's iterator.
    """
    if scipy_sparse_batch is None or (hasattr(scipy_sparse_batch, "shape") and scipy_sparse_batch.shape[0] == 0):
        print("CollateFn (Sparse): Received batch_item that is None or has 0 rows.")
        num_cols = 0
        dtype_to_use = torch.float32
        if scipy_sparse_batch is not None:
            num_cols = scipy_sparse_batch.shape[1]
            try:
                dtype_to_use = torch.from_numpy(scipy_sparse_batch.data[:0]).dtype
            except (AttributeError, TypeError):
                try:
                    if hasattr(scipy_sparse_batch, "dtype"):
                        if scipy_sparse_batch.dtype == np.float32:
                            dtype_to_use = torch.float32
                        elif scipy_sparse_batch.dtype == np.float64:
                            dtype_to_use = torch.float64
                        elif scipy_sparse_batch.dtype == np.int32:
                            dtype_to_use = torch.int32
                except Exception:
                    pass

        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=dtype_to_use),
            (0, num_cols),
        )

    if not isinstance(scipy_sparse_batch, sp.coo_matrix):
        scipy_sparse_batch = scipy_sparse_batch.tocoo()

    if scipy_sparse_batch.nnz == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.from_numpy(scipy_sparse_batch.data[:0]).dtype),
            torch.Size(scipy_sparse_batch.shape),
        )

    values = torch.from_numpy(scipy_sparse_batch.data)
    indices = torch.from_numpy(np.vstack((scipy_sparse_batch.row, scipy_sparse_batch.col))).long()
    sparse_shape = torch.Size(scipy_sparse_batch.shape)

    return torch.sparse_coo_tensor(indices, values, sparse_shape)


def construct_iterable_dataloader(
    array_uri: str,
    is_sparse: bool,
    attribute_name: str = "data",
    num_rows: int = None,
    num_columns: int = None,
    batch_size: int = 1000,
    num_workers_dl: int = 2,
    num_yields_per_worker: int = 5,
) -> DataLoader:
    """Construct an instance of `CellArrayIterableDataset` with PyTorch DataLoader.

    Args:
        array_uri:
            URI of the TileDB array.

        attribute_name:
            Name of the attribute to read from.

        num_rows:
            The total number of rows in the TileDB array.

        num_columns:
            The total number of columns in the TileDB array.

        is_sparse:
            True if the array is sparse, False for dense.

        batch_size:
            Number of random samples per batch generated by the dataset.

        num_workers_dl:
            Number of worker processes for the DataLoader.

        num_yields_per_worker:
            Number of batches each worker should yield per epoch.
    """
    tiledb_ctx_config = {
        "sm.tile_cache_size": 2000 * 1024**2,  # 2000MB tile cache
        "sm.num_reader_threads": 4,
    }

    if num_rows is None or num_columns is None:
        raise ValueError("num_rows and num_columns must be provided for CellArrayIterableDataset.")

    dataset = CellArrayIterableDataset(
        array_uri=array_uri,
        attribute_name=attribute_name,
        num_rows=num_rows,
        num_columns=num_columns,
        is_sparse=is_sparse,
        batch_size=batch_size,
        num_yields_per_epoch_per_worker=num_yields_per_worker,
        cellarr_ctx_config=tiledb_ctx_config,
    )

    collate_to_use = sparse_batch_collate_fn if is_sparse else dense_batch_collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers_dl,
        worker_init_fn=seed_worker,
        collate_fn=collate_to_use,
        pin_memory=not is_sparse and torch.cuda.is_available() and num_workers_dl > 0,
        persistent_workers=True if num_workers_dl > 0 else False,
        prefetch_factor=2 if num_workers_dl > 0 else None,
    )

    return dataloader
