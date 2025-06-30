from typing import Optional
from warnings import warn

import scipy.sparse as sp
import tiledb
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.sparse import SparseCellArray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class SparseArrayDataset(Dataset):
    def __init__(
        self,
        array_uri: str,
        attribute_name: str = "data",
        num_rows: Optional[int] = None,
        num_columns: Optional[int] = None,
        sparse_format=sp.csr_matrix,
        cellarr_ctx_config: Optional[dict] = None,
        transform=None,
    ):
        """PyTorch Dataset for sparse TileDB arrays accessed via SparseCellArray.

        Args:
            array_uri:
                URI of the TileDB sparse array.

            attribute_name:
                Name of the attribute to read from.

            num_rows:
                Total number of rows in the dataset.
                If None, will infer from `array.shape[0]`.

            num_columns:
                The number of columns in the dataset.
                If None, will attempt to infer `from array.shape[1]`.

            sparse_format:
                Format to return, defaults to csr_matrix.

            cellarr_ctx_config:
                Optional TileDB context configuration dict for CellArray.

            transform:
                Optional transform to be applied on a sample.
        """
        self.array_uri = array_uri
        self.attribute_name = attribute_name
        self.sparse_format = sparse_format
        self.cellarr_ctx_config = cellarr_ctx_config
        self.transform = transform
        self.cell_array_instance = None

        if num_rows is not None and num_columns is not None:
            self._len = num_rows
            self.num_columns = num_columns
        else:
            print(f"Dataset '{array_uri}': num_rows or num_columns not provided. Probing sparse array...")
            init_ctx_config = tiledb.Config(self.cellarr_ctx_config) if self.cellarr_ctx_config else None
            try:
                temp_arr = SparseCellArray(
                    uri=self.array_uri,
                    attr=self.attribute_name,
                    config_or_context=init_ctx_config,
                    return_sparse=True,
                    sparse_format=self.sparse_format,
                )

                if temp_arr.ndim == 1:
                    self._len = num_rows if num_rows is not None else temp_arr.shape[0]
                    self.num_columns = 1
                elif temp_arr.ndim == 2:
                    self._len = num_rows if num_rows is not None else temp_arr.shape[0]
                    self.num_columns = num_columns if num_columns is not None else temp_arr.shape[1]
                else:
                    raise ValueError(f"Array ndim {temp_arr.ndim} not supported.")

                print(f"Dataset '{array_uri}': Inferred sparse shape. Rows: {self._len}, Columns: {self.num_columns}")

            except Exception as e:
                if num_rows is None or num_columns is None:
                    raise ValueError(
                        f"num_rows and num_columns must be provided if inferring sparse array shape fails for '{array_uri}'. Original error: {e}"
                    ) from e
                self._len = num_rows if num_rows is not None else 0
                self.num_columns = num_columns if num_columns is not None else 0
                warn(
                    f"Falling back to provided or zero dimensions for sparse '{array_uri}' due to inference error: {e}",
                    RuntimeWarning,
                )

        if self.num_columns is None or self.num_columns <= 0 and self._len > 0:
            raise ValueError(
                f"num_columns ({self.num_columns}) is invalid or could not be determined for sparse array '{array_uri}'."
            )

        if self._len == 0:
            warn(f"SparseDataset for '{array_uri}' has length 0.", RuntimeWarning)

    def _init_worker_state(self):
        if self.cell_array_instance is None:
            ctx = tiledb.Ctx(self.cellarr_ctx_config) if self.cellarr_ctx_config else None
            self.cell_array_instance = SparseCellArray(
                uri=self.array_uri,
                attr=self.attribute_name,
                mode="r",
                config_or_context=ctx,
                return_sparse=True,
                sparse_coerce=self.sparse_format,
            )

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if not 0 <= idx < self._len:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._len}.")

        self._init_worker_state()

        item_slice = (slice(idx, idx + 1), slice(None))

        scipy_sparse_sample = self.cell_array_instance[item_slice]

        if self.transform:  # e.g., convert to COO for easier collation
            scipy_sparse_sample = self.transform(scipy_sparse_sample)

        if not isinstance(scipy_sparse_sample, sp.coo_matrix):
            scipy_sparse_sample = scipy_sparse_sample.tocoo()

        return scipy_sparse_sample


def sparse_coo_collate_fn(batch):
    """Custom collate_fn for a batch of SciPy COO sparse matrices.

    Converts them into a single batched PyTorch sparse COO tensor.

    Each item in 'batch' is a SciPy coo_matrix representing one sample.
    """
    all_data = []
    all_row_indices = []
    all_col_indices = []

    for i, scipy_coo in enumerate(batch):
        if scipy_coo.nnz > 0:
            all_data.append(torch.from_numpy(scipy_coo.data))
            all_row_indices.append(torch.full_like(torch.from_numpy(scipy_coo.row), fill_value=i, dtype=torch.long))
            all_col_indices.append(torch.from_numpy(scipy_coo.col))

    if not all_data:
        num_columns = batch[0].shape[1] if batch else 0
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (len(batch), num_columns))

    data_cat = torch.cat(all_data)
    row_indices_cat = torch.cat(all_row_indices)
    col_indices_cat = torch.cat(all_col_indices)

    indices = torch.stack([row_indices_cat, col_indices_cat], dim=0)
    num_columns = batch[0].shape[1]
    batch_size = len(batch)

    sparse_tensor = torch.sparse_coo_tensor(indices, data_cat, (batch_size, num_columns))
    return sparse_tensor


def construct_sparse_array_dataloader(
    array_uri: str,
    attribute_name: str = "data",
    num_rows: Optional[int] = None,
    num_columns: Optional[int] = None,
    batch_size: int = 1000,
    num_workers_dl: int = 2,
) -> DataLoader:
    """Construct an instance of `SparseArrayDataset` with PyTorch DataLoader.

    Args:
        array_uri:
            URI of the TileDB array.

        attribute_name:
            Name of the attribute to read from.

        num_rows:
            The total number of rows in the TileDB array.

        num_columns:
            The total number of columns in the TileDB array.

        batch_size:
            Number of random samples per batch generated by the dataset.

        num_workers_dl:
            Number of worker processes for the DataLoader.
    """
    tiledb_ctx_config = {
        "sm.tile_cache_size": 1000 * 1024**2,
        "sm.num_reader_threads": 4,
    }

    dataset = SparseArrayDataset(
        array_uri=array_uri,
        attribute_name=attribute_name,
        num_rows=num_rows,
        num_columns=num_columns,
        sparse_format=sp.coo_matrix,
        cellarr_ctx_config=tiledb_ctx_config,
    )

    if len(dataset) == 0:
        print("Dataset is empty, cannot create DataLoader.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_dl,
        collate_fn=sparse_coo_collate_fn,
        pin_memory=False,
        persistent_workers=True if num_workers_dl > 0 else False,
    )

    return dataloader
