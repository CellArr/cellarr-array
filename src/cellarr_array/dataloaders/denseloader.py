from typing import Optional
from warnings import warn

import numpy as np
import tiledb
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.dense import DenseCellArray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class DenseArrayDataset(Dataset):
    def __init__(
        self,
        array_uri: str,
        attribute_name: str = "data",
        num_rows: Optional[int] = None,
        num_columns: Optional[int] = None,
        cellarr_ctx_config: Optional[dict] = None,
        transform=None,
    ):
        """PyTorch Dataset for dense TileDB arrays accessed via DenseCellArray.

        Args:
            array_uri:
                URI of the TileDB dense array.

            attribute_name:
                Name of the attribute to read from.

            num_rows:
                Total number of rows in the dataset.
                If None, will infer from `array.shape[0]`.

            num_columns:
                The number of columns in the dataset.
                If None, will attempt to infer `from array.shape[1]`.

            cellarr_ctx_config:
                Optional TileDB context configuration dict for CellArray.

            transform:
                Optional transform to be applied on a sample.
        """
        self.array_uri = array_uri
        self.attribute_name = attribute_name
        self.cellarr_ctx_config = cellarr_ctx_config
        self.transform = transform
        self.cell_array_instance = None

        if num_rows is not None and num_columns is not None:
            self._len = num_rows
            self.num_columns = num_columns
        else:
            # Infer the array shape
            print(f"Dataset '{array_uri}': num_rows or num_columns not provided. Probing array...")
            init_ctx_config = tiledb.Config(self.cellarr_ctx_config) if self.cellarr_ctx_config else None
            try:
                temp_arr = DenseCellArray(
                    uri=self.array_uri, attr=self.attribute_name, config_or_context=init_ctx_config
                )
                if temp_arr.ndim == 1:
                    self._len = num_rows if num_rows is not None else temp_arr.shape[0]
                    self.num_columns = 1
                elif temp_arr.ndim == 2:
                    self._len = num_rows if num_rows is not None else temp_arr.shape[0]
                    self.num_columns = num_columns if num_columns is not None else temp_arr.shape[1]
                else:
                    raise ValueError(f"Array ndim {temp_arr.ndim} not supported.")

                print(f"Dataset '{array_uri}': Inferred shape. Rows: {self._len}, Columns: {self.num_columns}")

            except Exception as e:
                if num_rows is None or num_columns is None:
                    raise ValueError(
                        f"num_rows and num_columns must be provided if inferring array shape fails for '{array_uri}'."
                    ) from e
                self._len = num_rows
                self.feature_dim = num_columns
                warn(
                    f"Falling back to provided or zero dimensions for '{array_uri}' due to inference error: {e}",
                    RuntimeWarning,
                )

        if self.num_columns is None or self.num_columns <= 0 and self._len > 0:  # Check if num_columns is valid
            raise ValueError(
                f"num_columns ({self.num_columns}) is invalid or could not be determined for array '{array_uri}'."
            )

        if self._len == 0:
            warn(f"Dataset for '{array_uri}' has length 0.", RuntimeWarning)

    def _init_worker_state(self):
        """Initializes the DenseCellArray instance for the current worker."""
        if self.cell_array_instance is None:
            ctx = tiledb.Ctx(self.cellarr_ctx_config) if self.cellarr_ctx_config else None
            self.cell_array_instance = DenseCellArray(
                uri=self.array_uri, attr=self.attribute_name, mode="r", config_or_context=ctx
            )

            # Sanity check: worker's shape against dataset's established dims
            # if self.cell_array_instance.shape[0] != self._len or \
            #    (self.cell_array_instance.ndim > 1 and self.cell_array_instance.shape[1] != self.feature_dim) or \
            #    (self.cell_array_instance.ndim == 1 and self.feature_dim != 1) :
            #     print(f"Warning: Worker for {self.array_uri} sees shape {self.cell_array_instance.shape} "
            #           f"but dataset initialized with len={self._len}, feat={self.feature_dim}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if not 0 <= idx < self._len:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._len}.")

        self._init_worker_state()

        if self.cell_array_instance.ndim == 2:
            item_slice = (slice(idx, idx + 1), slice(None))
        elif self.cell_array_instance.ndim == 1:
            item_slice = slice(idx, idx + 1)
        else:
            raise ValueError(f"Array ndim {self.cell_array_instance.ndim} not supported in __getitem__.")

        sample_data_np = self.cell_array_instance[item_slice]
        if sample_data_np.ndim == 2 and sample_data_np.shape[0] == 1:
            sample_data_np = sample_data_np.squeeze(0)
        elif sample_data_np.ndim == 1 and sample_data_np.shape[0] == 1 and self.feature_dim == 1:
            pass
        elif sample_data_np.ndim == 0 and self.feature_dim == 1:
            sample_data_np = np.array([sample_data_np])

        if self.transform:
            sample_data_np = self.transform(sample_data_np)

        return torch.from_numpy(sample_data_np)


def construct_dense_array_dataloader(
    array_uri: str,
    attribute_name: str = "data",
    num_rows: Optional[int] = None,
    num_columns: Optional[int] = None,
    batch_size: int = 1000,
    num_workers_dl: int = 2,
) -> DataLoader:
    """Construct an instance of `DenseArrayDataset` with PyTorch DataLoader.

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
        "sm.tile_cache_size": 1000 * 1024**2,  # 1000 MB tile cache per worker
        "sm.num_reader_threads": 4,
    }

    dataset = DenseArrayDataset(
        array_uri=array_uri,
        attribute_name=attribute_name,
        num_rows=num_rows,
        num_columns=num_columns,
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
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if num_workers_dl > 0 else False,
    )

    return dataloader
