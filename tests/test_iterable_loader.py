import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader

from cellarr_array.dataloaders.iterabledataloader import (
    CellArrayIterableDataset,
    dense_batch_collate_fn,
    sparse_batch_collate_fn,
)
from cellarr_array.dataloaders.utils import seed_worker

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class TestCellArrayIterableDataset:
    def test_dense_iterable_dataset_basic(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset_batch_size = 16
        num_yields = 5

        dataset = CellArrayIterableDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            is_sparse=False,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields,
            cellarr_ctx_config=tiledb_ctx_config,
        )

        count = 0
        for batch_data in dataset:
            assert isinstance(batch_data, np.ndarray)
            assert batch_data.shape == (dataset_batch_size, dense_array_params["cols"])
            assert batch_data.dtype == dense_array_params["attr_dtype"]
            count += 1
        assert count == num_yields

    def test_sparse_iterable_dataset_basic(self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        dataset_batch_size = 20
        num_yields = 4

        dataset = CellArrayIterableDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            is_sparse=True,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields,
            cellarr_ctx_config=tiledb_ctx_config,
        )

        count = 0
        for batch_data in dataset:
            assert isinstance(batch_data, sp.spmatrix)
            assert batch_data.shape == (dataset_batch_size, sparse_array_params["cols"])
            assert batch_data.dtype == sparse_array_params["attr_dtype"]
            count += 1
        assert count == num_yields

    def test_dense_iterable_dataloader_iteration(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset_batch_size = 16
        num_yields_per_worker = 5
        num_workers = 0

        dataset = CellArrayIterableDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            is_sparse=False,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields_per_worker,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            collate_fn=dense_batch_collate_fn,
        )

        num_batches_processed = 0
        total_cells_in_batches = 0
        for torch_batch in dataloader:
            assert isinstance(torch_batch, torch.Tensor)
            assert torch_batch.shape == (dataset_batch_size, dense_array_params["cols"])
            assert torch_batch.dtype == torch.float32
            num_batches_processed += 1
            total_cells_in_batches += torch_batch.shape[0]

        assert num_batches_processed == num_yields_per_worker
        assert total_cells_in_batches == num_yields_per_worker * dataset_batch_size

    def test_sparse_iterable_dataloader_iteration(
        self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config
    ):
        dataset_batch_size = 20
        num_yields_per_worker = 4
        num_workers = 0

        dataset = CellArrayIterableDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            is_sparse=True,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields_per_worker,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            collate_fn=sparse_batch_collate_fn,
        )

        num_batches_processed = 0
        for torch_batch in dataloader:
            assert isinstance(torch_batch, torch.Tensor)
            assert torch_batch.is_sparse
            assert torch_batch.layout == torch.sparse_coo
            assert torch_batch.shape == (dataset_batch_size, sparse_array_params["cols"])
            if sparse_array_params["attr_dtype"] == np.float64:
                assert torch_batch.dtype == torch.float64
            num_batches_processed += 1

        assert num_batches_processed == num_yields_per_worker

    def test_dense_iterable_dataloader_multiprocessing(
        self, created_dense_array_uri, dense_array_params, tiledb_ctx_config
    ):
        dataset_batch_size = 10
        num_yields_per_worker = 3
        num_workers = 2  # Test with multiple workers

        dataset = CellArrayIterableDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            is_sparse=False,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields_per_worker,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            collate_fn=dense_batch_collate_fn,
            persistent_workers=True,
        )

        num_batches_processed = 0
        total_samples_processed = 0
        for torch_batch in dataloader:
            assert torch_batch.shape == (dataset_batch_size, dense_array_params["cols"])
            num_batches_processed += 1
            total_samples_processed += torch_batch.shape[0]

        assert num_batches_processed == num_yields_per_worker * num_workers
        assert total_samples_processed == num_yields_per_worker * num_workers * dataset_batch_size

    def test_sparse_iterable_dataloader_multiprocessing(
        self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config
    ):
        dataset_batch_size = 15
        num_yields_per_worker = 2
        num_workers = 2

        dataset = CellArrayIterableDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            is_sparse=True,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields_per_worker,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            collate_fn=sparse_batch_collate_fn,
            persistent_workers=True,
        )

        num_batches_processed = 0
        for torch_batch in dataloader:
            assert torch_batch.is_sparse
            assert torch_batch.shape == (dataset_batch_size, sparse_array_params["cols"])
            num_batches_processed += 1

        assert num_batches_processed == num_yields_per_worker * num_workers

    def test_iterable_dataset_empty_array(self, empty_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset_batch_size = 10
        num_yields = 5

        dataset = CellArrayIterableDataset(
            array_uri=empty_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=0,  # Array has 0 rows
            num_columns=dense_array_params["cols"],
            is_sparse=False,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields,
            cellarr_ctx_config=tiledb_ctx_config,
        )

        batches_yielded = 0
        for batch_data in dataset:
            assert isinstance(batch_data, np.ndarray)
            assert batch_data.shape == (0, dense_array_params["cols"])
            batches_yielded += 1

        assert batches_yielded == 1

        dataloader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=dense_batch_collate_fn,
        )

        dl_batches = 0
        for torch_batch in dataloader:
            assert torch_batch.shape == (0, dense_array_params["cols"])
            dl_batches += 1
        assert dl_batches == 1

    def test_iterable_dataset_small_array_full_iteration(
        self, created_dense_array_uri, dense_array_params, tiledb_ctx_config
    ):
        dataset_batch_size = 10
        num_yields_to_cover_once = dense_array_params["rows"] // dataset_batch_size

        dataset_once = CellArrayIterableDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            is_sparse=False,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=num_yields_to_cover_once,
            cellarr_ctx_config=tiledb_ctx_config,
        )

        batches_count = 0
        for batch_np in dataset_once:
            assert batch_np.shape == (dataset_batch_size, dense_array_params["cols"])
            batches_count += 1
        assert batches_count == num_yields_to_cover_once

    def test_iterable_default_num_yields(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset_batch_size = 10
        expected_yields = dense_array_params["rows"] // dataset_batch_size
        if expected_yields == 0 and dense_array_params["rows"] > 0:  # rows = 100, batch_size=10 -> 10
            expected_yields = 1

        dataset = CellArrayIterableDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            is_sparse=False,
            batch_size=dataset_batch_size,
            num_yields_per_epoch_per_worker=None,
            cellarr_ctx_config=tiledb_ctx_config,
        )

        assert dataset.num_yields_per_epoch_per_worker == expected_yields

        count = 0
        for _ in dataset:
            count += 1
        assert count == expected_yields
