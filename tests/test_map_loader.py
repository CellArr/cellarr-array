import numpy as np
import pytest
import scipy.sparse as sp
import tiledb
import torch
from torch.utils.data import DataLoader

from cellarr_array.core import DenseCellArray, SparseCellArray
from cellarr_array.dataloaders.denseloader import DenseArrayDataset
from cellarr_array.dataloaders.sparseloader import SparseArrayDataset, sparse_coo_collate_fn
from cellarr_array.dataloaders.utils import seed_worker

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class TestDenseArrayDataset:
    def test_dataset_initialization(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset = DenseArrayDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            cellarr_ctx_config=tiledb_ctx_config,
        )
        assert len(dataset) == dense_array_params["rows"]
        assert dataset.num_columns == dense_array_params["cols"]

    def test_dataset_infer_shape(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset = DenseArrayDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            cellarr_ctx_config=tiledb_ctx_config,
        )
        assert len(dataset) == dense_array_params["rows"]
        assert dataset.num_columns == dense_array_params["cols"]

    def test_getitem(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset = DenseArrayDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            cellarr_ctx_config=tiledb_ctx_config,
        )
        sample = dataset[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (dense_array_params["cols"],)
        assert sample.dtype == torch.float32

        sample_last = dataset[dense_array_params["rows"] - 1]
        assert isinstance(sample_last, torch.Tensor)

        with pytest.raises(IndexError):
            _ = dataset[dense_array_params["rows"]]

    def test_dataloader_iteration_dense(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        batch_size = 4
        dataset = DenseArrayDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
        )

        num_batches = 0
        total_samples = 0
        for batch_data in dataloader:
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.shape[0] <= batch_size
            assert batch_data.shape[1] == dense_array_params["cols"]
            assert batch_data.dtype == torch.float32
            num_batches += 1
            total_samples += batch_data.shape[0]

        assert total_samples == dense_array_params["rows"]
        assert num_batches == (dense_array_params["rows"] + batch_size - 1) // batch_size

    def test_dataloader_multiprocessing_dense(self, created_dense_array_uri, dense_array_params, tiledb_ctx_config):
        batch_size = 4
        num_workers = 2
        dataset = DenseArrayDataset(
            array_uri=created_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=dense_array_params["rows"],
            num_columns=dense_array_params["cols"],
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            persistent_workers=True,
        )

        total_samples = 0
        for batch_data in dataloader:
            assert batch_data.shape[0] <= batch_size
            assert batch_data.shape[1] == dense_array_params["cols"]
            total_samples += batch_data.shape[0]
        assert total_samples == dense_array_params["rows"]

    def test_empty_dense_dataset(self, empty_dense_array_uri, dense_array_params, tiledb_ctx_config):
        dataset = DenseArrayDataset(
            array_uri=empty_dense_array_uri,
            attribute_name=dense_array_params["attr_name"],
            num_rows=0,
            num_columns=dense_array_params["cols"],
            cellarr_ctx_config=tiledb_ctx_config,
        )
        assert len(dataset) == 0
        with pytest.raises(IndexError):
            _ = dataset[0]

        dataloader = DataLoader(dataset, batch_size=4)
        count = 0
        for _ in dataloader:
            count += 1
        assert count == 0


class TestSparseArrayDataset:
    def test_dataset_initialization_sparse(self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        dataset = SparseArrayDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            sparse_format=sp.coo_matrix,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        assert len(dataset) == sparse_array_params["rows"]
        assert dataset.num_columns == sparse_array_params["cols"]

    def test_dataset_infer_shape_sparse(self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        dataset = SparseArrayDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            sparse_format=sp.csr_matrix,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        assert len(dataset) == sparse_array_params["rows"]
        assert dataset.num_columns == sparse_array_params["cols"]

    def test_getitem_sparse(self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        direct_read_tiledb_config_obj = tiledb.Config(tiledb_ctx_config) if tiledb_ctx_config else None

        dataset = SparseArrayDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            sparse_format=sp.coo_matrix,
            cellarr_ctx_config=direct_read_tiledb_config_obj,
        )
        sample = dataset[0]  # Returns a SciPy COO matrix
        assert isinstance(sample, sp.coo_matrix)
        assert sample.shape == (1, sparse_array_params["cols"])
        direct_read_arr = (
            DenseCellArray(
                uri=created_sparse_array_uri,
                attr=sparse_array_params["attr_name"],
                config_or_context=direct_read_tiledb_config_obj,
            )
            if SparseCellArray is None
            else SparseCellArray(
                uri=created_sparse_array_uri,
                attr=sparse_array_params["attr_name"],
                config_or_context=direct_read_tiledb_config_obj,
                # sparse_format=sp.coo_matrix,
            )
        )

        if isinstance(direct_read_arr, SparseCellArray):
            original_row_sparse = direct_read_arr[0:1, :].tocoo()
            assert sample.nnz == original_row_sparse.nnz

            print(sample)
            print(original_row_sparse)

            if sample.nnz > 0:
                assert np.array_equal(sample.data, original_row_sparse.data)
                assert np.array_equal(sample.col, original_row_sparse.col)

        with pytest.raises(IndexError):
            _ = dataset[sparse_array_params["rows"]]

    def test_dataloader_iteration_sparse(self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        batch_size = 8
        dataset = SparseArrayDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            sparse_format=sp.coo_matrix,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=sparse_coo_collate_fn,
            worker_init_fn=seed_worker,
        )

        num_batches = 0
        total_samples = 0
        for batch_data in dataloader:
            assert isinstance(batch_data, torch.Tensor)
            assert batch_data.is_sparse
            assert batch_data.layout == torch.sparse_coo
            assert batch_data.shape[0] <= batch_size
            assert batch_data.shape[1] == sparse_array_params["cols"]

            if sparse_array_params["attr_dtype"] == np.float64:
                assert batch_data.dtype == torch.float64
            elif sparse_array_params["attr_dtype"] == np.float32:
                assert batch_data.dtype == torch.float32

            num_batches += 1
            total_samples += batch_data.shape[0]

        assert total_samples == sparse_array_params["rows"]
        assert num_batches == (sparse_array_params["rows"] + batch_size - 1) // batch_size

    def test_dataloader_multiprocessing_sparse(self, created_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        batch_size = 8
        num_workers = 2
        dataset = SparseArrayDataset(
            array_uri=created_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=sparse_array_params["rows"],
            num_columns=sparse_array_params["cols"],
            sparse_format=sp.coo_matrix,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=sparse_coo_collate_fn,
            worker_init_fn=seed_worker,
            persistent_workers=True if num_workers > 0 else False,
        )

        total_samples = 0
        for batch_data in dataloader:
            assert batch_data.shape[0] <= batch_size
            assert batch_data.shape[1] == sparse_array_params["cols"]
            assert batch_data.is_sparse
            total_samples += batch_data.shape[0]
        assert total_samples == sparse_array_params["rows"]

    def test_empty_sparse_dataset(self, empty_sparse_array_uri, sparse_array_params, tiledb_ctx_config):
        dataset = SparseArrayDataset(
            array_uri=empty_sparse_array_uri,
            attribute_name=sparse_array_params["attr_name"],
            num_rows=0,  # Explicitly 0
            num_columns=sparse_array_params["cols"],
            sparse_format=sp.coo_matrix,
            cellarr_ctx_config=tiledb_ctx_config,
        )
        assert len(dataset) == 0
        with pytest.raises(IndexError):
            _ = dataset[0]

        dataloader = DataLoader(dataset, batch_size=4, collate_fn=sparse_coo_collate_fn)
        count = 0
        for batch_data in dataloader:
            assert batch_data.shape == (0, sparse_array_params["cols"])
            count += 1

        num_batches = 0
        for _ in dataloader:
            num_batches += 1
        assert num_batches == 0  # Since len(dataset) is 0
