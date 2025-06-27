import random
import time

import numpy as np
import torch
from scipy import sparse as sp
from torch.utils.data import DataLoader

from cellarr_array.dataloaders import CellArrayIterableDataset, DenseArrayDataset, SparseArrayDataset
from cellarr_array.dataloaders.iterabledataloader import dense_batch_collate_fn, sparse_batch_collate_fn
from cellarr_array.dataloaders.sparseloader import sparse_coo_collate_fn
from cellarr_array.dataloaders.utils import seed_worker
from cellarr_array.utils.mock import generate_tiledb_dense_array, generate_tiledb_sparse_array


def benchmark_dataloader(
    dataloader: DataLoader,
    num_epochs: int,
    num_batches_to_iterate_per_epoch: int,
    description: str,
    is_iterable: bool = False,
    cells_per_iterable_batch: int = 1,
):
    """Benchmarks the DataLoader by iterating through a specified number of batches.

    Args:
        dataloader:
            The PyTorch DataLoader instance to benchmark.

        num_epochs:
            Number of epochs to run the benchmark for.

        num_batches_to_iterate_per_epoch:
            Max number of batches to iterate in each epoch.
            For IterableDataset, this is total batches from all workers.

        description:
            Description of the DataLoader configuration being benchmarked.

        is_iterable:
            True if the dataloader uses an IterableDataset that yields full batches.

        cells_per_iterable_batch:
            For IterableDataset, how many cells are in each yielded batch.
            For MapDataset, this is effectively the dataloader.batch_size.
    """
    print(f"\n--- Benchmarking: {description} ---")
    total_cells_processed = 0
    total_time_taken = 0

    iterate_all = num_batches_to_iterate_per_epoch == -1

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        cells_in_epoch = 0
        batches_in_epoch = 0

        actual_max_batches = 0  # Initialize
        if iterate_all:
            if is_iterable and hasattr(dataloader.dataset, "num_yields_per_epoch_per_worker"):
                num_yields_per_worker = dataloader.dataset.num_yields_per_epoch_per_worker
                num_workers_for_calc = (
                    dataloader.num_workers if dataloader.num_workers and dataloader.num_workers > 0 else 1
                )
                actual_max_batches = num_workers_for_calc * num_yields_per_worker
                if actual_max_batches == 0 and dataloader.dataset.total_cells_in_array > 0:
                    actual_max_batches = 1
                print(f"  (Iterable with iterate_all=True: Expecting up to {actual_max_batches} batches this epoch)")
            elif not is_iterable:  # Map-style
                if len(dataloader) > 0:  # len(dataloader) is only valid if dataset is not empty
                    actual_max_batches = len(dataloader)
                    print(f"  (Map-style with iterate_all=True: Expecting {actual_max_batches} batches this epoch)")
                else:
                    actual_max_batches = 0
                    print(f"  (Map-style with iterate_all=True: DataLoader length is 0)")
            else:
                actual_max_batches = float("inf")
                print(f"  (Iterable with iterate_all=True and unknown length: Iterating until StopIteration)")
        else:
            actual_max_batches = num_batches_to_iterate_per_epoch

        if actual_max_batches == 0:
            print(f"  Epoch {epoch+1}: No batches to iterate based on configuration. Skipping epoch.")
            continue

        for i, data_batch in enumerate(dataloader):
            if i >= actual_max_batches:
                break

            if hasattr(data_batch, "shape"):
                cells_in_this_batch = data_batch.shape[0]
            else:
                cells_in_this_batch = cells_per_yielded_batch

            cells_in_epoch += cells_in_this_batch
            batches_in_epoch += 1

            if isinstance(data_batch, torch.Tensor):
                if data_batch.numel() > 0:
                    _ = data_batch.device

            if i % 20 == 0 and i > 0 and batches_in_epoch < actual_max_batches:
                print(
                    f"    Epoch {epoch+1}, Batch {batches_in_epoch}/{'all' if actual_max_batches == float('inf') else actual_max_batches}: Shape {data_batch.shape if hasattr(data_batch, 'shape') else 'N/A'}"
                )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_time_taken += epoch_duration
        total_cells_processed += cells_in_epoch

        cells_per_sec_epoch = (cells_in_epoch / epoch_duration) if epoch_duration > 1e-6 else float("inf")

        print(
            f"  Epoch {epoch+1} finished: {batches_in_epoch} batches, {cells_in_epoch} cells in {epoch_duration:.3f}s. "
            f"cells/sec: {cells_per_sec_epoch:.2f}"
        )

    avg_time_per_epoch = total_time_taken / num_epochs if num_epochs > 0 else 0
    avg_cells_per_sec = total_cells_processed / total_time_taken if total_time_taken > 1e-6 else float("inf")
    print(f"--- Benchmark Summary for '{description}' ---")
    print(f"  Total cells processed: {total_cells_processed} over {num_epochs} epochs.")
    print(f"  Average time per epoch: {avg_time_per_epoch:.3f}s.")
    print(f"  Average throughput: {avg_cells_per_sec:.2f} cells/sec.")
    print("--------------------------------------------")
    return avg_cells_per_sec


def run_benchmark(num_rows=50000, num_cols=256):
    torch.manual_seed(0)  # For reproducibility of DataLoader shuffle if used
    np.random.seed(0)
    random.seed(0)

    # --- Configuration ---
    DENSE_ARRAY_URI = "benchmark_dense_array.tdb"
    SPARSE_ARRAY_URI = "benchmark_sparse_array.tdb"

    NUM_ROWS_DENSE = num_rows
    NUM_COLS_DENSE = num_cols
    DENSE_CHUNK_WRITE = 2000

    NUM_ROWS_SPARSE = num_rows
    NUM_COLS_SPARSE = num_cols
    SPARSE_DENSITY = 0.020  # Lower density for larger sparse array
    SPARSE_CHUNK_WRITE = 5000

    # DataLoader/Benchmark parameters
    BENCH_NUM_EPOCHS = 2
    # How many batches the dataloader should attempt to iterate.
    # For IterableDataset, this is the total from all workers for one pass.
    # For MapDataset, this is num_cells / batch_size.
    # Set to -1 to try to iterate through the whole conceptual "epoch"
    BENCH_MAX_BATCHES_PER_EPOCH = 50  # Iterate up to 50 batches from the dataloader per epoch for quick bench

    MAP_STYLE_BATCH_SIZE = 128
    ITERABLE_STYLE_BATCH_SIZE_N = 128  # N random cells per yielded batch

    # For a fair comparison, num_workers should be the same if possible
    # However, IterableDataset with batch_size=None in DataLoader means workers generate full batches.
    # MapDataset with batch_size=X means workers generate individual cells, then collated.
    NUM_WORKERS = 4  # Use 0 for debugging, >0 for performance. Ensure this is reasonable for your system.

    # TileDB Context config for readers
    tiledb_ctx_config = {
        "sm.tile_cache_size": str(50 * 1024**2),  # 50MB tile cache per worker
        # "sm.num_reader_threads": "2", # TileDB context values are usually strings
    }

    # --- Generate Arrays (do this once) ---
    # generate_tiledb_dense_array(DENSE_ARRAY_URI, NUM_ROWS_DENSE, NUM_COLS_DENSE, chunk_size=DENSE_CHUNK_WRITE)
    # generate_tiledb_sparse_array(
    #     SPARSE_ARRAY_URI, NUM_ROWS_SPARSE, NUM_COLS_SPARSE, density=SPARSE_DENSITY, chunk_size=SPARSE_CHUNK_WRITE
    # )
    print("Using MOCK arrays for this benchmark run. To use real TileDB arrays, uncomment generation lines.")

    # --- Dense Array Benchmarks ---
    print("\n" + "=" * 30 + " DENSE ARRAY BENCHMARKS " + "=" * 30)
    # 1. Dense Map-Style DataLoader
    dense_map_dataset = DenseArrayDataset(
        array_uri=DENSE_ARRAY_URI,  # Mock will use its internal shape
        num_rows=NUM_ROWS_DENSE,  # Mock will use its internal shape
        num_columns=NUM_COLS_DENSE,  # Mock will use its internal shape
        cellarr_ctx_config=tiledb_ctx_config,
    )
    dense_map_dataloader = DataLoader(
        dense_map_dataset,
        batch_size=MAP_STYLE_BATCH_SIZE,
        shuffle=True,  # Standard for map-style training
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    benchmark_dataloader(
        dense_map_dataloader,
        BENCH_NUM_EPOCHS,
        BENCH_MAX_BATCHES_PER_EPOCH,
        f"Dense Map-Style (Batch Size: {MAP_STYLE_BATCH_SIZE}, Workers: {NUM_WORKERS})",
        cells_per_iterable_batch=MAP_STYLE_BATCH_SIZE,
    )

    # 2. Dense Iterable-Style DataLoader
    # For iterable, num_yields_per_worker needs to be set for an epoch definition
    # Let's aim for roughly the same number of total cells processed as map-style benchmark for comparison.
    # total_cells_to_process_map = BENCH_MAX_BATCHES_PER_EPOCH * MAP_STYLE_BATCH_SIZE
    # num_yields_iterable = (total_cells_to_process_map + ITERABLE_STYLE_BATCH_SIZE_N -1) // ITERABLE_STYLE_BATCH_SIZE_N
    # num_yields_per_worker_dense = (num_yields_iterable + NUM_WORKERS - 1) // NUM_WORKERS if NUM_WORKERS > 0 else num_yields_iterable
    # This calculation ensures iterable processes a similar amount of data for comparison
    # Or simply use a fixed number of yields like BENCH_MAX_BATCHES_PER_EPOCH refers to total batches from dataloader
    num_total_batches_for_iterable_dense = BENCH_MAX_BATCHES_PER_EPOCH
    dense_iterable_num_yields = (
        (num_total_batches_for_iterable_dense + NUM_WORKERS - 1) // NUM_WORKERS
        if NUM_WORKERS > 0
        else num_total_batches_for_iterable_dense
    )

    dense_iterable_dataset = CellArrayIterableDataset(
        array_uri=DENSE_ARRAY_URI,
        attribute_name="data",
        num_rows=NUM_ROWS_DENSE,  # Mock uses its own
        num_columns=NUM_COLS_DENSE,  # Mock uses its own
        is_sparse=False,
        batch_size=ITERABLE_STYLE_BATCH_SIZE_N,
        num_yields_per_epoch_per_worker=dense_iterable_num_yields,
        cellarr_ctx_config=tiledb_ctx_config,
    )
    dense_iterable_dataloader = DataLoader(
        dense_iterable_dataset,
        batch_size=None,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        collate_fn=dense_batch_collate_fn,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    benchmark_dataloader(
        dense_iterable_dataloader,
        BENCH_NUM_EPOCHS,
        num_total_batches_for_iterable_dense,
        f"Dense Iterable-Style (N per Batch: {ITERABLE_STYLE_BATCH_SIZE_N}, Workers: {NUM_WORKERS})",
        is_iterable=True,
        cells_per_iterable_batch=ITERABLE_STYLE_BATCH_SIZE_N,
    )

    # --- Sparse Array Benchmarks ---
    print("\n" + "=" * 30 + " SPARSE ARRAY BENCHMARKS " + "=" * 30)
    # 1. Sparse Map-Style DataLoader
    sparse_map_dataset = SparseArrayDataset(
        array_uri=SPARSE_ARRAY_URI,
        attribute_name="data",
        num_rows=NUM_ROWS_SPARSE,  # Mock uses its own
        num_columns=NUM_COLS_SPARSE,  # Mock uses its own
        sparse_format=sp.coo_matrix,  # Get COO for mapstyle_sparse_collate_fn
        cellarr_ctx_config=tiledb_ctx_config,
    )
    sparse_map_dataloader = DataLoader(
        sparse_map_dataset,
        batch_size=MAP_STYLE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        collate_fn=sparse_coo_collate_fn,  # Custom collate for list of individual sparse cells
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    benchmark_dataloader(
        sparse_map_dataloader,
        BENCH_NUM_EPOCHS,
        BENCH_MAX_BATCHES_PER_EPOCH,
        f"Sparse Map-Style (Batch Size: {MAP_STYLE_BATCH_SIZE}, Workers: {NUM_WORKERS})",
        cells_per_iterable_batch=MAP_STYLE_BATCH_SIZE,
    )

    # 2. Sparse Iterable-Style DataLoader
    num_total_batches_for_iterable_sparse = BENCH_MAX_BATCHES_PER_EPOCH
    sparse_iterable_num_yields = (
        (num_total_batches_for_iterable_sparse + NUM_WORKERS - 1) // NUM_WORKERS
        if NUM_WORKERS > 0
        else num_total_batches_for_iterable_sparse
    )

    sparse_iterable_dataset = CellArrayIterableDataset(
        array_uri=SPARSE_ARRAY_URI,
        attribute_name="data",
        num_rows=NUM_ROWS_SPARSE,  # Mock uses its own
        num_columns=NUM_COLS_SPARSE,  # Mock uses its own
        is_sparse=True,
        batch_size=ITERABLE_STYLE_BATCH_SIZE_N,
        num_yields_per_epoch_per_worker=sparse_iterable_num_yields,
        cellarr_ctx_config=tiledb_ctx_config,
    )
    sparse_iterable_dataloader = DataLoader(
        sparse_iterable_dataset,
        batch_size=None,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        collate_fn=sparse_batch_collate_fn,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    benchmark_dataloader(
        sparse_iterable_dataloader,
        BENCH_NUM_EPOCHS,
        num_total_batches_for_iterable_sparse,
        f"Sparse Iterable-Style (N per Batch: {ITERABLE_STYLE_BATCH_SIZE_N}, Workers: {NUM_WORKERS})",
        is_iterable=True,
        cells_per_iterable_batch=ITERABLE_STYLE_BATCH_SIZE_N,
    )

    print("\nBenchmark run complete.")
    # Consider cleaning up dummy arrays if they were actually created:
    # if os.path.exists(DENSE_ARRAY_URI): shutil.rmtree(DENSE_ARRAY_URI)
    # if os.path.exists(SPARSE_ARRAY_URI): shutil.rmtree(SPARSE_ARRAY_URI)


if __name__ == "__main__":
    run_benchmark()
