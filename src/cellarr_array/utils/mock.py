import shutil
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import tiledb

from ..core import DenseCellArray, SparseCellArray
from ..core.helpers import CellArrConfig, create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def generate_tiledb_dense_array(
    uri: str,
    rows: int,
    cols: int,
    attr_name: str = "data",
    attr_dtype: np.dtype = np.float32,
    chunk_size: int = 1000,
    tiledb_config: Optional[Dict] = None,
):
    """Generates a dense TileDB array and fills it with random float32 data.

    Args:
        uri:
            URI for the new TileDB array.

        rows:
            Number of rows.

        cols:
            Number of columns (features).

        attr_name:
            Name of the attribute.

        attr_dtype:
            Data type of the attribute.

        chunk_size:
            Number of rows to write per batch.

        tiledb_config:
            TileDB context configuration.
    """
    if tiledb.array_exists(uri):
        print(f"Array {uri} already exists. Removing.")
        shutil.rmtree(uri)

    print(f"Creating dense array at '{uri}' with shape ({rows}, {cols})")
    cfg = CellArrConfig(ctx_config=tiledb_config if tiledb_config else {})

    create_cellarray(
        uri=uri,
        shape=(rows, cols),
        attr_dtype=attr_dtype,
        sparse=False,
        dim_names=["rows", "cols"],
        attr_name=attr_name,
        # config=cfg
    )

    ctx = tiledb.Ctx(cfg.ctx_config) if cfg.ctx_config else None
    arr_writer = DenseCellArray(uri=uri, attr=attr_name, mode="w", config_or_context=ctx)

    print("shape of writer", arr_writer.shape)

    print(f"Writing data to dense array '{uri}'...")
    for i in range(0, rows, chunk_size):
        end_row = min(i + chunk_size, rows)
        num_chunk_rows = end_row - i
        data_chunk = np.random.rand(num_chunk_rows, cols).astype(attr_dtype)
        print(i, end_row, num_chunk_rows, data_chunk.shape)
        arr_writer.write_batch(data_chunk, start_row=i)
        if (i // chunk_size) % 10 == 0:
            print(f"  Dense write: {end_row}/{rows} rows written.")

    print(f"Finished writing to dense array '{uri}'.")


def generate_tiledb_sparse_array(
    uri: str,
    rows: int,
    cols: int,
    density: float = 0.01,
    attr_name: str = "data",
    attr_dtype: np.dtype = np.float32,
    chunk_size: int = 1000,
    tiledb_config: Optional[Dict] = None,
    sparse_format_to_write="coo",
):
    """Generates a sparse TileDB array and fills it with random float32 data.

    Args:
        uri:
            URI for the new TileDB array.

        rows:
            Number of rows.

        cols:
            Number of columns (features).

        density:
            Density of the sparse matrix.

        attr_name:
            Name of the attribute.

        attr_dtype:
            Data type of the attribute.

        chunk_size:
            Number of rows to generate and write per batch.

        tiledb_configs:
            TileDB context configuration.

        sparse_format_to_write:
            Scipy sparse format to use for generating chunks ('coo', 'csr', 'csc').

    """
    if tiledb.array_exists(uri):
        print(f"Array {uri} already exists. Removing.")
        shutil.rmtree(uri)

    print(f"Creating sparse array at '{uri}' with shape ({rows}, {cols}), density ~{density}")
    cfg = CellArrConfig(ctx_config=tiledb_config if tiledb_config else {})
    create_cellarray(
        uri=uri,
        shape=(rows, cols),
        attr_dtype=attr_dtype,
        sparse=True,
        dim_names=["rows", "cols"],
        attr_name=attr_name,
        # config=cfg
    )

    ctx = tiledb.Ctx(cfg.ctx_config) if cfg.ctx_config else None
    arr_writer = SparseCellArray(
        uri=uri,
        attr=attr_name,
        mode="w",
        config_or_context=ctx,
    )

    print(f"Writing data to sparse array '{uri}'...")
    for i in range(0, rows, chunk_size):
        end_row = min(i + chunk_size, rows)
        num_chunk_rows = end_row - i
        if num_chunk_rows <= 0:
            continue

        data_chunk_scipy = sp.random(
            num_chunk_rows, cols, density=density, format=sparse_format_to_write, dtype=attr_dtype
        )

        if data_chunk_scipy.nnz > 0:
            arr_writer.write_batch(data_chunk_scipy, start_row=i)

        if (i // chunk_size) % 10 == 0:
            print(f"  Sparse write: {end_row}/{rows} rows processed for writing.")

    print(f"Finished writing to sparse array '{uri}'.")
