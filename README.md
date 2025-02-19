[![PyPI-Server](https://img.shields.io/pypi/v/cellarr-array.svg)](https://pypi.org/project/cellarr-array/)
![Unit tests](https://github.com/cellarr/cellarr-array/actions/workflows/run-tests.yml/badge.svg)

# cellarr-array

This package provided high-level wrappers for TileDB arrays, for handling genomic data matrices.

## Install

To get started, install the package from [PyPI](https://pypi.org/project/cellarr-array/)

```bash
pip install cellarr-array
```

## Quick Start

### Creating Arrays

```python
import numpy as np
from scipy import sparse
from cellarr_array import create_cellarray, CellArrConfig

# Create a dense 2D array
dense_array = create_cellarray(
    uri="dense_matrix.tdb",
    shape=(10000, 5000),
    attr_dtype=np.float32,
    sparse=False,
    dim_names=["cells", "genes"]
)

# Create a sparse 2D array with custom compression
config = CellArrConfig(
    tile_capacity=1000,
    attrs_filters={"data": [{"name": "zstd", "level": 7}]}
)
sparse_array = create_cellarray(
    uri="sparse_matrix.tdb",
    shape=(10000, 5000),
    attr_dtype=np.float32,
    sparse=True,
    config=config,
    dim_names=["cells", "genes"]
)

# Create a 1D array
array_1d = create_cellarray(
    uri="vector.tdb",
    shape=(1000,),
    attr_dtype=np.float32,
    sparse=False
)
```

### Writing Data

```python
# Writing to dense arrays
data = np.random.random((1000, 5000)).astype(np.float32)
dense_array.write_batch(data, start_row=0)

# Writing to sparse arrays
sparse_data = sparse.random(1000, 5000, density=0.1, format="csr", dtype=np.float32)
sparse_array.write_batch(sparse_data, start_row=0)

# Writing to 1D arrays
data_1d = np.random.random(100).astype(np.float32)
array_1d.write_batch(data_1d, start_row=0)
```

### Reading Data

```python
# Slicing operations (similar to NumPy)

# Full slice
full_data = dense_array[:]

# Partial slice
subset = dense_array[100:200, 1000:2000]

# Using lists of indices
cells = [10, 20, 30]
genes = [5, 15, 25]
subset = dense_array[cells, genes]

# Mixed slicing
subset = dense_array[100:200, genes]
```

### Working with Sparse Arrays

```python
from cellarr_array import SparseCellArray

# Create a sparse array with CSR output format
csr_array = SparseCellArray(
    uri="sparse_matrix.tdb",
    return_sparse=True
)

# Get result as CSR matrix
result = csr_array[100:200, 500:1000]

# Result is scipy.sparse.coo_matrix
assert sparse.isspmatrix_csr(result)

# Perform sparse operations
nnz = result.nnz
density = result.nnz / (result.shape[0] * result.shape[1])

# Convert to other sparse formats if needed
result_csc = result.tocsc()
```

Likewise create a CSC output format

```python
from scipy import sparse

# Create a sparse array with CSC output format
csc_array = SparseCellArray(
    uri="sparse_matrix.tdb",
    return_sparse=True,
    sparse_coerce=sparse.csc_matrix
)

# Get result as CSR matrix
result = csc_array[100:200, 500:1000]
print(result)
```

### Array Maintenance

```python
# Consolidate fragments
array.consolidate()

# Custom consolidation
config = ConsolidationConfig(
    steps=2,
    vacuum_after=True
)
array.consolidate(config)

# Vacuum
array.vacuum()
```

<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).
