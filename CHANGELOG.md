# Changelog

## Version 0.1.0

- Support cellarr-arrays on user provided tiledb array objects.
- Migrate github actions to the newer version from biocsetup.
- Renaming module names, documentation and tests

## Version 0.0.2

- Support in-memory tiledb objects. Updated tests and documentation.

## Version 0.0.1

Initial implementation of the sparse and dense arrays backed by TileDB.

- Supports reading of objects
  - Directly slices the TileDB object is all arguments to subset are contiguous blocks.
  - Otherwise redirects them to `multi_index`, if one of the argument to subset is a slice, drops the last because of inclusive upper bounds in this method.

  This helps keeps slicing consistent across various operations and trying to be performant in the process.

- Supports writing of various data objects into dense and sparse arrays. Expects all chunks to be aligned along the rows.
