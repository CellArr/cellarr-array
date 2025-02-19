from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import tiledb
from scipy import sparse

from .config import ConsolidationConfig
from .helpers import SliceHelper

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArray(ABC):
    """Abstract base class for TileDB array operations."""

    def __init__(
        self,
        uri: str,
        attr: str = "data",
        mode: Optional[Literal["r", "w", "n", "d"]] = None,
        config_or_context: Optional[Union[tiledb.Config, tiledb.Ctx]] = None,
        validate: bool = True,
    ):
        """Initialize the object.

        Args:
            uri:
                URI to the array.

            attr:
                Attribute to access.
                Defaults to "data".

            mode:
                Open the array object in read 'r', write 'w', modify
                exclusive 'm' mode, or delete 'd' mode.

                Defaults to None for automatic mode switching.

            config_or_context:
                Config or context object.
                Defaults to None.

            validate:
                Whether to validate the attributes.
                Defaults to True.
        """
        self.uri = uri
        self._mode = mode

        if config_or_context is None:
            config_or_context = tiledb.Config()

        if isinstance(config_or_context, tiledb.Config):
            ctx = tiledb.Ctx(config_or_context)
        elif isinstance(config_or_context, tiledb.Ctx):
            ctx = config_or_context
        else:
            raise TypeError("'config_or_context' must be either TileDB config or a context object.")

        self._ctx = ctx
        self._array = None
        self._shape = None
        self._ndim = None
        self._dim_names = None
        self._attr_names = None

        if validate:
            self._validate(attr=attr)

        self._attr = attr

    def _validate(self, attr):
        with self.open_array(mode="r") as A:
            if A.ndim > 2:
                raise ValueError("Only 1D and 2D arrays are supported.")

            if attr not in self.attr_names:
                raise ValueError(
                    f"Attribute '{attr}' does not exist in the array. Available attributes: {self.attr_names}."
                )

    @property
    def mode(self) -> Optional[str]:
        """Get current array mode."""
        return self._mode

    @mode.setter
    def mode(self, value: Optional[str]):
        """Set array mode.

        Args:
            value:
                One of `None`, 'r', 'w', or 'm', 'd'.
        """
        if value is not None and value not in ["r", "w", "m", "d"]:
            raise ValueError("Mode must be one of: None, 'r', 'w', 'm', 'd'")
        self._mode = value

    @property
    def dim_names(self) -> List[str]:
        """Get dimension names of the array."""
        if self._dim_names is None:
            with self.open_array(mode="r") as A:
                self._dim_names = [dim.name for dim in A.schema.domain]
        return self._dim_names

    @property
    def attr_names(self) -> List[str]:
        """Get attribute names of the array."""
        if self._attr_names is None:
            with self.open_array(mode="r") as A:
                self._attr_names = [A.schema.attr(i).name for i in range(A.schema.nattr)]
        return self._attr_names

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get array shape from schema domain."""
        if self._shape is None:
            with self.open_array(mode="r") as A:
                self._shape = tuple(int(dim.domain[1] - dim.domain[0] + 1) for dim in A.schema.domain)
        return self._shape

    @property
    def nonempty_domain(self) -> Tuple[int, ...]:
        """Get array non-empty domain."""
        if self._nonempty_domain is None:
            with self.open_array(mode="r") as A:
                self._nonempty_domain = A.nonempty_domain()
        return self._nonempty_domain

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        if self._ndim is None:
            self._ndim = len(self.shape)
        return self._ndim

    @contextmanager
    def open_array(self, mode: Optional[str] = None):
        """Context manager for array operations.

        Args:
            mode:
                Override mode for this operation.
        """
        mode = mode if mode is not None else self.mode
        mode = mode if mode is not None else "r"  # Default to read mode

        array = tiledb.open(self.uri, mode=mode, ctx=self._ctx)
        try:
            yield array
        finally:
            array.close()

    def __getitem__(self, key: Union[slice, Tuple[Union[slice, List[int]], ...]]):
        """Get item implementation that routes to either direct slicing or multi_index
        based on the type of indices provided.

        Args:
            key:
                Slice or list of indices for each dimension in the array.
        """
        if not isinstance(key, tuple):
            key = (key,)

        if len(key) > self.ndim:
            raise IndexError(f"Invalid number of dimensions: got {len(key)}, expected {self.ndim}")

        # Normalize all indices
        normalized_key = tuple(SliceHelper.normalize_index(idx, self.shape[i]) for i, idx in enumerate(key))

        # Check if we can use direct slicing
        use_direct = all(isinstance(idx, slice) for idx in normalized_key)

        if use_direct:
            return self._direct_slice(normalized_key)
        else:
            return self._multi_index(normalized_key)

    @abstractmethod
    def _direct_slice(self, key: Tuple[slice, ...]) -> np.ndarray:
        """Implementation for direct slicing."""
        pass

    @abstractmethod
    def _multi_index(self, key: Tuple[Union[slice, List[int]], ...]) -> np.ndarray:
        """Implementation for multi-index access."""
        pass

    def vacuum(self) -> None:
        """Remove deleted fragments from the array."""
        tiledb.vacuum(self.uri)

    def consolidate(self, config: Optional[ConsolidationConfig] = None) -> None:
        """Consolidate array fragments.

        Args:
            config:
                Optional consolidation configuration.
        """
        if config is None:
            config = ConsolidationConfig()

        consolidation_cfg = tiledb.Config()

        consolidation_cfg["sm.consolidation.steps"] = config.steps
        consolidation_cfg["sm.consolidation.step_min_frags"] = config.step_min_frags
        consolidation_cfg["sm.consolidation.step_max_frags"] = config.step_max_frags
        consolidation_cfg["sm.consolidation.buffer_size"] = config.buffer_size
        consolidation_cfg["sm.mem.total_budget"] = config.total_budget

        tiledb.consolidate(self.uri, config=consolidation_cfg)

        if config.vacuum_after:
            self.vacuum()

    @abstractmethod
    def write_batch(self, data: Union[np.ndarray, sparse.spmatrix], start_row: int, **kwargs) -> None:
        """Write a batch of data to the array starting at the specified row.

        Args:
            data:
                Data to write (numpy array for dense, scipy sparse matrix for sparse).

            start_row:
                Starting row index for writing.

            **kwargs:
                Additional arguments for write operation.
        """
        pass
