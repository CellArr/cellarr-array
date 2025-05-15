from abc import ABC, abstractmethod
from contextlib import contextmanager

try:
    from types import EllipsisType
except ImportError:
    # TODO: This is required for Python <3.10. Remove once Python 3.9 reaches EOL in October 2025
    EllipsisType = type(...)
from typing import Any, List, Literal, Optional, Tuple, Union

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
        uri: Optional[str] = None,
        tiledb_array_obj: Optional[tiledb.Array] = None,
        attr: str = "data",
        mode: Optional[Literal["r", "w", "d", "m"]] = None,
        config_or_context: Optional[Union[tiledb.Config, tiledb.Ctx]] = None,
        validate: bool = True,
    ):
        """Initialize the object.

        Args:
            uri:
                URI to the array.
                Required if 'tiledb_array_obj' is not provided.

            tiledb_array_obj:
                Optional, an already opened ``tiledb.Array`` instance.
                If provided, 'uri' can be None, and 'config_or_context' is ignored.

            attr:
                Attribute to access.
                Defaults to "data".

            mode:
                Open the array object in read 'r', write 'w', modify
                'm' mode, or delete 'd' mode.

                Defaults to None for automatic mode switching.

                If 'tiledb_array_obj' is provided, this mode should ideally match
                the mode of the provided array or be None.

            config_or_context:
                Optional config or context object. Ignored if 'tiledb_array_obj' is provided,
                as context will be derived from the object.

                Defaults to None.

            validate:
                Whether to validate the attributes.
                Defaults to True.
        """
        self._array_passed_in = False
        self._opened_array_external = None
        self._ctx = None

        if tiledb_array_obj is not None:
            if not isinstance(tiledb_array_obj, tiledb.Array):
                raise ValueError("'tiledb_array_obj' must be a tiledb.Array instance.")

            if not tiledb_array_obj.isopen:
                # Option 1: Raise error
                raise ValueError("If 'tiledb_array_obj' is provided, it must be an open tiledb.Array instance.")
                # Option 2: Try to reopen (less safe as we don't know original intent)
                # try:
                #     tiledb_array_obj.reopen()
                # except tiledb.TileDBError as e:
                #     raise ValueError(
                #         f"Provided 'tiledb_array_obj' is closed and could not be reopened: {e}"
                #     )

            self.uri = tiledb_array_obj.uri
            self._array_passed_in = True
            self._opened_array_external = tiledb_array_obj

            # infer mode if possible, or require it matches
            if mode is not None and tiledb_array_obj.mode != mode:
                # we could try to reopen with the desired mode
                raise ValueError(
                    f"Provided array mode '{tiledb_array_obj.mode}' does not match requested mode '{mode}'.",
                    "Re-open the external array with the desired mode or pass matching mode.",
                )

            self._mode = tiledb_array_obj.mode
            self._ctx = tiledb_array_obj.ctx
        elif uri is not None:
            self.uri = uri
            self._mode = mode
            self._array_passed_in = False
            self._opened_array_external = None

            if config_or_context is None:
                self._ctx = None
            elif isinstance(config_or_context, tiledb.Config):
                self._ctx = tiledb.Ctx(config_or_context)
            elif isinstance(config_or_context, tiledb.Ctx):
                self._ctx = config_or_context
            else:
                raise TypeError("'config_or_context' must be a TileDB Config or Ctx object.")
        else:
            raise ValueError("Either 'uri' or 'tiledb_array_obj' must be provided.")

        self._shape = None
        self._ndim = None
        self._dim_names = None
        self._attr_names = None
        self._nonempty_domain = None

        if validate:
            self._validate(attr=attr)

        self._attr = attr

    def _validate(self, attr):
        with self.open_array(mode="r") as A:
            schema = A.schema
            if schema.ndim > 2:
                raise ValueError("Only 1D and 2D arrays are supported.")

            current_attr_names = [schema.attr(i).name for i in range(schema.nattr)]
            if attr not in current_attr_names:
                raise ValueError(
                    f"Attribute '{attr}' does not exist in the array. Available attributes: {current_attr_names}."
                )

    @property
    def mode(self) -> Optional[str]:
        """Get current array mode. If an external array is used, this is its open mode."""
        if self._array_passed_in and self._opened_array_external is not None:
            return self._opened_array_external.mode
        return self._mode

    @mode.setter
    def mode(self, value: Optional[str]):
        """Set array mode for subsequent operations if not using an external array.

        This action does not affect an already passed-in external array's mode.
        """
        if self._array_passed_in:
            # To change mode of an external array, user must reopen it and pass it again.
            current_ext_mode = self._opened_array_external.mode if self._opened_array_external else "unknown"
            if value != current_ext_mode:
                raise ValueError(
                    f"Cannot change mode of an externally managed array (current: {current_ext_mode}). "
                    "Re-open the external array with the new mode and re-initialize CellArray."
                )
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
        if self._shape is None:
            with self.open_array(mode="r") as A:
                self._shape = tuple(int(dim.domain[1] - dim.domain[0] + 1) for dim in A.schema.domain)
        return self._shape

    @property
    def nonempty_domain(self) -> Optional[Tuple[Any, ...]]:
        if self._nonempty_domain is None:
            with self.open_array(mode="r") as A:
                # nonempty_domain() can return None if the array is empty.
                ned = A.nonempty_domain()
                if ned is None:
                    self._nonempty_domain = None
                else:
                    self._nonempty_domain = tuple(ned) if isinstance(ned[0], tuple) else (ned,)
        return self._nonempty_domain

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        if self._ndim is None:
            with self.open_array(mode="r") as A:
                self._ndim = A.schema.ndim
                # self._ndim = len(self.shape)
        return self._ndim

    @contextmanager
    def open_array(self, mode: Optional[str] = None):
        """Context manager for array operations.

        Uses the externally provided array if available, otherwise opens from URI.

        Args:
            mode:
                Desired mode for the operation ('r', 'w', 'm', 'd').
                If an external array is used, this mode must be compatible with
                (or same as) the mode the external array was opened with.

                If None, uses the CellArray's default mode.
        """
        if self._array_passed_in and self._opened_array_external is not None:
            if not self._opened_array_external.isopen:
                # Attempt to reopen if closed. This assumes the user might have closed it
                # and expects CellArr to reopen it if still possible.
                try:
                    self._opened_array_external.reopen()
                except Exception as e:
                    raise tiledb.TileDBError(
                        f"Externally provided array is closed and could not be reopened: {e}"
                    ) from e

            effective_mode = mode if mode is not None else self._opened_array_external.mode

            current_external_mode = self._opened_array_external.mode
            if effective_mode == "r" and current_external_mode not in ["r", "w", "m"]:
                # Read ops ok on write/modify modes
                pass
            elif effective_mode in ["w", "d"] and current_external_mode != effective_mode:
                raise tiledb.TileDBError(
                    f"Requested operation mode '{effective_mode}' is incompatible with the "
                    f"externally provided array's mode '{current_external_mode}'. "
                    "Ensure the external array is opened in a compatible mode."
                )

            # DO NOT close self._opened_array_external here; its lifecycle is managed by the user.
            yield self._opened_array_external
        else:
            effective_mode = mode if mode is not None else self.mode
            effective_mode = effective_mode if effective_mode is not None else "r"
            array = tiledb.open(self.uri, mode=effective_mode, ctx=self._ctx)

            try:
                yield array
            finally:
                array.close()

    def __getitem__(self, key: Union[slice, EllipsisType, Tuple[Union[slice, List[int]], ...], EllipsisType]):
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

        num_ellipsis = sum(isinstance(i, EllipsisType) for i in normalized_key)
        if num_ellipsis > 1:
            raise IndexError(f"Found more than 1 Ellipsis (...) in key: {normalized_key}")

        # Check if we can use direct slicing
        use_direct = all(isinstance(idx, (slice, EllipsisType)) for idx in normalized_key)

        if use_direct:
            return self._direct_slice(normalized_key)
        else:
            if num_ellipsis > 0:
                raise IndexError(f"tiledb does not support ellipsis in multi-index access: {normalized_key}")
            return self._multi_index(normalized_key)

    @abstractmethod
    def _direct_slice(self, key: Tuple[Union[slice, EllipsisType], ...]) -> np.ndarray:
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
