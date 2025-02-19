from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@dataclass
class CellArrConfig:
    """Configuration class for TileDB array creation and access."""

    tile_capacity: int = 100000
    cell_order: str = "row-major"
    tile_order: str = "row-major"
    coords_filters: List[tiledb.Filter] = field(default_factory=lambda: [tiledb.LZ4Filter()])
    offsets_filters: List[tiledb.Filter] = field(default_factory=lambda: [tiledb.LZ4Filter()])
    attrs_filters: Dict[str, List[tiledb.Filter]] = field(default_factory=lambda: {"": [tiledb.LZ4Filter()]})
    ctx_config: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create_filter(filter_config: Union[Dict[str, Any], tiledb.Filter]) -> tiledb.Filter:
        """Create a TileDB Filter object from configuration."""
        if isinstance(filter_config, tiledb.Filter):
            return filter_config

        if isinstance(filter_config, dict):
            filter_name = filter_config.get("name", "").lower()
            filter_level = filter_config.get("level", None)

            if filter_name == "zstd":
                return tiledb.ZstdFilter(level=filter_level)
            elif filter_name == "gzip":
                return tiledb.GzipFilter(level=filter_level)
            elif filter_name == "bzip2":
                return tiledb.Bzip2Filter(level=filter_level)
            elif filter_name == "double-delta":
                return tiledb.DoubleDeltaFilter()
            elif filter_name == "bit-width-reduction":
                return tiledb.BitWidthReductionFilter()
            else:
                raise ValueError(f"Unsupported filter type: {filter_name}")

        raise TypeError("Filter must be either a TileDB Filter object or a configuration dictionary")

    def __post_init__(self):
        """Convert filter configurations to TileDB Filter objects."""
        if not isinstance(self.coords_filters, list):
            self.coords_filters = [self.coords_filters]
        self.coords_filters = [self.create_filter(f) for f in self.coords_filters]

        if not isinstance(self.offsets_filters, list):
            self.offsets_filters = [self.offsets_filters]
        self.offsets_filters = [self.create_filter(f) for f in self.offsets_filters]

        for attr, filters in self.attrs_filters.items():
            if not isinstance(filters, list):
                filters = [filters]
            self.attrs_filters[attr] = [self.create_filter(f) for f in filters]


@dataclass
class ConsolidationConfig:
    """Configuration for array consolidation."""

    steps: int = 100000
    step_min_frags: int = 2
    step_max_frags: int = 10
    buffer_size: int = 15000000000  # 15GB
    total_budget: int = 40000000000  # 40GB
    num_threads: int = 4
    vacuum_after: bool = True
