"""Expose most common parts of public API directly in `superblockify.` namespace."""

# pylint: disable=unused-import
from .attribute import new_edge_attribute_by_function
from .partitioning import BearingPartitioner
from .partitioning import BetweennessPartitioner
from .partitioning import DummyPartitioner
from .partitioning import MinimumPartitioner
from .partitioning import ResidentialPartitioner
from .partitioning import save_to_gpkg
from .partitioning import __all_partitioners__
from .partitioning import plot_speed_un_restricted
from .plot import paint_streets
from .plot import plot_by_attribute
from .plot import plot_road_type_for
from .population import get_ghsl
from .population import resample_load_window
from .population import add_edge_cells
from .population import get_edge_cells
from .population import add_edge_population
from .population import get_edge_population
from .population import get_population_area

# Filter deprecated partitioners from `__all_partitioners__` if they have the
# `__deprecated__` attribute.
all_partitioners = [
    part for part in __all_partitioners__ if not getattr(part, "__deprecated__", False)
]
