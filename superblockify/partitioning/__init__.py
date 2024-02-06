"""Partitioning init, subpackage for the base partitioner and all approaches."""

from inspect import isclass

# Import all partitioners for `__api.py`.
from .base import BasePartitioner
from .approaches import (
    BearingPartitioner,
    BetweennessPartitioner,
    DummyPartitioner,
    MinimumPartitioner,
    ResidentialPartitioner,
)

# Further utils.
from .utils import save_to_gpkg

# Partitioner plotting.
from .plot import (
    plot_partition_graph,
    plot_subgraph_component_size,
    plot_component_rank_size,
    plot_component_graph,
    plot_speed_un_restricted,
)

# List of all supported partitioners. For all partitioners that are subclasses of
# BasePartitioner or its subclasses, but not BasePartitioner itself.
__all_partitioners__ = [
    part
    for part in locals().values()
    if isclass(part) and BasePartitioner in part.mro() and part is not BasePartitioner
]
