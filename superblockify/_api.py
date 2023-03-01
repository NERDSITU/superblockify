"""Expose most common parts of public API directly in `superblockify.`
namespace."""

# pylint: disable=unused-import
from .attribute import new_edge_attribute_by_function
from .partitioning import BearingPartitioner
from .partitioning import DummyPartitioner
from .partitioning import ResidentialPartitioner
from .plot import paint_streets
from .plot import plot_by_attribute
from .plot import plot_road_type_for
