"""Partitioning init, subpackage for the various approaches."""
from inspect import isclass

# Import all partitioners for `__api.py`.
from .bearing import BearingPartitioner
from .dummy import DummyPartitioner
from .partitioner import BasePartitioner
from .streettype import ResidentialPartitioner

# Further utils.
from .utils import save_to_gpkg

# List of all supported partitioners. For all partitioners that are subclasses of
# BasePartitioner, but not BasePartitioner itself.
__all_partitioners__ = [
    part
    for part in globals().values()
    if isclass(part)
    and issubclass(part, BasePartitioner)
    and part is not BasePartitioner
]
