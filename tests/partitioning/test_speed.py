"""Tests for the speed submodule of the partitioning module."""
from configparser import ConfigParser
from os.path import join, dirname

from superblockify.partitioning.speed import add_edge_travel_times_restricted

config = ConfigParser()
config.read(join(dirname(__file__), "..", "..", "config.ini"))


def test_add_edge_travel_times_restricted(test_city_small_copy):
    """Test adding restricted travel times to a graph by design."""
    _, graph = test_city_small_copy
    # sparsified graph is a subview of the original graph, just take half of the edges
    sparsified = graph.edge_subgraph(list(graph.edges)[::2])
    add_edge_travel_times_restricted(graph, sparsified)
    # check that all edges have travel_time_restricted attribute
    # with a non-negative value
    for edge in graph.edges:
        assert "travel_time_restricted" in graph.edges[edge]
        assert graph.edges[edge]["travel_time_restricted"] >= 0
