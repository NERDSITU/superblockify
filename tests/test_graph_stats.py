"""Tests for the graph statistics module."""

import pytest
from shapely import Polygon, MultiPolygon

from superblockify.graph_stats import (
    basic_graph_stats,
    street_orientation_order,
)


def test_graph_stats(test_city_small_copy):
    """Test the graph statistics calculation."""
    _, graph = test_city_small_copy
    stats = basic_graph_stats(graph, area=1.0)
    assert len(stats) == 19
    assert stats["circuity_avg"] >= 1.0
    assert stats["k_avg"] >= 1
    assert stats["n"] >= 2
    assert stats["m"] >= 1
    assert 0 < stats["street_orientation_order"] <= 1


# test street_orientation_order for faulty input num_bins (not non-positive int)
@pytest.mark.parametrize("num_bins", [-1, 0, 1.5, None, "", (1, 2), [1], {1, 2}])
def test_street_orientation_order_faulty_num_bins(test_city_small_copy, num_bins):
    """Test street_orientation_order with faulty num_bins."""
    _, graph = test_city_small_copy
    with pytest.raises(ValueError):
        street_orientation_order(graph, num_bins=num_bins)


def test_calculate_component_metrics(test_city_small_precalculated_copy):
    """Test calculate_component_metrics.
    Should have been called in the test_city_small_precalculated_copy fixture."""
    part = test_city_small_precalculated_copy
    for comp in part.get_ltns():
        assert comp["population"] >= 0
        assert comp["area"] >= 0
        assert comp["population_density"] >= 0
        assert comp["n"] >= 2
        assert comp["m"] >= 1


def test_load_graphml_dtypes(test_city_small_precalculated_copy):
    """Test that the graph is loaded with the correct dtypes."""
    graph = test_city_small_precalculated_copy.graph.graph
    assert isinstance(graph["simplified"], bool)
    assert isinstance(graph["edge_population"], bool)  # bool showing if pop was added
    assert isinstance(graph["boundary"], (Polygon, MultiPolygon))
    assert isinstance(graph["area"], float)
    assert isinstance(graph["n"], int)
    assert isinstance(graph["m"], int)
    assert isinstance(graph["k_avg"], float)
    assert isinstance(graph["edge_length_total"], float)
    assert isinstance(graph["edge_length_avg"], float)
    assert isinstance(graph["streets_per_node_avg"], float)
    assert isinstance(graph["streets_per_node_counts"], dict)
    assert all(isinstance(k, int) for k in graph["streets_per_node_counts"].keys())
    assert all(isinstance(v, int) for v in graph["streets_per_node_counts"].values())
    assert isinstance(graph["streets_per_node_proportions"], dict)
    assert all(isinstance(k, int) for k in graph["streets_per_node_proportions"].keys())
    assert all(
        isinstance(v, float) for v in graph["streets_per_node_proportions"].values()
    )
    assert isinstance(graph["intersection_count"], int)
    assert isinstance(graph["street_length_total"], float)
    assert isinstance(graph["street_segment_count"], int)
    assert isinstance(graph["street_length_avg"], float)
    assert isinstance(graph["circuity_avg"], float)
    assert isinstance(graph["self_loop_proportion"], float)
    assert isinstance(graph["node_density_km"], float)
    assert isinstance(graph["intersection_density_km"], float)
    assert isinstance(graph["edge_density_km"], float)
    assert isinstance(graph["street_density_km"], float)
    assert isinstance(graph["street_orientation_order"], float)
