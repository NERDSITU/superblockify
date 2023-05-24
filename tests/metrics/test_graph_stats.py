"""Tests for the graph statistics module."""
import pytest

from superblockify.metrics.graph_stats import (
    basic_graph_stats,
    street_orientation_order,
)


def test_graph_stats(test_city_all_copy):
    """Test the graph statistics calculation."""
    _, graph = test_city_all_copy
    stats = basic_graph_stats(graph, area=1.0)
    assert len(stats) == 19


# test street_orientation_order for faulty input num_bins (not non-positive int)
@pytest.mark.parametrize("num_bins", [-1, 0, 1.5, None, "", (1, 2), [1], {1, 2}])
def test_street_orientation_order_faulty_num_bins(test_city_small_copy, num_bins):
    """Test street_orientation_order with faulty num_bins."""
    _, graph = test_city_small_copy
    with pytest.raises(ValueError):
        street_orientation_order(graph, num_bins=num_bins)
