"""Tests for the tessellation module."""
import pytest
from shapely import Polygon, MultiPolygon

from superblockify.population.tessellation import add_edge_cells


def test_add_edge_cells(test_city_small_copy):
    """Test the `add_edge_cells` function by design."""
    _, graph = test_city_small_copy
    add_edge_cells(graph, show_plot=True)
    # Check all edges have the `cell` attribute for Polygons
    for _, _, data in graph.edges(data=True):
        assert isinstance(data["cell"], (Polygon, MultiPolygon))


def test_add_edge_cells_unprojected_graph(test_city_small_copy):
    """Test the `add_edge_cells` function for unprojected graph."""
    _, graph = test_city_small_copy
    graph.graph["crs"] = "epsg:4326"
    with pytest.raises(ValueError):
        add_edge_cells(graph)


@pytest.mark.parametrize(
    "limit",
    [
        "string",
        1,
        1.0,
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ],
)
def test_add_edge_cells_invalid_limit(limit, test_city_small_copy):
    """Test the `add_edge_cells` function for an invalid limit polygon."""
    _, graph = test_city_small_copy
    limit = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError):
        add_edge_cells(graph, limit=limit)
