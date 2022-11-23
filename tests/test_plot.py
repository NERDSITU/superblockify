"""Tests for the plot module."""
from os import listdir

import osmnx as ox
import pytest

from superblockify.plot import paint_streets

TEST_DATA = "./tests/test_data/"


@pytest.mark.parametrize(
    "city_path",
    [city for city in listdir(f"{TEST_DATA}cities/") if city.endswith(".graphml")],
)
@pytest.mark.parametrize("e_l,n_a", [(0.5, 0.5), (1, 0)])
@pytest.mark.parametrize("save", [True, False])
def test_paint_streets(city_path, e_l, n_a, save):
    """Test `paint_streets` by design."""
    graph = ox.load_graphml(filepath=f"{TEST_DATA}cities/" + city_path)
    paint_streets(
        graph,
        edge_linewidth=e_l,
        node_alpha=n_a,
        save=save,
        filepath=f"{TEST_DATA}output/{city_path[:-8]}.pdf",
    )


@pytest.mark.parametrize(
    "city_path",
    [city for city in listdir(f"{TEST_DATA}cities/") if city.endswith(".graphml")][:2],
)
def test_paint_streets_overwrite_ec(city_path):
    """Test `paint_streets` trying to overwrite the edge colors."""
    graph = ox.load_graphml(filepath=f"{TEST_DATA}cities/" + city_path)
    with pytest.raises(ValueError):
        paint_streets(graph, edge_color="white")


@pytest.mark.parametrize(
    "city_path",
    [city for city in listdir(f"{TEST_DATA}cities/") if city.endswith(".graphml")][:2],
)
def test_paint_streets_empty_plot(city_path):
    """Test `paint_streets` trying plot empty plot."""
    graph = ox.load_graphml(filepath=f"{TEST_DATA}cities/" + city_path)
    with pytest.raises(ValueError):
        paint_streets(graph, edge_linewidth=0, node_size=0)
