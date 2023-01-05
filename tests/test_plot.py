"""Tests for the plot module."""
from configparser import ConfigParser

import pytest
from matplotlib import pyplot as plt

from superblockify import new_edge_attribute_by_function
from superblockify.plot import paint_streets, plot_by_attribute

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]


@pytest.mark.parametrize("e_l,n_a", [(0.5, 0.5), (1, 0)])
@pytest.mark.parametrize("save", [True, False])
def test_paint_streets(test_city_bearing, e_l, n_a, save):
    """Test `paint_streets` by design."""
    city_path, graph = test_city_bearing
    paint_streets(
        graph,
        edge_linewidth=e_l,
        node_alpha=n_a,
        save=save,
        filepath=f"{TEST_DATA}output/{city_path[:-8]}.pdf",
    )
    plt.close()


def test_paint_streets_overwrite_ec(test_city_bearing):
    """Test `paint_streets` trying to overwrite the edge colors."""
    _, graph = test_city_bearing
    with pytest.raises(ValueError):
        paint_streets(graph, edge_color="white")


def test_paint_streets_empty_plot(test_city_bearing):
    """Test `paint_streets` trying plot empty plot."""
    _, graph = test_city_bearing
    with pytest.raises(ValueError):
        paint_streets(graph, edge_linewidth=0, node_size=0)


def test_plot_by_attribute(test_city_bearing):
    """Test `plot_by_attribute` by design."""
    _, graph = test_city_bearing

    # Use osmid as attribute determining color
    # Some osmid attributes return lists, not ints, just take first element
    new_edge_attribute_by_function(
        graph,
        lambda osmid: osmid if isinstance(osmid, int) else osmid[0],
        "osmid",
        "osmid_0",
    )

    plot_by_attribute(graph, "osmid_0", cmap="rainbow")
    plt.close()


def test_plot_by_attribute_no_attribute(test_city_bearing):
    """Test `plot_by_attribute` with missing attribute."""
    _, graph = test_city_bearing
    with pytest.raises(ValueError):
        plot_by_attribute(graph, "non_existent_attribute")


@pytest.mark.parametrize(
    "minmax_val_faulty",
    [
        1,
        1.3,
        "str",
        tuple(),
        (1,),
        (1, 2, 3),
        (1, 2, 3, 4),
        (0, 0),
        (1, 1),
        (1, 0),
    ],
)
def test_plot_by_attribute_minmax_val_faulty(test_city_bearing, minmax_val_faulty):
    """Test `plot_by_attribute` with faulty minmax_val."""
    _, graph = test_city_bearing
    with pytest.raises(ValueError):
        plot_by_attribute(graph, "osmid", minmax_val=minmax_val_faulty)
