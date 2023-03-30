"""Tests for the plot module."""
from configparser import ConfigParser

import pytest
from matplotlib import pyplot as plt

from superblockify import new_edge_attribute_by_function
from superblockify.plot import (
    paint_streets,
    plot_by_attribute,
    make_edge_color_list,
    plot_road_type_for,
)

config = ConfigParser()
config.read("config.ini")


@pytest.mark.parametrize("e_l,n_a", [(0.5, 0.5), (1, 0)])
@pytest.mark.parametrize("save", [True, False])
def test_paint_streets(test_city_all_copy, e_l, n_a, save):
    """Test `paint_streets` by design."""
    city_path, graph = test_city_all_copy
    paint_streets(
        graph,
        edge_linewidth=e_l,
        node_alpha=n_a,
        save=save,
        filepath=f"{config['tests']['test_data_path']}output/{city_path}.pdf",
    )
    plt.close()


def test_paint_streets_overwrite_ec(test_city_all_copy):
    """Test `paint_streets` trying to overwrite the edge colors."""
    _, graph = test_city_all_copy
    with pytest.raises(ValueError):
        paint_streets(graph, edge_color="white")


def test_paint_streets_empty_plot(test_city_all_copy):
    """Test `paint_streets` trying plot empty plot."""
    _, graph = test_city_all_copy
    with pytest.raises(ValueError):
        paint_streets(graph, edge_linewidth=0, node_size=0)


def test_plot_by_attribute(test_city_all_copy):
    """Test `plot_by_attribute` by design."""
    _, graph = test_city_all_copy
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


def test_plot_by_attribute_no_attribute(test_city_all_copy):
    """Test `plot_by_attribute` with missing attribute."""
    _, graph = test_city_all_copy
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
def test_plot_by_attribute_minmax_val_faulty(test_city_all_copy, minmax_val_faulty):
    """Test `plot_by_attribute` with faulty minmax_val."""
    _, graph = test_city_all_copy
    with pytest.raises(ValueError):
        plot_by_attribute(graph, "osmid", minmax_val=minmax_val_faulty)


def test_make_edge_color_list(test_city_all_copy):
    """Test `make_edge_color_list` by design."""
    _, graph = test_city_all_copy
    colormap = plt.get_cmap("rainbow")
    edge_color_list = list(
        make_edge_color_list(graph, "bearing", cmap=colormap, attr_types="numerical")
    )
    assert len(edge_color_list) == len(graph.edges)
    assert isinstance(edge_color_list[0], tuple)
    assert len(edge_color_list[0]) == 4


@pytest.mark.parametrize(
    "attr_type,minmax",
    [
        ("ff", (0.5, 1.0)),  # attr_type not in ["numerical", "categorical"]
        ("categorical", 0),  # minmax not None
        ("numerical", (0.5, 1.0, 1.5)),  # minmax not two-element tuple or None
        ("numerical", (1.5)),  # minmax not two-element tuple or None
        ("numerical", True),  # minmax not two-element tuple or None
    ],
)
def test_make_edge_color_list_faulty_attr_type(test_city_all_copy, attr_type, minmax):
    """Test `make_edge_color_list` with faulty attr_type."""
    _, graph = test_city_all_copy
    colormap = plt.get_cmap("rainbow")
    with pytest.raises((ValueError, TypeError)):
        make_edge_color_list(
            graph, "bearing", cmap=colormap, attr_types=attr_type, minmax_val=minmax
        )


def test_make_edge_color_list_attr_unsortable(test_city_all_copy):
    """Test `make_edge_color_list` with unsortable attribute."""
    _, graph = test_city_all_copy
    colormap = plt.get_cmap("rainbow")
    # Set first edge with to a number, second one to a string, third one to a list
    node = list(graph.edges(data=True))[0]
    graph[node[0]][node[1]][0]["bearing"] = "str"

    make_edge_color_list(graph, "bearing", cmap=colormap, attr_types="categorical")


@pytest.mark.parametrize(
    "road_types",
    [
        ["residential"],
        ["residential", "unclassified"],
        [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
        ],
    ],
)
def test_plot_road_type_for(test_city_all_copy, road_types):
    """Test `plot_road_type_for` by design."""
    city_name, graph = test_city_all_copy
    plot_road_type_for(graph, included_types=road_types, name=city_name)
    plt.close()
