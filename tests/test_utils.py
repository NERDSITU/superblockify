"""Tests for the utils module."""
from configparser import ConfigParser

from superblockify.utils import load_graph_from_place

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]


def test_load_graph_from_place():
    """Test that the load_graph_from_place function works."""

    graph = load_graph_from_place(
        f"{TEST_DATA}/cities/Adliswil_small.graphml",
        "Adliswil, Bezirk Horgen, Zürich, Switzerland",
        network_type="drive",
    )

    assert graph is not None
    assert len(graph) > 0
    assert graph.size() > 0
