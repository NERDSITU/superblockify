"""Tests for the attribute module."""
import networkx as nx
import pytest

from superblockify.attribute import new_edge_attribute_by_function


def __func(var):
    """Function used in tests.

    Can be anything numerical for the
    parametrized tests.
    """
    return 2 * var


@pytest.mark.parametrize("test_input", [4, -4, 0, 2.0, 10e5])
@pytest.mark.parametrize("graph_len", [2, 3, 20])
def test_new_edge_attribute_by_function(test_input, graph_len):
    """Test `new_edge_attribute_by_function` by design."""

    expected = __func(test_input)

    g_path = nx.path_graph(graph_len)
    nx.set_edge_attributes(g_path, test_input, "in")
    new_edge_attribute_by_function(g_path, __func, "in", "out")

    assert nx.get_edge_attributes(g_path, "out") == {
        (i, i + 1): expected for i in range(graph_len - 1)
    }


@pytest.mark.parametrize("graph_len", [0, 1, 2, 3, 20])
def test_new_edge_attribute_by_function_no_attributes(graph_len):
    """Test `new_edge_attribute_by_function` when graph has no attributes."""

    g_path = nx.path_graph(graph_len)
    with pytest.raises(ValueError):
        new_edge_attribute_by_function(g_path, __func, "in", "out")


@pytest.mark.parametrize("test_input", [4, -4, 0, 2.0, 10e5])
@pytest.mark.parametrize("graph_len", [0, 1])
def test_new_edge_attribute_by_function_no_edges(test_input, graph_len):
    """Test `new_edge_attribute_by_function` when graph has no edges."""

    g_path = nx.path_graph(graph_len)
    nx.set_edge_attributes(g_path, test_input, "in")
    with pytest.raises(ValueError):
        new_edge_attribute_by_function(g_path, __func, "in", "out")


@pytest.mark.parametrize("source,destination", [("in", "in"), ("in", "other")])
def test_new_edge_attribute_by_function_overwriting_disallowed(source, destination):
    """Test `new_edge_attribute_by_function` to catch disallowed overwriting."""
    g_path = nx.path_graph(3)
    nx.set_edge_attributes(g_path, 4, "in")
    nx.set_edge_attributes(g_path, None, "other")
    with pytest.raises(ValueError):
        new_edge_attribute_by_function(g_path, __func, source, destination)


@pytest.mark.parametrize("test_input", [4, -4, 0, 2.0, 10e5])
@pytest.mark.parametrize("graph_len", [2, 3, 20])
def test_new_edge_attribute_by_function_overwriting_allowed(test_input, graph_len):
    """Test `new_edge_attribute_by_function` by design."""

    expected = __func(test_input)

    g_path = nx.path_graph(graph_len)
    nx.set_edge_attributes(g_path, test_input, "in")
    new_edge_attribute_by_function(g_path, __func, "in", "in", allow_overwriting=True)

    assert nx.get_edge_attributes(g_path, "in") == {
        (i, i + 1): expected for i in range(graph_len - 1)
    }
