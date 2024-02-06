"""Tests for the attribute module."""

from inspect import isclass

import pytest
from networkx import (
    set_edge_attributes,
    path_graph,
    get_edge_attributes,
    Graph,
    set_node_attributes,
)
from numpy import sum as npsum

from superblockify.attribute import (
    new_edge_attribute_by_function,
    get_edge_subgraph_with_attribute_value,
    determine_minmax_val,
    aggregate_edge_attr,
)


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

    g_path = path_graph(graph_len)
    set_edge_attributes(g_path, test_input, "in")
    new_edge_attribute_by_function(g_path, __func, "in", "out")

    assert get_edge_attributes(g_path, "out") == {
        (i, i + 1): expected for i in range(graph_len - 1)
    }


@pytest.mark.parametrize("graph_len", [0, 1, 2, 3, 20])
def test_new_edge_attribute_by_function_no_attributes(graph_len):
    """Test `new_edge_attribute_by_function` when graph has no attributes."""

    g_path = path_graph(graph_len)
    with pytest.raises(ValueError):
        new_edge_attribute_by_function(g_path, __func, "in", "out")


@pytest.mark.parametrize("test_input", [4, -4, 0, 2.0, 10e5])
@pytest.mark.parametrize("graph_len", [0, 1])
def test_new_edge_attribute_by_function_no_edges(test_input, graph_len):
    """Test `new_edge_attribute_by_function` when graph has no edges."""

    g_path = path_graph(graph_len)
    set_edge_attributes(g_path, test_input, "in")
    with pytest.raises(ValueError):
        new_edge_attribute_by_function(g_path, __func, "in", "out")


@pytest.mark.parametrize("source,destination", [("in", "in"), ("in", "other")])
def test_new_edge_attribute_by_function_overwriting_disallowed(source, destination):
    """Test `new_edge_attribute_by_function` to catch disallowed overwriting."""
    g_path = path_graph(3)
    set_edge_attributes(g_path, 4, "in")
    set_edge_attributes(g_path, None, "other")
    with pytest.raises(ValueError):
        new_edge_attribute_by_function(g_path, __func, source, destination)


@pytest.mark.parametrize("test_input", [4, -4, 0, 2.0, 10e5])
@pytest.mark.parametrize("graph_len", [2, 3, 20])
def test_new_edge_attribute_by_function_overwriting_allowed(test_input, graph_len):
    """Test `new_edge_attribute_by_function` by design, with overwriting."""

    expected = __func(test_input)

    g_path = path_graph(graph_len)
    set_edge_attributes(g_path, test_input, "in")
    new_edge_attribute_by_function(g_path, __func, "in", "in", allow_overwriting=True)

    assert get_edge_attributes(g_path, "in") == {
        (i, i + 1): expected for i in range(graph_len - 1)
    }


@pytest.mark.parametrize("attribute_value", list(range(10)))
def test_get_edge_subgraph_with_attribute_value(attribute_value):
    """Test `get_edge_subgraph_with_attribute_value` by design."""
    g_path = path_graph(21)
    set_edge_attributes(
        g_path, {edge: {"attr": int(edge[0] / 2)} for edge in g_path.edges}
    )
    # Compare edges
    assert (
        get_edge_subgraph_with_attribute_value(g_path, "attr", attribute_value).edges
        == Graph(
            [
                (attribute_value * 2, attribute_value * 2 + 1),
                (attribute_value * 2 + 1, attribute_value * 2 + 2),
            ]
        ).edges
    )


@pytest.mark.parametrize("attribute_label", ["attr", "other", 1, 2.0])
def test_get_edge_subgraph_with_attribute_value_no_attribute(attribute_label):
    """Test `get_edge_subgraph_with_attribute_value` when graph has no attribute."""
    g_path = path_graph(5)
    with pytest.raises(ValueError):
        get_edge_subgraph_with_attribute_value(g_path, attribute_label, 0)


def test_get_edge_subgraph_with_attribute_value_empty_graph():
    """Test `get_edge_subgraph_with_attribute_value` when graph is empty."""
    g_path = path_graph(0)
    with pytest.raises(ValueError):
        get_edge_subgraph_with_attribute_value(g_path, "attr", 0)


def test_get_edge_subgraph_with_attribute_value_empty_subgraph():
    """Test `get_edge_subgraph_with_attribute_value` when subgraph is empty."""
    g_path = path_graph(5)
    set_edge_attributes(
        g_path, {edge: {"attr": int(edge[0] / 2)} for edge in g_path.edges}
    )
    with pytest.raises(ValueError):
        get_edge_subgraph_with_attribute_value(g_path, "attr", 10)

    # ValueError
    #     If `minmax_val` is not a tuple of length 2 or None.
    # ValueError
    #     If `minmax_val[0]` is not smaller than `minmax_val[1]`.
    # ValueError
    #     If `attr_type` is not "edge" or "node".


@pytest.mark.parametrize(
    "attr_type,minmax_val,expected",
    [
        ("edge", None, (0, 3)),
        ("edge", (1, 3), (1, 3)),
        ("edge", (None, 4), (0, 4)),
        ("edge", (2, None), (2, 3)),
        ("edge", (1.5, 2), (1.5, 2)),
        ("edge", (None, None), (0, 3)),
        ("node", None, (0, 4)),
        ("node", (0, 4), (0, 4)),
        ("node", (None, 5), (0, 5)),
        ("node", (1, None), (1, 4)),
        ("node", (0.5, 1), (0.5, 1)),
        ("node", (None, None), (0, 4)),
    ],
)
def test_determine_minmax_val(minmax_val, attr_type, expected):
    """Test `determine_minmax_val` by design."""
    graph = path_graph(5)
    set_edge_attributes(graph, {edge: {"attr": edge[0]} for edge in graph.edges})
    set_node_attributes(graph, {node: {"attr": node} for node in graph.nodes})
    assert determine_minmax_val(graph, minmax_val, "attr", attr_type) == expected


@pytest.mark.parametrize("attr_type", ["", None, "Edge", "edge", "Node", "node"])
@pytest.mark.parametrize("attr", ["", None, 1])
@pytest.mark.parametrize(
    "minmax_val",
    [
        "",
        None,
        1,
        (1, 2, 3),
        (2, 1),
        (1, 1),
        (None, 3),
        (None, None),
        (1, None),
    ],
)
def test_determine_minmax_val_invalid_input(minmax_val, attr, attr_type):
    """Test `determine_minmax_val` with invalid input."""
    graph = path_graph(5)
    set_edge_attributes(graph, {edge: {"attr": edge[0]} for edge in graph.edges})
    with pytest.raises(ValueError):
        determine_minmax_val(graph, minmax_val, attr, attr_type)


@pytest.mark.parametrize(
    "key,func,dismiss_none,expected",
    [
        ("attr", sum, True, 6),
        ("attr", sum, False, TypeError),
        ("non_attr", sum, True, KeyError),
        ("attr", max, True, 3),
        ("attr", min, True, 0),
        ("attr", lambda x: x, True, [0, 1, 2, 3]),
        ("attr", lambda x: x, False, [0, None, 1, 2, 3]),
        ("attr", npsum, True, 6),
        ("attr", npsum, False, TypeError),
        ("attr", lambda x: sum(x) / len(x), True, 1.5),
        ("attr", lambda x: sum(x) / len(x), False, TypeError),
    ],
)
def test_aggregate_edge_attr(key, func, dismiss_none, expected):
    """Test `aggregate_edge_attr` by design.
    `g_path` is loop 0-1-2-3-4-0 with weights 0-1-2-3-None.
    """
    g_path = path_graph(5)
    set_edge_attributes(g_path, {edge: {"attr": edge[0]} for edge in g_path.edges})
    g_path.add_edge(0, 4, attr=None)
    # check if expected is child class of Exception
    if isclass(expected) and issubclass(expected, Exception):
        with pytest.raises(expected):
            aggregate_edge_attr(g_path, key, func, dismiss_none)
    else:
        assert aggregate_edge_attr(g_path, key, func, dismiss_none) == expected
