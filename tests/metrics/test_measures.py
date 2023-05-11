"""Tests for the measure calculation module."""
from dataclasses import dataclass
from math import isclose
from random import Random

import pytest
from networkx import (
    MultiDiGraph,
    path_graph,
    get_edge_attributes,
    to_scipy_sparse_array,
    Graph,
    set_edge_attributes,
    star_graph,
    complete_graph,
    wheel_graph,
)
from numpy import full, array, inf, array_equal, int32, allclose
from scipy.sparse.csgraph import dijkstra

from superblockify.metrics.measures import (
    calculate_directness,
    _network_measures_filtered_flattened,
    calculate_global_efficiency,
    calculate_coverage,
    betweenness_centrality,
    _calculate_betweenness,
)


@pytest.mark.parametrize(
    "distance1,distance2,expected",
    [
        # 0, inf and the diagonal are always excluded
        (full((3, 3), 1), full((3, 3), 1), (full(6, 1), full(6, 1))),
        (full((3, 3), 1), full((3, 3), 2), (full(6, 1), full(6, 2))),
        (
            full((1000, 1000), 1),
            full((1000, 1000), 2),
            (full(1000000 - 1000, 1), full(1000000 - 1000, 2)),
        ),
        (
            array([[0, 1, 2], [1, 0, 3], [1, 1, 0]]),
            array([[0, 2, 3], [1, 0, 1], [1, 6, 0]]),
            (array([1, 2, 1, 3, 1, 1]), array([2, 3, 1, 1, 1, 6])),
        ),
        # check 0 is excluded
        (array([[0, 1], [0, 0]]), array([[0, 1], [0, 0]]), (array([1]), array([1]))),
        (array([[0, 20], [0, 0]]), array([[0, 2], [0, 0]]), (array([20]), array([2]))),
        (array([[0, 0], [4, 0]]), array([[0, 9], [12, 0]]), (array([4]), array([12]))),
        # check inf is excluded
        (array([[0, inf], [1, 0]]), array([[0, 2], [3, 0]]), (array([1]), array([3]))),
        (
            array([[0, 1, 2], [inf, 0, 3], [4, 5, 0]]),
            array([[0, 6, 5], [4, 0, 3], [2, 1, 0]]),
            (array([1, 2, 3, 4, 5]), array([6, 5, 3, 2, 1])),
        ),
        (
            array([[0, 1, 2], [3, 0, 4], [5, 6, 0]]),
            array([[0, 5, 4], [inf, 0, 3], [2, 1, 0]]),
            (array([1, 2, 4, 5, 6]), array([5, 4, 3, 2, 1])),
        ),
        (
            array([[0, 2, 3], [1, 0, 1], [1, inf, 0]]),
            array([[0, 1, 2], [inf, 0, 3], [6, 1, 0]]),
            (array([2, 3, 1, 1]), array([1, 2, 3, 6])),
        ),
        (
            array([[0, 1, 2], [1, 0, 3], [1, 1, 0]]),
            array([[0, 2, 3], [1, 0, 1], [1, inf, 0]]),
            (array([1, 2, 1, 3, 1]), array([2, 3, 1, 1, 1])),
        ),
    ],
)
def test__network_measures_filtered_flattened(distance1, distance2, expected):
    """Test flattening and filtering the distance matrices"""
    dist_matrix = {"S": distance1, "N": distance2}
    assert array_equal(
        _network_measures_filtered_flattened(dist_matrix, "S", "N"), expected
    )


@pytest.mark.parametrize(
    "distance1,distance2,expected",
    [
        # 0, inf and the diagonal are always excluded
        (full((3, 3), 1), full((3, 3), 1), 1),
        (full((3, 3), 1), full((3, 3), 2), 0.5),
        (full((1000, 1000), 1), full((1000, 1000), 2), 0.5),
        (
            array([[0, 1, 2], [1, 0, 3], [1, 1, 0]]),
            array([[0, 2, 3], [1, 0, 1], [1, 6, 0]]),
            (1 / 2 + 2 / 3 + 1 / 1 + 3 / 1 + 1 / 6 + 1 / 1) / 6,
        ),
        # check 0 is excluded
        (array([[0, 1], [0, 0]]), array([[0, 1], [0, 0]]), 1),
        (array([[0, 20], [0, 0]]), array([[0, 2], [0, 0]]), 10),
        # check inf is excluded
        (array([[0, inf], [1, 0]]), array([[0, 1], [1, 0]]), 1),
        (
            array([[0, 1, 6], [inf, 0, 1], [1, 1, 0]]),
            array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            (1 + 6 + 1 + 1 + 1) / 5,
        ),
        (
            array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            array([[0, 1, 2], [inf, 0, 1], [1, 1, 0]]),
            (1 + 1 / 2 + 1 + 1 + 1) / 5,
        ),
        (
            array([[0, 2, 3], [1, 0, 1], [1, inf, 0]]),
            array([[0, 1, 2], [inf, 0, 3], [6, 1, 0]]),
            (2 / 1 + 3 / 2 + 1 / 3 + 1 / 6) / 4,
        ),
        (
            array([[0, 1, 2], [1, 0, 3], [1, 1, 0]]),
            array([[0, 2, 3], [1, 0, 1], [1, inf, 0]]),
            (1 / 2 + 2 / 3 + 1 / 1 + 3 / 1 + 1 / 1) / 5,
        ),
        # check diagonal is excluded
        (array([[10, 1], [1, 10]]), array([[1, 1], [1, 1]]), 1),
        (array([[1, 10], [10, 1]]), array([[1, 1], [1, 1]]), 10),
    ],
)
def test_calculate_directness(distance1, distance2, expected):
    """Test calculating directness"""
    dist_matrix = {"S": distance1, "N": distance2}
    assert calculate_directness(dist_matrix, "S", "N") == expected


@pytest.mark.parametrize(
    "distance1,distance2,expected",
    [
        # 0, inf and the diagonal are always excluded
        (full((3, 3), 1), full((3, 3), 1), 1),
        (full((3, 3), 1), full((3, 3), 2), 2),
        (full((1000, 1000), 1), full((1000, 1000), 2), 2),
        (
            array([[0, 1, 2], [1, 0, 3], [1, 1, 0]]),
            array([[0, 2, 3], [1, 0, 1], [1, 6, 0]]),
            (1 / 1 + 1 / 2 + 1 / 1 + 1 / 3 + 1 / 1 + 1 / 1)
            / (1 / 2 + 1 / 3 + 1 / 1 + 1 / 1 + 1 / 6 + 1 / 1),
        ),
        # check 0 is excluded
        (array([[0, 1], [0, 0]]), array([[0, 1], [0, 0]]), 1),
        (array([[0, 20], [0, 0]]), array([[0, 2], [0, 0]]), 0.1),
        # check inf is excluded
        (array([[0, inf], [1, 0]]), array([[0, 1], [1, 0]]), 1),
        (
            array([[0, 1, 6], [inf, 0, 1], [1, 1, 0]]),
            array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            (1 / 1 + 1 / 6 + 1 / 1 + 1 / 1 + 1 / 1)
            / (1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1),
        ),
        (
            array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            array([[0, 1, 2], [inf, 0, 1], [1, 1, 0]]),
            (1 / 1 + 1 / 1 + 1 / 1 + 1 / 1 + 1 / 1)
            / (1 / 1 + 1 / 2 + 1 / 1 + 1 / 1 + 1 / 1),
        ),
        (
            array([[0, 2, 3], [1, 0, 1], [1, inf, 0]]),
            array([[0, 1, 2], [inf, 0, 3], [6, 1, 0]]),
            (1 / 2 + 1 / 3 + 1 / 1 + 1 / 1) / (1 / 1 + 1 / 2 + 1 / 3 + 1 / 6),
        ),
        (
            array([[0, 1, 2], [1, 0, 3], [1, 1, 0]]),
            array([[0, 2, 3], [1, 0, 1], [1, inf, 0]]),
            (1 / 1 + 1 / 2 + 1 / 1 + 1 / 3 + 1 / 1)
            / (1 / 2 + 1 / 3 + 1 / 1 + 1 / 1 + 1 / 1),
        ),
        # check diagonal is excluded
        (array([[10, 1], [1, 10]]), array([[1, 1], [1, 1]]), 1),
        (array([[1, 10], [10, 1]]), array([[1, 1], [1, 1]]), 0.1),
    ],
)
def test_calculate_global_efficiency(distance1, distance2, expected):
    """Test calculating directness"""
    dist_matrix = {"S": distance1, "N": distance2}
    assert calculate_global_efficiency(dist_matrix, "S", "N") == expected


@pytest.mark.parametrize(
    "weights_in,weights_out,expected",
    [
        ([1], [], 1),
        ([1, 2], [], 1),
        ([1], [1], 0.5),
        ([], [1], 0),
        ([1], [1, 2], 0.25),
        ([1, 2], [1], 0.75),
        ([1, 2], [1, 2], 0.5),
        ([1, 2, 3], [1, 2], 6 / 9),
        ([1, 2], [1, 2, 3], 3 / 9),
        (list(range(1, 10)), list(range(1, 103)), None),
        ([], [], ValueError),
        ([1, "1"], [1, 2], TypeError),
        ([1, 2], [1, "1"], TypeError),
        ([True, 2], [1, 2], 0.5),  # bools are ints 1 and 0
        ([False, True], [False, False], 1),
    ],
)
def test_calculate_coverage(weights_in, weights_out, expected):
    """Test calculating coverage"""

    # make path graph with len(weights_in) + len(weights_out) edges
    graph = path_graph(
        len(weights_in) + len(weights_out) + 1, create_using=MultiDiGraph
    )
    # add weights to edges enumerate weights_in + weights_out in one
    for i, weight in enumerate(weights_in + weights_out):
        graph.edges[i, i + 1, 0]["weight"] = weight
    # sparsified graph is subview of graph with only weights_out edges
    sparsified = graph.edge_subgraph(
        [
            (i, i + 1, 0)
            for i in range(len(weights_in), len(weights_in) + len(weights_out))
        ]
    )

    @dataclass
    class Part:
        """Mock class to pass as partitioner"""

        graph: MultiDiGraph
        sparsified: MultiDiGraph

    # if expected is any type of error, assert that it is raised
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            calculate_coverage(Part(graph, sparsified), "weight")
    else:
        # otherwise assert that the expected value is returned
        assert isclose(
            calculate_coverage(Part(graph, sparsified), "weight"),
            expected
            if expected is not None
            else sum(weights_in) / sum(weights_in + weights_out),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attr_suffix": None, "seed": 0},
        {"attr_suffix": "_test", "seed": Random(7357)},
    ],
)
@pytest.mark.parametrize("half_k", [False, True])
def test_betweenness_centrality_options(
    test_one_city_precalculated_copy, kwargs, half_k
):
    """Test betweenness centrality calculation"""
    part = test_one_city_precalculated_copy
    part.calculate_metrics_before()
    betweenness_centrality(
        part.graph,
        part.metric.node_list,
        part.metric.distance_matrix["S"],
        part.metric.predecessor_matrix["S"],
        k=None if half_k else int(part.graph.number_of_nodes() / 2),
        **kwargs,
    )
    for bc_type in ["normal", "length", "linear"]:
        assert all(
            data[
                f"node_betweenness_{bc_type}"
                f"{kwargs['attr_suffix'] if kwargs['attr_suffix'] is not None else ''}"
            ]
            >= 0
            for data in part.graph.nodes.values()
        )
        edge_label = (
            f"edge_betweenness_{bc_type}"
            f"{kwargs['attr_suffix'] if kwargs['attr_suffix'] is not None else ''}"
        )
        assert len(part.graph.edges()) == len(
            get_edge_attributes(part.graph, edge_label)
        )
        assert all(
            vals >= 0 for vals in get_edge_attributes(part.graph, edge_label).values()
        )


@pytest.mark.parametrize(
    "graph,expected",
    [
        (
            MultiDiGraph(path_graph(1)),
            {
                "normal": array([0.0]),
                "length": array([0.0]),
                "linear": array([0.0]),
            },
        ),
        (
            MultiDiGraph(path_graph(2)),
            {
                "normal": array([0.0, 0.0]),
                "length": array([0.0, 0.0]),
                "linear": array([0.0, 0.0]),
                "edge_normal": {(0, 1, 0): 0.5, (1, 0, 0): 0.5},
                "edge_length": {(0, 1, 0): 0.5, (1, 0, 0): 0.5},
                "edge_linear": {(0, 1, 0): 0.5, (1, 0, 0): 0.5},
            },
        ),
        (
            MultiDiGraph(path_graph(3)),
            {
                "normal": array([0.0, 1.0, 0.0]),
                "length": array([0.0, 0.5, 0.0]),
                "linear": array([0.0, 0.5, 0.0]),
                "edge_normal": {
                    (0, 1, 0): 1 / 3,
                    (1, 0, 0): 1 / 3,
                    (1, 2, 0): 1 / 3,
                    (2, 1, 0): 1 / 3,
                },
                "edge_length": {
                    (0, 1, 0): 1 / 3,
                    (1, 0, 0): 1 / 4,
                    (1, 2, 0): 1 / 4,
                    (2, 1, 0): 1 / 3,
                },
                "edge_linear": {
                    (0, 1, 0): 1 / 3,
                    (1, 0, 0): 1 / 3,
                    (1, 2, 0): 1 / 3,
                    (2, 1, 0): 1 / 3,
                },
            },
        ),
        (
            MultiDiGraph(path_graph(4)),
            {
                "normal": array([0.0, 2 / 3, 2 / 3, 0.0]),
                "length": array([0.0, 7 / 18, 7 / 18, 0.0]),
                "linear": array([0.0, 4 / 9, 4 / 9, 0.0]),
            },
        ),
        (
            MultiDiGraph(star_graph(3)),
            {
                "normal": array([1.0, 0.0, 0.0, 0.0]),
                "length": array([0.5, 0.0, 0.0, 0.0]),
                "linear": array([0.5, 0.0, 0.0, 0.0]),
                "edge_normal": {
                    (0, 1, 0): 1 / 4,
                    (1, 0, 0): 1 / 4,
                    (0, 2, 0): 1 / 4,
                    (2, 0, 0): 1 / 4,
                    (0, 3, 0): 1 / 4,
                    (3, 0, 0): 1 / 4,
                },
                "edge_length": {
                    (0, 1, 0): 1 / 6,
                    (1, 0, 0): 1 / 4,
                    (0, 2, 0): 1 / 6,
                    (2, 0, 0): 1 / 4,
                    (0, 3, 0): 1 / 6,
                    (3, 0, 0): 1 / 4,
                },
                "edge_linear": {
                    (0, 1, 0): 1 / 4,
                    (1, 0, 0): 1 / 4,
                    (0, 2, 0): 1 / 4,
                    (2, 0, 0): 1 / 4,
                    (0, 3, 0): 1 / 4,
                    (3, 0, 0): 1 / 4,
                },
            },
        ),
        (
            MultiDiGraph(complete_graph(5)),
            {
                "normal": array([0.0, 0.0, 0.0, 0.0, 0.0]),
                "length": array([0.0, 0.0, 0.0, 0.0, 0.0]),
                "linear": array([0.0, 0.0, 0.0, 0.0, 0.0]),
            },
        ),
        (
            MultiDiGraph(wheel_graph(5)),
            {
                "normal": array([1 / 3, 0.0, 0.0, 0.0, 0.0]),
                "length": array([1 / 6, 0.0, 0.0, 0.0, 0.0]),
                "linear": array([1 / 6, 0.0, 0.0, 0.0, 0.0]),
                "edge_normal": {
                    (0, 1, 0): 0.1,
                    (1, 0, 0): 0.1,
                    (0, 2, 0): 0.1,
                    (2, 0, 0): 0.1,
                    (0, 3, 0): 0.1,
                    (3, 0, 0): 0.1,
                    (0, 4, 0): 0.1,
                    (4, 0, 0): 0.1,
                    (1, 2, 0): 0.05,
                    (2, 1, 0): 0.05,
                    (2, 3, 0): 0.05,
                    (3, 2, 0): 0.05,
                    (3, 4, 0): 0.05,
                    (4, 3, 0): 0.05,
                    (4, 1, 0): 0.05,
                    (1, 4, 0): 0.05,
                },
            },
        ),
    ],
)
def test_calculate_betweenness_scales(graph, expected):
    """Test calculation of edge betweenness with scaled results of toy graphs."""
    set_edge_attributes(graph, 1, "weight")
    sparse_graph = to_scipy_sparse_array(graph, nodelist=sorted(graph))
    dist, pred = dijkstra(sparse_graph, return_predecessors=True, directed=True)
    betweenness_centrality(graph, list(sorted(graph)), dist, pred, weight="weight")
    assert array_equal(
        [graph.nodes[node]["node_betweenness_normal"] for node in sorted(graph)],
        expected["normal"],
    )
    assert allclose(
        [graph.nodes[node]["node_betweenness_length"] for node in sorted(graph)],
        expected["length"],
    )
    assert allclose(
        [graph.nodes[node]["node_betweenness_linear"] for node in sorted(graph)],
        expected["linear"],
    )
    if "edge_normal" in expected:
        edge_bc = get_edge_attributes(graph, "edge_betweenness_normal")
        assert allclose(
            list(edge_bc.values()),
            [expected["edge_normal"][edge] for edge in edge_bc.keys()],
        )
    if "edge_length" in expected:
        edge_bc = get_edge_attributes(graph, "edge_betweenness_length")
        assert allclose(
            list(edge_bc.values()),
            [expected["edge_length"][edge] for edge in edge_bc.keys()],
        )
    if "edge_linear" in expected:
        edge_bc = get_edge_attributes(graph, "edge_betweenness_linear")
        assert allclose(
            list(edge_bc.values()),
            [expected["edge_linear"][edge] for edge in edge_bc.keys()],
        )


@pytest.mark.parametrize(
    "graph,expected",
    [
        (
            MultiDiGraph(Graph([(0, 1)])),
            {
                "normal": array([0.0, 0.0]),
                "length": array([0.0, 0.0]),
                "linear": array([0.0, 0.0]),
            },
        ),
        (
            MultiDiGraph(Graph([(0, 1), (1, 0)])),
            {
                "normal": array([0.0, 0.0]),
                "length": array([0.0, 0.0]),
                "linear": array([0.0, 0.0]),
            },
        ),
        (
            MultiDiGraph(Graph([(0, 1), (1, 2)])),
            {
                "normal": array([0.0, 2.0, 0.0]),
                "length": array([0.0, 1.0, 0.0]),
                "linear": array([0.0, 1.0, 0.0]),
            },
        ),
        (
            MultiDiGraph(Graph([(0, 1), (1, 2), (2, 3)])),
            {
                "normal": array([0.0, 4.0, 4.0, 0.0]),
                "length": array([0.0, 7 / 3, 7 / 3, 0.0]),
                "linear": array([0.0, 8 / 3, 8 / 3, 0.0]),
            },
        ),
        (
            MultiDiGraph(Graph([(0, 1), (1, 2), (2, 3), (3, 4)])),
            {
                "normal": array([0.0, 6.0, 8.0, 6.0, 0.0]),
                "length": array([0.0, 43 / 12, 17 / 3, 43 / 12, 0.0]),
                "linear": array([0.0, 53 / 12, 25 / 3, 53 / 12, 0.0]),
            },
        ),
        (
            MultiDiGraph(Graph([(0, 1), (0, 2), (0, 3)])),
            {
                "normal": array([6.0, 0.0, 0.0, 0.0]),
                "length": array([3.0, 0.0, 0.0, 0.0]),
                "linear": array([3.0, 0.0, 0.0, 0.0]),
            },
        ),
        (
            MultiDiGraph(
                Graph(
                    [
                        (0, 1, {"weight": 2.0}),
                        (0, 2, {"weight": 2.0}),
                        (0, 3, {"weight": 2.0}),
                    ]
                )
            ),
            {
                "normal": array([6.0, 0.0, 0.0, 0.0]),
                "length": array([1.5, 0.0, 0.0, 0.0]),
                "linear": array([3.0, 0.0, 0.0, 0.0]),
            },
        ),
        (
            MultiDiGraph([(0, 1), (1, 2), (2, 3), (3, 0)]),
            {
                "normal": array([3.0, 3.0, 3.0, 3.0]),
                "length": array([11 / 6, 11 / 6, 11 / 6, 11 / 6]),
                "linear": array([13 / 6, 13 / 6, 13 / 6, 13 / 6]),
            },
        ),
        (
            MultiDiGraph(Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])),
            {
                "normal": array([2.0, 0.0, 0.0, 0.0]),
                "length": array([1.0, 0.0, 0.0, 0.0]),
                "linear": array([1.0, 0.0, 0.0, 0.0]),
            },
        ),
    ],
)
def test__calculate_betweenness_unscaled(graph, expected):
    """Test calculation of betweenness centrality graph to dict, unscaled."""
    sparse_graph = to_scipy_sparse_array(graph, nodelist=sorted(graph))
    dist, pred = dijkstra(sparse_graph, return_predecessors=True, directed=True)
    b_c = _calculate_betweenness(pred.astype(int32), dist, index_subset=None)
    assert array_equal(b_c["node"]["normal"], expected["normal"])
    assert allclose(b_c["node"]["length"], expected["length"])
    assert allclose(b_c["node"]["linear"], expected["linear"])


@pytest.mark.parametrize(
    "dist,pred,expected",
    [
        (
            array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            {
                "normal": array([0.0, 0.0, 0.0]),
                "length": array([0.0, 0.0, 0.0]),
                "linear": array([0.0, 0.0, 0.0]),
            },
        ),
        (
            array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            array([[-9999, 0, 1], [1, -9999, 1], [1, 2, -9999]]),
            {
                "normal": array([0.0, 2.0, 0.0]),
                "length": array([0.0, 1.0, 0.0]),
                "linear": array([0.0, 1.0, 0.0]),
            },
        ),
        (
            array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]),
            array(
                [[-9999, 0, 1, 2], [1, -9999, 1, 2], [1, 2, -9999, 2], [1, 2, 3, -9999]]
            ),
            {
                "normal": array([0.0, 4.0, 4.0, 0.0]),
                "length": array([0.0, 7 / 3, 7 / 3, 0.0]),
                "linear": array([0.0, 8 / 3, 8 / 3, 0.0]),
            },
        ),
        (
            array(
                [
                    [0, 1, 2, 3, 4],
                    [1, 0, 1, 2, 3],
                    [2, 1, 0, 1, 2],
                    [3, 2, 1, 0, 1],
                    [4, 3, 2, 1, 0],
                ]
            ),
            array(
                [
                    [-9999, 0, 1, 2, 3],
                    [1, -9999, 1, 2, 3],
                    [1, 2, -9999, 2, 3],
                    [1, 2, 3, -9999, 3],
                    [1, 2, 3, 4, -9999],
                ]
            ),
            {
                "normal": array([0.0, 6.0, 8.0, 6.0, 0.0]),
                "length": array([0.0, 43 / 12, 17 / 3, 43 / 12, 0.0]),
                "linear": array([0.0, 53 / 12, 25 / 3, 53 / 12, 0.0]),
            },
        ),
        (
            array([[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]]),
            array(
                [[-9999, 0, 0, 0], [1, -9999, 0, 0], [2, 0, -9999, 0], [3, 0, 0, -9999]]
            ),
            {
                "normal": array([6.0, 0.0, 0.0, 0.0]),
                "length": array([3.0, 0.0, 0.0, 0.0]),
                "linear": array([3.0, 0.0, 0.0, 0.0]),
            },
        ),
    ],
)
def test__calculate_betweenness_unscaled_paths(dist, pred, expected):
    """Test calculation of betweenness centrality paths to dict, unscaled."""
    b_c = _calculate_betweenness(pred.astype(int32), dist, index_subset=None)
    assert array_equal(b_c["node"]["normal"], expected["normal"])
    assert allclose(b_c["node"]["length"], expected["length"])
    assert allclose(b_c["node"]["linear"], expected["linear"])


@pytest.mark.parametrize("graph", [path_graph(4), complete_graph(4), star_graph(4)])
def test_betweenness_centrality_weight_missing(graph):
    """Test betweenness centrality with missing weight."""
    set_edge_attributes(graph, 1, "weight")
    graph = MultiDiGraph(graph)
    # delete one weight attribute
    del graph.edges[0, 1, 0]["weight"]
    with pytest.raises(ValueError):
        betweenness_centrality(graph, None, None, None, weight="weight")
