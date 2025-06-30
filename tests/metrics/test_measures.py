"""Tests for the measure calculation module."""
from sys import version_info
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
from numpy import full, array, inf, array_equal, int32, int64, allclose
from numpy.random import default_rng
from scipy.sparse.csgraph import dijkstra

from superblockify.metrics.measures import (
    calculate_directness,
    _network_measures_filtered_flattened,
    calculate_global_efficiency,
    calculate_coverage,
    betweenness_centrality,
    _calculate_betweenness,
    __calculate_high_bc_clustering,
    __calculate_high_bc_anisotropy,
    add_relative_changes,
)
from superblockify.utils import __edges_to_1d, percentual_increase, logger


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
            (
                expected
                if expected is not None
                else sum(weights_in) / sum(weights_in + weights_out)
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attr_suffix": None, "seed": 0},
        {"attr_suffix": "_test", "seed": Random(7357)},
    ],
)
@pytest.mark.parametrize("half_k", [False, True])
@pytest.mark.parametrize("max_range", [None, 1000])
def test_betweenness_centrality_options(
    test_one_city_precalculated_copy, kwargs, half_k, max_range
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
        max_range=max_range,
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


def _downcast_(sparse_graph):
    # Try to downcast indices to int32
    if sparse_graph.indices.dtype != int32:
        logger.debug("Downcasting indices to int32.")
        downcasted_indices = sparse_graph.indices.astype(int32)
        if array_equal(downcasted_indices, sparse_graph.indices):
            sparse_graph.indices = downcasted_indices
        else:
            logger.warning("Downcasting indices to int32 failed.")
    # Try to downcast indptr to int32
    if sparse_graph.indptr.dtype != int32:
        logger.debug("Downcasting indptr to int32.")
        downcasted_indptr = sparse_graph.indptr.astype(int32)
        if array_equal(downcasted_indptr, sparse_graph.indptr):
            sparse_graph.indptr = downcasted_indptr
        else:
            logger.warning("Downcasting indptr to int32 failed.")


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
                "normal": array([0.0, 0.0, 0.0, 1 / 6, 1 / 6]),
                "length": array([0.0, 0.0, 0.0, 1 / 12, 1 / 12]),
                "linear": array([0.0, 0.0, 0.0, 1 / 12, 1 / 12]),
            },
        ),
    ],
)
@pytest.mark.skipif(
    version_info < (3, 11), reason="skip for python < 3.11"
)
def test_calculate_betweenness_scales(graph, expected):
    """Test calculation of edge betweenness with scaled results of toy graphs."""
    set_edge_attributes(graph, 1, "weight")
    sparse_graph = to_scipy_sparse_array(graph, nodelist=sorted(graph))
    _downcast_(sparse_graph)
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
        print([expected["edge_normal"][edge] for edge in edge_bc.keys()])
        print(list(edge_bc.values()))
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
                "normal": array([0.0, 2.0, 0.0, 0.0]),
                "length": array([0.0, 1.0, 0.0, 0.0]),
                "linear": array([0.0, 1.0, 0.0, 0.0]),
            },
        ),
    ],
)
@pytest.mark.skipif(
    version_info < (3, 11), reason="skip for python < 3.11"
)
def test__calculate_betweenness_unscaled(graph, expected):
    """Test calculation of betweenness centrality graph to dict, unscaled."""
    sparse_graph = to_scipy_sparse_array(graph, nodelist=sorted(graph))
    _downcast_(sparse_graph)
    padding = len(str(len(graph)))
    edges = __edges_to_1d(
        array([u for u, _ in graph.edges(keys=False)], dtype=int32),
        array([v for _, v in graph.edges(keys=False)], dtype=int32),
        padding,
    )
    edges.sort()
    dist, pred = dijkstra(sparse_graph, return_predecessors=True, directed=True)
    b_c = _calculate_betweenness(
        edges_uv_id=edges,
        pred=pred.astype(int32),
        dist=dist,
        edge_padding=padding,
        index_subset=None,
    )
    assert array_equal(b_c["node"]["normal"], expected["normal"])
    assert allclose(b_c["node"]["length"], expected["length"])
    assert allclose(b_c["node"]["linear"], expected["linear"])


@pytest.mark.parametrize(
    "edges_uv_id,dist,pred,expected",
    [
        (  # fully connected 3 nodes
            array([1, 2, 10, 12, 20, 21], dtype=int64),
            array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
            array([[-9999, 0, 0], [1, -9999, 1], [2, 2, -9999]]),
            {
                "normal": array([0.0, 0.0, 0.0]),
                "length": array([0.0, 0.0, 0.0]),
                "linear": array([0.0, 0.0, 0.0]),
            },
        ),
        (  # path graph 0-1-2
            array([1, 10, 12, 21], dtype=int64),
            array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
            array([[-9999, 0, 1], [1, -9999, 1], [1, 2, -9999]]),
            {
                "normal": array([0.0, 2.0, 0.0]),
                "length": array([0.0, 1.0, 0.0]),
                "linear": array([0.0, 1.0, 0.0]),
            },
        ),
        (  # path graph 0-1-2-3
            array([1, 10, 12, 21, 23, 32], dtype=int64),
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
        (  # path graph 0-1-2-3-4
            array([1, 10, 12, 21, 23, 32, 34, 43], dtype=int64),
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
        (  # star graph, 0 center
            array([1, 2, 3, 10, 20, 30], dtype=int64),
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
def test__calculate_betweenness_unscaled_paths(edges_uv_id, dist, pred, expected):
    """Test calculation of betweenness centrality paths to dict, unscaled."""
    padding = len(str(len(dist)))
    b_c = _calculate_betweenness(
        edges_uv_id, pred.astype(int32), dist, edge_padding=padding, index_subset=None
    )
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


@pytest.fixture(scope="module", params=[10, 100, 1000, 60000])
def clustering_data(request):
    """Generate random data for clustering tests."""
    rng = default_rng(29384)
    coord = array(
        [
            (  # x-coord
                rng.uniform(low=-10, high=10, size=request.param)
                + rng.uniform(low=-180, high=180)
            ),
            (  # y-coord
                rng.uniform(low=-10, high=10, size=request.param)
                + rng.uniform(low=-90, high=90)
            ),  # betweenness centrality
            rng.uniform(low=0, high=1, size=request.param),
        ]
    ).T
    return coord[coord[:, 2].argsort()], rng.integers(low=0, high=request.param)


def test___calculate_high_bc_clustering(
    clustering_data,
):  # pylint: disable=redefined-outer-name
    """Test calculation of betweenness centrality clustering."""
    assert 0.0 < __calculate_high_bc_clustering(*clustering_data) < 1.0


@pytest.mark.parametrize(
    "coord_bc,threshold_idx",
    [
        (array([]), 0),  # length 0
        (array([[0, 0, 0]]), 1),  # length 1
        (array([[0, 0, 0], [1, 1, 1]]), 2),  # index out of bounds
    ],
)
def test___calculate_high_bc_clustering_faulty(coord_bc, threshold_idx):
    """Test error catching for betweenness centrality clustering."""
    with pytest.raises(ValueError):
        __calculate_high_bc_clustering(coord_bc, threshold_idx)


def test___calculate_high_bc_anisotropy(
    clustering_data,
):  # pylint: disable=redefined-outer-name
    """Test calculation of betweenness centrality anisotropy."""
    coord_high_bc = clustering_data[0][clustering_data[1] :, :2]
    anisotropy = __calculate_high_bc_anisotropy(coord_high_bc)
    assert 1.0 <= anisotropy
    # check invariance to x and y coordinate swap
    assert isclose(__calculate_high_bc_anisotropy(coord_high_bc[:, ::-1]), anisotropy)


@pytest.mark.parametrize(
    "coords,expected",
    [
        ([[0, 0], [1, 0], [0, 1], [1, 1]], 1.0),  # square, round distribution
        ([[-20, 10], [-10, 10], [-20, 20], [-10, 20]], 1.0),  # square, round distr.
        ([[1, 0], [0, 1], [1, 2], [2, 1]], 1.0),  # diamond, round distribution
        ([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], 1.0),  # square + center
        ([[0, 0], [1, 0], [0, 2], [1, 2]], 4.0),  # 2:1 rectangle, long distr.
        ([[0, 0], [2, 0], [0, 1], [2, 1]], 4.0),  # 1:2 rectangle, long distr.
        ([[0, 0], [1, 0], [0, 2], [1, 2], [0.5, 1]], 4.0),  # 2:1 rect. + center
        ([[0, 0], [0, 1]], inf),  # vertical line, infinite anisotropy
        ([[0, 0], [1, 0]], inf),  # horizontal line, infinite anisotropy
    ],
)
def test___calculate_high_bc_anisotropy_special_cases(coords, expected):
    """Test calculation of betweenness centrality anisotropy."""
    assert __calculate_high_bc_anisotropy(array(coords)) == expected


@pytest.mark.parametrize("coords", [[], [[0, 0]], [[1, 1]]])
def test___test___calculate_high_bc_anisotropy_faulty(coords):
    """Test error catching of betweenness centrality anisotropy."""
    with pytest.raises(ValueError):
        __calculate_high_bc_anisotropy(array(coords))


@pytest.mark.parametrize(
    "attr_pairs", [("a", "b"), [("a", "b")], [("a", "b"), ("c", "d")]]
)
@pytest.mark.parametrize(
    "list_a,list_b",
    [
        ([], []),  # empty lists
        ([9.8], [9.8]),  # single element lists
        ([1, 2, 3], [2, 4, 6]),
        ([0, 4, inf], [1, -2, 3]),
    ],
)
def test_add_relative_changes(list_a, list_b, attr_pairs):
    """Test error catching of add_relative_changes."""
    test_dict_list = [
        {"a": val_a, "b": val_b, "c": val_b, "d": val_a}
        for val_a, val_b in zip(list_a, list_b)
    ]
    add_relative_changes(test_dict_list, attr_pairs)
    # now `change_a` should have value percentual_increase(val_a, val_b)
    assert allclose(
        [val["change_a"] for val in test_dict_list],
        [percentual_increase(val_a, val_b) for val_a, val_b in zip(list_a, list_b)],
    )
    if isinstance(attr_pairs, list) and len(attr_pairs) == 2:
        # now `change_c` should have value percentual_increase(val_b, val_a)
        assert allclose(
            [val["change_c"] for val in test_dict_list],
            [percentual_increase(val_b, val_a) for val_a, val_b in zip(list_a, list_b)],
        )


def test_add_relative_changes_key_error():
    """Test error catching of add_relative_changes."""
    with pytest.raises(KeyError):
        add_relative_changes([{"a": 1, "b": 2}], [("a", "c")])
