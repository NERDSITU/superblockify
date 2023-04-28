"""Tests for the measure calculation module."""
from dataclasses import dataclass
from math import isclose

import pytest
from networkx import MultiDiGraph, path_graph
from numpy import full, array, inf, array_equal

from superblockify.metrics.measures import (
    calculate_directness,
    _network_measures_filtered_flattened,
    calculate_global_efficiency,
    calculate_coverage,
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
        ([1, '1'], [1, 2], TypeError),
        ([1, 2], [1, '1'], TypeError),
        ([True, 2], [1, 2], 0.5), # bools are ints 1 and 0
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
            print(calculate_coverage(Part(graph, sparsified), "weight"))
    else:
        # otherwise assert that the expected value is returned
        assert isclose(
            calculate_coverage(Part(graph, sparsified), "weight"),
            expected
            if expected is not None
            else sum(weights_in) / sum(weights_in + weights_out),
        )
