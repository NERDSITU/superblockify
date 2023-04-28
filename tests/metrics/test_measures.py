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

