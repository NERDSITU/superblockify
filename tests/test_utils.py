"""Tests for the utils module."""
from configparser import ConfigParser

import pytest
from numpy import array, array_equal

from superblockify.utils import (
    load_graph_from_place,
    has_pairwise_overlap,
    compare_dicts,
)

config = ConfigParser()
config.read("config.ini")


def test_load_graph_from_place():
    """Test that the load_graph_from_place function works."""

    graph = load_graph_from_place(
        f"{config['tests']['test_data_path']}/cities/Adliswil.graphml",
        "Adliswil, Bezirk Horgen, Zürich, Switzerland",
        network_type="drive",
    )

    assert graph is not None
    assert len(graph) > 0
    assert graph.size() > 0


@pytest.mark.parametrize(
    "lists,expected",
    [
        ([[]], array([[False]])),
        ([[1]], array([[True]])),
        ([[1, 2], [3, 4]], array([[True, False], [False, True]])),
        ([[1], [1]], array([[True, True], [True, True]])),
        ([[], []], array([[False, False], [False, False]])),
        (
            [[1, 2], [3, 4], [5, 6]],
            array([[True, False, False], [False, True, False], [False, False, True]]),
        ),
        (
            [[1], [1], [2]],
            array([[True, True, False], [True, True, False], [False, False, True]]),
        ),
        (
            [[1, 2], [3, 4], [5, 6], [1]],
            array(
                [
                    [True, False, False, True],
                    [False, True, False, False],
                    [False, False, True, False],
                    [True, False, False, True],
                ]
            ),
        ),
        # long list, range
        (
            [list(range(1000)), list(range(1000))],
            array([[True, True], [True, True]]),
        ),
        (
            [list(range(1000)), list(range(1000, 2000))],
            array([[True, False], [False, True]]),
        ),
        (
            [
                list(range(int(1e5))),
                list(range(int(1e5), int(2e5))),
                list(range(int(1.8e5), int(3e5))),
            ],
            array(
                [
                    [True, False, False],
                    [False, True, True],
                    [False, True, True],
                ],
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test__has_pairwise_overlap(lists, expected):
    """Test `_has_pairwise_overlap` by design."""
    # Check if ndarrays are equal
    # pylint: disable=protected-access
    assert array_equal(has_pairwise_overlap(lists), expected)


@pytest.mark.parametrize(
    "lists",
    [
        [],
        False,
        True,
        1,
        1.0,
        "a",
        None,
        array([]),
        array([[]]),
        array([1]),
        [1],
        [1, 2],
        [[1, 2], [3, 4], [5, 6], 1],
        [[1, 2], [3, 4], [5, 6], "a"],
    ],
)
def test__has_pairwise_overlap_exception(lists):
    """Test `_has_pairwise_overlap` exception handling."""
    with pytest.raises(ValueError):
        # pylint: disable=protected-access
        has_pairwise_overlap(lists)


@pytest.mark.parametrize(
    "dict1,dict2,expected",
    [
        ({}, {}, True),
        ({}, {"a": 1}, False),  # missing key
        ({"a": 1}, {}, False),  # missing key
        ({"a": 1}, {"a": 1}, True),
        ({"a": 1}, {"a": 2}, False),  # different value
        ({"a": 1}, {"b": 1}, False),  # different key
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, True),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, False),  # different value
        ({"a": 1, "b": 2}, {"a": 1, "c": 2}, False),  # different key
        ({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}, False),  # missing key
        ({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2}, False),  # missing key
        ({(1, 2): 1}, {(1, 2): 1}, True),
        ({(1, 2): 1}, {(1, 2): 2}, False),  # different value
        ({"a": array([1, 2])}, {"a": array([1, 2])}, True),
        ({"a": array([1, 2])}, {"a": array([1, 3])}, False),  # different value
        ({"a": array([])}, {"a": array([])}, True),
        ({"a": array([])}, {"a": array([1])}, False),  # different value
        ({"a": array([1])}, {"a": [1]}, False),  # different type
        ({"a": array([[1, 2], [3, 4]])}, {"a": array([[1, 2], [3, 4]])}, True),
        ({"a": array([[1, 2], [3, 4]])}, {"a": array([[1, 2], [3, 5]])}, False),
        ({"a": array([[1, 2], [3, 4]])}, {"a": array([[1, 2], [3, 4], [5, 6]])}, False),
        # differing types
        ({}, None, False),  # None
        ({}, set(), False),  # set
        ({}, [], False),  # list
        ({}, array([]), False),  # ndarray
        ({}, array([[]]), False),  # ndarray
        ({}, tuple(), False),  # tuple
        # nested dicts
        ({"a": {"b": 1}}, {"a": {"b": 1}}, True),
        ({"a": {"b": 1}}, {"a": {"b": 2}}, False),
        (
            {"a": {"a": [1]}, "b": {"b": array([1])}},
            {"a": {"a": [1]}, "b": {"b": array([1])}},
            True,
        ),
        (
            {"a": {"a": [1]}, "b": {"b": array([1])}, "c": {"c": 1}},
            {"a": {"a": [1]}, "b": {"b": array([1])}},
            False,
        ),
        (
            {"a": {"a": {"a": {"a": array([1])}}}},
            {"a": {"a": {"a": {"a": array([1])}}}},
            True,
        ),
    ],
)
def test_compare_dicts(dict1, dict2, expected):
    """Test `compare_dicts`."""
    assert compare_dicts(dict1, dict2) == expected
