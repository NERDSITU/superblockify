"""Tests for the utils module."""
from configparser import ConfigParser

import pytest
from numpy import array, array_equal

from superblockify.utils import load_graph_from_place, has_pairwise_overlap

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]


def test_load_graph_from_place():
    """Test that the load_graph_from_place function works."""

    graph = load_graph_from_place(
        f"{TEST_DATA}/cities/Adliswil.graphml",
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
