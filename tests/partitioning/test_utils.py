"""Tests for the partitioning utils."""
from os.path import join, exists

import pytest

from superblockify.partitioning.utils import (
    show_highway_stats,
    save_to_gpkg,
    remove_dead_ends_directed,
)


@pytest.mark.parametrize("save_path", [None, "test.gpkg"])
def test_save_to_gpkg(
    test_city_small_precalculated_copy, save_path, _teardown_test_folders
):
    """Test saving to geopackage."""
    save_path = (
        None
        if save_path is None
        else join(
            test_city_small_precalculated_copy.results_dir,
            test_city_small_precalculated_copy.name + "-filepath.gpkg",
        )
    )
    save_to_gpkg(test_city_small_precalculated_copy, save_path=save_path)
    # Check that the file exists
    assert exists(
        join(
            test_city_small_precalculated_copy.results_dir,
            test_city_small_precalculated_copy.name
            + ("-filepath.gpkg" if save_path else ".gpkg"),
        )
    )


@pytest.mark.parametrize(
    "replace_attibute",
    [
        [("sparsified", None)],  # no sparsified graph
        [("sparsified", 1)],  # wrong type
        [("components", None), ("partitions", None)],  # no components or partitions
        [("components", None), ("partitions", 1)],  # wrong type
        [("components", 1)],  # wrong type
        [("components", [1])],  # wrong type
        [("components", [{"subgraph": None}])],  # no 'name' attribute
        [("components", [{"name": None}])],  # no 'subgraph' attribute
    ],
)
def test_save_to_gpkg_faulty_subgraphs(
    test_one_city_precalculated_copy, replace_attibute
):
    """Test saving to geopackage with faulty subgraphs."""
    for attribute, value in replace_attibute:
        setattr(test_one_city_precalculated_copy, attribute, value)
    with pytest.raises(ValueError):
        save_to_gpkg(test_one_city_precalculated_copy)


def test_show_highway_stats(test_city_all_copy):
    """Test showing highway stats by design."""
    _, graph = test_city_all_copy
    show_highway_stats(graph)


def test_remove_dead_ends_directed(test_city_all_copy):
    """Test removing dead ends by design."""
    _, graph = test_city_all_copy
    num_edges, num_nodes = len(graph.edges), len(graph.nodes)
    remove_dead_ends_directed(graph)
    assert len(graph.edges) <= num_edges
    assert len(graph.nodes) <= num_nodes


def test_remove_dead_ends_unidirected(test_city_all_copy):
    """Test removing dead ends error for undirected graph."""
    _, graph = test_city_all_copy
    graph = graph.to_undirected()
    with pytest.raises(ValueError):
        remove_dead_ends_directed(graph)
