"""Tests for the partitioning utils."""
from os.path import join, exists
from uuid import UUID, uuid4

import pytest
from networkx import gnp_random_graph

from superblockify.partitioning.utils import (
    show_highway_stats,
    save_to_gpkg,
    remove_dead_ends_directed,
    split_up_isolated_edges_directed,
    get_new_node_id,
)


@pytest.mark.parametrize("save_path", [None, "test.gpkg"])
def test_save_to_gpkg(test_city_small_precalculated_copy, save_path):
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


def test_remove_dead_ends_undirected(test_city_all_copy):
    """Test removing dead ends error for undirected graph."""
    _, graph = test_city_all_copy
    graph = graph.to_undirected()
    with pytest.raises(ValueError):
        remove_dead_ends_directed(graph)


def test_split_up_isolated_edges_directed(test_city_small_precalculated_copy):
    """Test splitting up isolated edges by design."""
    part = test_city_small_precalculated_copy
    num_edges, num_nodes = len(part.graph.edges), len(part.graph.nodes)
    split_up_isolated_edges_directed(part.graph, part.sparsified)
    assert len(part.graph.edges) >= num_edges
    assert len(part.graph.nodes) >= num_nodes


@pytest.mark.parametrize("to_directed", ["graph", "sparsified"])
def test_split_up_isolated_edges_undirected(
    to_directed, test_city_small_precalculated_copy
):
    """Test splitting up isolated edges error for undirected graph."""
    part = test_city_small_precalculated_copy
    setattr(part, to_directed, getattr(part, to_directed).to_undirected())
    with pytest.raises(ValueError):
        split_up_isolated_edges_directed(part.graph, part.sparsified)


@pytest.mark.parametrize("degree", [3])
@pytest.mark.parametrize("into,out", [(True, False), (False, True), (True, True)])
def test_split_up_isolated_edges_directed_unsupported_isolated_edge_degree(
    degree, into, out, test_city_all_copy
):
    """Test splitting up isolated edges for graphs where the degree is too high."""
    _, graph = test_city_all_copy
    sparse_graph = graph.edge_subgraph(
        (u, v, k)
        for u, v, k in graph.edges(keys=True)
        if not (graph.degree(u) == 1 and graph.degree(v) == 1)
    )
    # Add #degree nodes+edges to the graph, connecting to a node that is not in the
    # sparsified graph
    connection_node_id = next(node_id for node_id in graph.nodes)
    new_id = get_new_node_id(graph)
    graph.add_node(new_id, x=0, y=0, osmid=0)
    if into:
        graph.add_edges_from(
            [(new_id, connection_node_id, {"osmid": 0}) for _ in range(degree)]
        )
    if out:
        graph.add_edges_from(
            [(connection_node_id, new_id, {"osmid": 0}) for _ in range(degree)]
        )

    with pytest.raises(NotImplementedError):
        split_up_isolated_edges_directed(graph, sparse_graph)


@pytest.mark.parametrize("len_graph", [0, 1, 2, 4, 10])
@pytest.mark.parametrize("patch_uuid", [False, True])
def test_get_new_node_id(len_graph, patch_uuid, monkeypatch):
    """Test getting a new node id."""
    rand_graph = gnp_random_graph(len_graph, 0.5)
    if patch_uuid and len_graph > 0:
        # uuid4().int yields rand_graph.nodes[0] on the first try
        def first_colliding_uuid4():
            # monkeypatch itself back to uuid4
            monkeypatch.setattr("superblockify.partitioning.utils.uuid4", uuid4)
            return UUID(int=list(rand_graph.nodes)[0])

        monkeypatch.setattr(
            "superblockify.partitioning.utils.uuid4", first_colliding_uuid4
        )
    rand_graph.add_node(get_new_node_id(rand_graph))
    assert len(rand_graph.nodes) == len_graph + 1
