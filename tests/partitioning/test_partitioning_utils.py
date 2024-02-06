"""Tests for the partitioning utils."""
from os.path import join, exists
from uuid import UUID, uuid4

import pytest
from networkx import gnp_random_graph
from numpy import int64, float64
from osmnx import graph_to_gdfs, get_undirected

from superblockify.partitioning.utils import (
    show_highway_stats,
    show_graph_stats,
    save_to_gpkg,
    remove_dead_ends_directed,
    split_up_isolated_edges_directed,
    get_new_node_id,
    _make_yaml_compatible,
    reduce_graph,
)


@pytest.mark.parametrize("save_path", [None, "test.gpkg"])
@pytest.mark.parametrize("ltn_boundary", [True, False])
def test_save_to_gpkg(test_city_small_precalculated_copy, save_path, ltn_boundary):
    """Test saving to geopackage."""
    save_path = (
        None
        if save_path is None
        else join(
            test_city_small_precalculated_copy.results_dir,
            test_city_small_precalculated_copy.name + "-filepath.gpkg",
        )
    )
    save_to_gpkg(
        test_city_small_precalculated_copy,
        save_path=save_path,
        ltn_boundary=ltn_boundary,
    )
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


def test_show_graph_stats(test_city_all_copy):
    """Test showing graph stats by design."""
    _, graph = test_city_all_copy
    show_graph_stats(graph)


def test_remove_dead_ends_directed(test_city_all_copy):
    """Test removing dead ends by design."""
    _, graph = test_city_all_copy
    num_edges, num_nodes = graph.number_of_edges(), graph.number_of_nodes()
    remove_dead_ends_directed(graph)
    assert graph.number_of_edges() <= num_edges
    assert graph.number_of_nodes() <= num_nodes


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
    # Check that for each edge with the same geometry, the cell_id is the same
    edges = graph_to_gdfs(part.graph, nodes=False, fill_edge_geometry=True)
    # group by geometry
    edges = edges.groupby("geometry")
    for _, group in edges:
        assert len(group["cell_id"].unique()) == 1


@pytest.mark.parametrize("degree", [3, 4, 5])
def test_split_up_isolated_edges_directed_higher_orders(
    test_city_small_precalculated_copy, degree
):
    """Test splitting up isolated edges by design with higher degree than 2."""
    part = test_city_small_precalculated_copy
    # Get some edge not in the sparsified graph
    rest = part.graph.edge_subgraph(
        [
            (u, v, k)
            for u, v, k in part.graph.edges(keys=True, data=False)
            if (u, v, k) not in part.sparsified.edges(keys=True)
        ]
    )
    rest_un = get_undirected(rest)
    isolated = [
        (u, v)
        for u, v in rest_un.edges()
        if rest_un.degree(u) == 1 and rest_un.degree(v) == 1
    ]
    # double that edge in both directions
    if len(isolated) == 0:
        # pass test if no isolated edges are found
        return
    u_id, v_id = isolated[0]
    part.graph.add_edges_from(
        [(u_id, v_id, -deg) for deg in range(0, degree - 2)]
        + [(v_id, u_id, -deg) for deg in range(0, degree - 2)],
        **part.graph.edges[(u_id, v_id, 0)],
    )
    num_edges, num_nodes = len(part.graph.edges), len(part.graph.nodes)
    split_up_isolated_edges_directed(part.graph, part.sparsified)
    assert len(part.graph.edges) >= num_edges
    assert len(part.graph.nodes) >= num_nodes
    # Check that for each edge with the same geometry, the cell_id is the same
    edges = graph_to_gdfs(part.graph, nodes=False, fill_edge_geometry=True)
    # group by geometry
    edges = edges.groupby("geometry")
    for _, group in edges:
        assert len(group["cell_id"].unique()) == 1


@pytest.mark.parametrize("to_directed", ["graph", "sparsified"])
def test_split_up_isolated_edges_undirected(
    to_directed, test_city_small_precalculated_copy
):
    """Test splitting up isolated edges error for undirected graph."""
    part = test_city_small_precalculated_copy
    setattr(part, to_directed, getattr(part, to_directed).to_undirected())
    with pytest.raises(ValueError):
        split_up_isolated_edges_directed(part.graph, part.sparsified)


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


@pytest.mark.parametrize(
    "test_dict, expected",
    [
        ({}, {}),
        ({"a": 1}, {"a": 1}),
        ({"a": None}, {"a": "None"}),
        ({"a": int64(1)}, {"a": 1}),
        ({"a": float64(1.0)}, {"a": 1.0}),
        ({"a": "1"}, {"a": "1"}),
        ({"a": True}, {"a": True}),
        ({"a": False}, {"a": False}),
        ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}),
        ({"a": (int64(1), float64(1.0))}, {"a": [1, 1.0]}),
        ({"a": {"b": 1}}, {"a": {"b": 1}}),
        ({"a": {"b": int64(1)}}, {"a": {"b": 1}}),
        (
            {"a": {"b": [{"c": "m"}, {"d": (1, 3, float64(4.3))}]}},
            {"a": {"b": [{"c": "m"}, {"d": [1, 3, 4.3]}]}},
        ),
        (1, 1),
        (1.0, 1.0),
        ("1", "1"),
        (True, True),
        (False, False),
        ([1, 2, 3], [1, 2, 3]),
        ((int64(1), float64(1.0)), [1, 1.0]),
    ],
)
def test__make_yaml_compatible(test_dict, expected):
    """Test making a dict yaml compatible."""
    assert _make_yaml_compatible(test_dict) == expected


@pytest.mark.parametrize("max_nodes", [2, 4, 10, 300, 1000, 4000, None])
def test_reduce_graph(test_city_all_copy, max_nodes):
    """Test `reduce_graph` by design."""
    _, graph = test_city_all_copy
    reduced = reduce_graph(graph, max_nodes)
    if max_nodes is None or graph.number_of_nodes() <= max_nodes:
        assert reduced == graph
    else:
        assert reduced.number_of_nodes() <= max_nodes
        assert reduced.number_of_edges() <= graph.number_of_edges()
        if reduced.number_of_edges() > 1:
            # Check for attributes: reduced_population, reduced_area,
            # reduced_street_orientation_order, reduced_circuity_avg, reduced_n,
            # reduced_m
            assert "reduced_population" in reduced.graph
            assert "reduced_area" in reduced.graph
            assert "reduced_street_orientation_order" in reduced.graph
            assert "reduced_circuity_avg" in reduced.graph
            assert "reduced_n" in reduced.graph
            assert "reduced_m" in reduced.graph
