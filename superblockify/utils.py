"""Utility functions for superblockify."""

from itertools import chain

import osmnx as ox
from networkx import Graph
from networkx.utils import graphs_equal


def extract_attributes(graph, edge_attributes, node_attributes):
    """Extract only the specified attributes from a graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to extract attributes from.
    edge_attributes : set
        The edge attributes to keep.
    node_attributes : set
        The node attributes to keep.

    Returns
    -------
    networkx.MultiDiGraph
        The graph with only the specified attributes.
    """

    # Get all unique edge attributes
    all_attributes = set(
        chain.from_iterable(d.keys() for *_, d in graph.edges(data=True))
    )

    excess_attributes = all_attributes - edge_attributes
    # Delete all excess attributes
    for _, _, attr_dict in graph.edges(data=True):
        for att in excess_attributes:
            attr_dict.pop(att, None)

    # Get all unique node attributes
    all_attributes = set(
        chain.from_iterable(d.keys() for *_, d in graph.nodes(data=True))
    )

    excess_attributes = all_attributes - node_attributes
    # Delete all excess attributes
    for _, attr_dict in graph.nodes(data=True):
        for att in excess_attributes:
            attr_dict.pop(att, None)

    return graph


def load_graph_from_place(save_as, search_string, **gfp_kwargs):
    """Load a graph from a place and save it to a file.

    The method filters the attributes of the graph to only include the ones that
    are needed for the package.

    Parameters
    ----------
    save_as : str
        The name of the file to save the graph to.
    search_string : str or list of str
        The place to load the graph from. Find the place name on OpenStreetMap.
        https://nominatim.openstreetmap.org/
    **gfp_kwargs
        Keyword arguments to pass to osmnx.graph_from_place.

    Returns
    -------
    networkx.MultiDiGraph
        The graph loaded from the place.
    """
    graph = ox.graph_from_place(search_string, **gfp_kwargs)
    graph = ox.distance.add_edge_lengths(graph)
    graph = extract_attributes(
        graph,
        edge_attributes={"geometry", "osmid", "length", "highway"},
        node_attributes={"y", "x", "osmid"},
    )
    # Add edge bearings - the precision >1 is important for binning
    graph = ox.add_edge_bearings(graph, precision=2)
    graph = ox.project_graph(graph)
    ox.save_graphml(graph, filepath=save_as)
    return graph


def compare_components_and_partitions(list1, list2):
    """Compare two lists of dictionaries.

    The lists are equal if they have the same length and the dictionaries in
    the lists are equal. If a value of a dictionary is a networkx.Graph, it
    compares the graphs with networkx.graph_equal().

    Parameters
    ----------
    list1 : list of dict
        The first list of dictionaries to compare.
    list2 : list of dict
        The second list of dictionaries to compare.

    Returns
    -------
    bool
        True if the lists are equal, False otherwise.
    """
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if list1[i].keys() != list2[i].keys():
            return False
        for key in list1[i].keys():
            if all(isinstance(x, Graph) for x in [list1[i][key], list2[i][key]]):
                if not graphs_equal(list1[i][key], list2[i][key]):
                    return False
            elif list1[i][key] != list2[i][key]:
                return False
    return True
