"""Utility functions for superblockify."""

from itertools import chain
from re import match

import osmnx as ox
from networkx import Graph, is_isomorphic
from numpy import zeros, array, fill_diagonal, ndarray, array_equal


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
        Can otherwise be OSM relation ID or a list of those. They have the format
        "R1234567". Not mixed with place names.
    **gfp_kwargs
        Keyword arguments to pass to osmnx.graph_from_place.

    Returns
    -------
    networkx.MultiDiGraph
        The graph loaded from the place.
    """

    # check the format of the search string, match to regex R\d+, str or list of str
    # use re.match(r"R\d+", search_string) or to all elements in list
    if (
        isinstance(search_string, str)
        and match(r"R\d+", search_string)
        or isinstance(search_string, list)
        and all(isinstance(s, str) and match(r"R\d+", s) for s in search_string)
    ):
        mult_polygon = ox.geocode_to_gdf(search_string, by_osmid=True)
        # geopandas.GeoDataFrame, every column is polygon
        # make shapely.geometry.MultiPolygon from all polygons
        mult_polygon = mult_polygon["geometry"].unary_union
        graph = ox.graph_from_polygon(mult_polygon, **gfp_kwargs)
    else:
        graph = ox.graph_from_place(search_string, **gfp_kwargs)

    graph = ox.distance.add_edge_lengths(graph)
    graph = ox.add_edge_speeds(graph)  # adds attribute "maxspeed"
    graph = ox.add_edge_travel_times(graph)  # adds attribute "travel_time"
    graph = extract_attributes(
        graph,
        edge_attributes={
            "geometry",
            "osmid",
            "length",
            "highway",
            "speed_kph",
            "travel_time",
        },
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
    for element1, element2 in zip(list1, list2):
        if element1.keys() != element2.keys():
            return False
        for key in element1.keys():
            if all(isinstance(x, Graph) for x in [element1[key], element2[key]]):
                # Check if Graphs are isomorphic, as the attributes might differ
                if not is_isomorphic(element1[key], element2[key]):
                    return False
            elif element1[key] != element2[key]:
                return False
    return True


def has_pairwise_overlap(lists):
    """Return a boolean array indicating overlap between pairs of lists.

    Uses numpy arrays and vector broadcasting to speed up the calculation.
    For short lists using set operations is faster.

    Parameters
    ----------
    lists : list of lists
        The lists to check for pairwise overlap. Lists can be of different length.

    Raises
    ------
    ValueError
        If lists is not a list of lists.
    ValueError
        If lists is empty.

    Returns
    -------
    has_overlap : ndarray
        A boolean array indicating whether there is overlap between each pair of
        lists. has_overlap[i, j] is True if there is overlap between list i and
        list j, and False otherwise.

    """
    if not isinstance(lists, list) or not all(isinstance(lst, list) for lst in lists):
        raise ValueError("The input must be a list of lists.")
    if not lists:
        raise ValueError("The input must not be the empty list.")

    # Convert lists to sets
    sets = [set(lst) for lst in lists]

    # Compute the pairwise intersection of the sets
    intersections = zeros((len(sets), len(sets)))
    for i, _ in enumerate(sets):
        for j, _ in enumerate(sets):
            intersections[i, j] = len(sets[i] & sets[j])
            intersections[j, i] = intersections[i, j]

    # Compute the pairwise union of the sets
    unions = array([len(s) for s in sets]).reshape(-1, 1) + len(lists) - 1

    # Compute whether there is overlap between each pair of sets
    has_overlap = intersections > 0
    overlaps = intersections / unions
    fill_diagonal(overlaps, 0)
    has_overlap |= overlaps > 0

    return has_overlap


def compare_dicts(dict1, dict2):
    """Compare two dictionaries recursively.

    Function to recursively compare nested dicts that might contain numpy arrays.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to compare.
    dict2 : dict
        The second dictionary to compare.

    Returns
    -------
    bool
        True if the dictionaries are equal, False otherwise.
    """

    if type(dict1).__name__ != type(dict2).__name__:
        return False

    if isinstance(dict1, dict):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1.keys():
            if not compare_dicts(dict1[key], dict2[key]):
                return False
        return True

    if isinstance(dict1, ndarray):
        return array_equal(dict1, dict2)

    return dict1 == dict2
