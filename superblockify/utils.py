"""Utility functions for superblockify."""

from ast import literal_eval
from itertools import chain
from os.path import getsize
from re import match

import osmnx as ox
from networkx import Graph, is_isomorphic, set_node_attributes
from numba import njit, int64, int32, prange
from numpy import (
    zeros,
    array,
    fill_diagonal,
    ndarray,
    array_equal,
    empty,
    int64 as np_int64,
    sign,
    inf,
    isinf,
    nan,
)
from osmnx._errors import CacheOnlyInterruptError
from osmnx.stats import count_streets_per_node
from shapely import wkt

from .partitioning.utils import reduce_graph
from .config import logger, Config
from .graph_stats import basic_graph_stats
from .population.approximation import add_edge_population


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


def load_graph_from_place(
    save_as,
    search_string,
    add_population=False,
    only_cache=False,
    max_nodes=Config.MAX_NODES,
    **gfp_kwargs,
):
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
    add_population : bool, optional
        Whether to add population data to the graph. Default is False.
    only_cache : bool, optional
        Whether to only load the graph from cache. Default is False.
    max_nodes : int, optional
        Maximum number of nodes in the graph. If the graph has more nodes, it will
        be reduced to `max_nodes`, by taking the ego graph of a representative,
        central node. If None, the graph will not be reduced. Default is set in
        :mod:`superblockify.config`.
    **gfp_kwargs
        Keyword arguments to pass to osmnx.graph_from_place.

    Returns
    -------
    networkx.MultiDiGraph or None
        The graph loaded from the place. If only_cache is True, returns None.
    """

    # check the format of the search string, match to regex R\d+, str or list of str
    # use re.match(r"R\d+", search_string) or to all elements in list
    mult_polygon = ox.geocode_to_gdf(
        search_string,
        by_osmid=isinstance(search_string, str)
        and match(r"R\d+", search_string)
        or isinstance(search_string, list)
        and all(isinstance(s, str) and match(r"R\d+", s) for s in search_string),
    )

    # geopandas.GeoDataFrame, every column is polygon
    # make shapely.geometry.MultiPolygon from all polygons
    # Project to WGS84 to query OSM
    mult_polygon = ox.projection.project_gdf(mult_polygon, to_crs="epsg:4326")
    logger.debug("Reprojected boundary to WGS84 - Downloading graph")
    # Get graph - automatically adds distances before simplifying
    ox.settings.cache_only_mode = only_cache
    try:
        graph = ox.graph_from_polygon(mult_polygon.geometry.union_all(), **gfp_kwargs)
    except CacheOnlyInterruptError:  # pragma: no cover
        logger.debug("Only loaded graph from cache")
        return None
    logger.debug("Downloaded graph - Preprocessing")
    # Add edge bearings
    graph = ox.add_edge_bearings(graph)  # precision=2)  # the precision >1 is
    # important for binning when using the deprecated BearingPartitioner

    # Project to local UTM - coordinates can be used as
    graph = ox.project_graph(graph)

    graph = ox.add_edge_speeds(graph)  # adds attribute "maxspeed"
    graph = ox.add_edge_travel_times(graph)  # adds attribute "travel_time"
    # count the number of streets per node / degree
    street_count = count_streets_per_node(graph)
    set_node_attributes(graph, values=street_count, name="street_count")
    graph = extract_attributes(
        graph,
        edge_attributes={
            "geometry",
            "osmid",
            "length",
            "highway",
            "speed_kph",
            "travel_time",
            "bearing",
        },
        node_attributes={"y", "x", "lat", "lon", "osmid", "street_count"},
    )
    # Add edge population and area
    if add_population:
        add_edge_population(graph)

    # Add boundary as union of all polygons as attribute - in UTM crs of centroid
    mult_polygon = ox.projection.project_gdf(mult_polygon)
    graph.graph["boundary_crs"] = mult_polygon.crs
    graph.graph["boundary"] = mult_polygon.geometry.union_all()
    graph.graph["area"] = graph.graph["boundary"].area
    # update graph attributes with basic graph stats
    graph.graph.update(basic_graph_stats(graph, area=graph.graph["area"]))
    # Save graph
    if max_nodes is not None and graph.number_of_nodes() > max_nodes:
        logger.debug("Reducing graph to %s nodes", max_nodes)
        graph = reduce_graph(graph, max_nodes=max_nodes)
    logger.debug("Preprocessing done - Saving graph")
    ox.save_graphml(graph, filepath=save_as)
    logger.debug("Saved graph (%s MB) to %s", getsize(save_as) >> 20, save_as)
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


@njit(int64(int32, int32, int64))
def __edge_to_1d(edge_u, edge_v, max_len):  # pragma: no cover
    """Convert edge to 1D representation.

    Parameters
    ----------
    edge_u : int
        First node index
    edge_v : int
        Second node index
    max_len : int
        Maximum length of the node indices

    Returns
    -------
    int
        1D representation of the edge
    """
    return edge_u * 10**max_len + edge_v


@njit(int64[:](int32[:], int32[:], int64), parallel=True)
def __edges_to_1d(edge_u, edge_v, max_len):  # pragma: no cover
    """Convert edges to 1D representation.

    Parameters
    ----------
    edge_u : np.ndarray
        First node indices
    edge_v : np.ndarray
        Second node indices
    max_len : int
        Maximum length of the node indices

    Returns
    -------
    ndarray
        1D representation of the edges
    """
    edges = empty(len(edge_u), dtype=np_int64)
    for i in prange(len(edge_u)):  # pylint: disable=not-an-iterable
        edges[i] = __edge_to_1d(edge_u[i], edge_v[i], max_len)
    return edges


def load_graphml_dtypes(filepath=None, attribute_label=None, attribute_dtype=None):
    """Load a graphml file with custom dtypes.

    Parameters
    ----------
    filepath : str
        Path to the graphml file.
    attribute_label : str, optional
        The attribute label to convert to the specified dtype.
    attribute_dtype : type, optional
        The dtype to convert the attribute to.

    Returns
    -------
    networkx.MultiDiGraph
        The graph.
    """

    node_dtypes = {
        "y": float,
        "x": float,
    }
    edge_dtypes = {
        "bearing": float,
        "length": float,
        "speed_kph": float,
        "travel_time": float,
        "travel_time_restricted": float,
        "rel_increase": float,
        "population": float,
        "area": float,
        "cell_id": int,
        "edge_betweenness_normal": float,
        "edge_betweenness_length": float,
        "edge_betweenness_linear": float,
        "edge_betweenness_normal_restricted": float,
        "edge_betweenness_length_restricted": float,
        "edge_betweenness_linear_restricted": float,
    }
    graph_dtypes = {
        "simplified": bool,
        "edge_population": bool,
        "boundary": wkt.loads,
        "area": float,
        "n": int,
        "m": int,
        "k_avg": float,
        "edge_length_total": float,
        "edge_length_avg": float,
        "streets_per_node_avg": float,
        "streets_per_node_counts": literal_eval,
        "streets_per_node_proportions": literal_eval,
        "intersection_count": int,
        "street_length_total": float,
        "street_segment_count": int,
        "street_length_avg": float,
        "circuity_avg": float,
        "self_loop_proportion": float,
        "node_density_km": float,
        "intersection_density_km": float,
        "edge_density_km": float,
        "street_density_km": float,
        "street_orientation_order": float,
    }
    # Add the same graph_dtypes, but with the `reduced_` prefix
    graph_dtypes.update(
        {f"reduced_{key}": value for key, value in graph_dtypes.items()}
    )

    if attribute_label is not None and attribute_dtype is not None:
        edge_dtypes[attribute_label] = attribute_dtype
    graph = ox.load_graphml(
        filepath=filepath,
        node_dtypes=node_dtypes,
        edge_dtypes=edge_dtypes,
        graph_dtypes=graph_dtypes,
    )
    return graph


def percentual_increase(val_1, val_2):
    """Compute the percentual increase between two values.

    3 -> 4 = 33.33% = 1/3
    4 -> 3 = -25.00% = -1/4

    Parameters
    ----------
    val_1 : float
        The first value.
    val_2 : float
        The second value.

    Returns
    -------
    float
        The relative difference between the two values.

    Notes
    -----
    If both values are zero, the result is zero.
    If one value is zero, the result is +-infinity.
    """
    if val_1 == val_2:
        return 0
    if val_1 == 0 or val_2 == 0:
        return inf * (sign(val_2 - val_1) if val_2 != 0 else -1)
    if isinf(val_1) and isinf(val_2):
        return nan
    if isinf(val_1):
        return -inf
    if isinf(val_2):
        return inf * sign(val_1) * (sign(val_2) if val_2 != 0 else 1)
    return (val_2 / val_1) - 1
