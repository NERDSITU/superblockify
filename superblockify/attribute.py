"""Work with graph attributes."""

from networkx import get_edge_attributes, set_edge_attributes, get_node_attributes
from numpy import amin, amax

from .config import logger


def new_edge_attribute_by_function(
    graph, function, source_attribute, destination_attribute, allow_overwriting=False
):
    """Maps new edge attributes from an existing attribute to a new one.

    Works on the passed graph and writes new attribute as a result of the
    function to it.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    function : function
        Function that takes the source attribute
    source_attribute : string
        Name of an existing node attribute
    destination_attribute : string
        Name of a node attribute which is to be written
    allow_overwriting : bool, optional
        Set to `True` if graph attributes are allowed to be overwritten,
        as in the ase where source and destination attribute have the same name.

    Raises
    ------
    ValueError
        If the graph has already attributes with the destination key,
        and overfriting is `False`.
    ValueError
        If the source attribute does not exist in the graph.

    Examples
    --------
    >>> G = nx.path_graph(3)
    >>> nx.set_edge_attributes(G, 4, 'four')
    >>> new_edge_attribute_by_function(G, lambda x: 2*x, 'four', 'eight')
    >>> G.edges.data('eight')
    EdgeDataView([(0, 1, 8), (1, 2, 8)])

    """

    attributes = get_edge_attributes(graph, source_attribute)

    if not bool(attributes):
        raise ValueError(
            f"Graph with {len(graph)} node(s) has no attributes for "
            f"the key '{source_attribute}'."
        )

    if (source_attribute == destination_attribute) and not allow_overwriting:
        raise ValueError(
            f"Cannot overwrite the attribute '{source_attribute}' if "
            f"`allow_overwriting` is set to `False`."
        )

    if bool(get_edge_attributes(graph, destination_attribute)) and (
        not allow_overwriting
    ):
        raise ValueError(
            f"Destination attribute '{destination_attribute}' has values, set "
            f"`allow_overwriting` to `True` if you want to overwrite these."
        )

    for edge, attr in attributes.items():
        attributes[edge] = function(attr)

    set_edge_attributes(graph, attributes, destination_attribute)


def get_edge_subgraph_with_attribute_value(graph, attribute_label, attribute_value):
    """Return subgraph view of edges with a given attribute value.
    The graph, edge, and node attributes in the returned subgraph view are references to
    the corresponding attributes in the original graph. The view is read-only.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    attribute_label : string
        Name of an existing node attribute
    attribute_value : int
        Value of the attribute to be selected

    Returns
    -------
    networkx.Graph
        Subgraph of edges with the given attribute value

    Raises
    ------
    ValueError
        If the graph has no attributes with the given key.
    ValueError
        If the returned subgraph is empty.

    Examples
    --------
    >>> G = nx.path_graph(6)
    >>> nx.set_edge_attributes(G, {edge: {'attr': int(edge[0]/2)} for edge in G.edges})
    >>> get_edges_with_attribute_value(G, 'attr', 1).edges
    EdgeView([(2, 3), (3, 4)])

    """

    attributes = get_edge_attributes(graph, attribute_label)

    if not bool(attributes):
        raise ValueError(
            f"Graph with {len(graph)} node(s) has no attributes for "
            f"the key '{attribute_label}'."
        )

    edges = [edge for edge, attr in attributes.items() if attr == attribute_value]

    if not bool(edges):
        raise ValueError(
            f"Graph with {len(graph)} node(s) has no edges with the "
            f"attribute '{attribute_label}' set to '{attribute_value}'."
        )

    return graph.edge_subgraph(edges)


def determine_minmax_val(graph, minmax_val, attr, attr_type="edge"):
    """Determine the min and max values of an attribute in a graph.

    This function is used to determine the min and max values of an attribute.
    If `minmax_val` is None, the min and max values of the attribute in the graph
    are used. If `minmax_val` is a tuple of length 2, the values are used as
    min and max values. If `minmax_val` is a tuple of length 2, but the first
    value is larger than the second, a ValueError is raised.
    If only one value in the tuple is given, the other value is set accordingly.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    minmax_val : tuple, None
        Tuple of (min, max) values of the attribute to be plotted or None
    attr : string
        Graph's attribute to select min and max values by
    attr_type : string, optional
        Type of the attribute, either "edge" or "node"

    Returns
    -------
    tuple
        Tuple of (min, max) values.

    Raises
    ------
    ValueError
        If `minmax_val` is not a tuple of length 2 or None.
    ValueError
        If `minmax_val[0]` is not smaller than `minmax_val[1]`.
    ValueError
        If `attr_type` is not "edge" or "node".

    """

    if minmax_val is not None and (
        not isinstance(minmax_val, tuple) or len(minmax_val) != 2
    ):
        raise ValueError(
            f"The `minmax_val` attribute was set to {minmax_val}, "
            f"it should be a tuple of length 2 or None."
        )

    if attr_type not in ["edge", "node"]:
        raise ValueError(
            f"The `attr_type` attribute was set to {attr_type}, "
            f"it should be either 'edge' or 'node'."
        )

    # Determine min and max values of the edge attribute
    logger.debug("Given minmax_val for edge attribute %s: %s", attr, minmax_val)
    if minmax_val is None or minmax_val[0] is None or minmax_val[1] is None:
        # Min and max of the edge attribute, ignoring `None` values
        if attr_type == "edge":
            minmax = (
                amin(
                    [
                        v
                        for v in get_edge_attributes(graph, attr).values()
                        if v is not None
                    ]
                ),
                amax(
                    [
                        v
                        for v in get_edge_attributes(graph, attr).values()
                        if v is not None
                    ]
                ),
            )
        else:
            minmax = (
                amin(
                    [
                        v
                        for v in get_node_attributes(graph, attr).values()
                        if v is not None
                    ]
                ),
                amax(
                    [
                        v
                        for v in get_node_attributes(graph, attr).values()
                        if v is not None
                    ]
                ),
            )
        if minmax_val is None or minmax_val == (None, None):
            minmax_val = minmax
        elif minmax_val[0] is None:
            minmax_val = (minmax[0], minmax_val[1])
        else:
            minmax_val = (minmax_val[0], minmax[1])
        logger.debug(
            "Determined minmax_val for edge attribute %s: %s", attr, minmax_val
        )
    if minmax_val[0] >= minmax_val[1]:
        raise ValueError(
            f"The `minmax_val` attribute is {minmax_val}, "
            f"but the first value must be smaller than the second."
        )
    return minmax_val


def aggregate_edge_attr(graph, key, func, dismiss_none=True):
    """Aggregate edge attributes by function.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph, subgraph or view
    key : immutable
        Edge attribute key
    func : function
        Function to aggregate values. Able to handle lists.
    dismiss_none : bool, optional
        If True, dismiss None values. Default: True

    Returns
    -------
    dict
        Dictionary of aggregated edge attributes

    Raises
    ------
    KeyError
        If there are no edge attributes for the given key.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=1)
    >>> G.add_edge(2, 3, weight=2)
    >>> G.add_edge(3, 4, weight=3)
    >>> G.add_edge(4, 1, weight=None)
    >>> aggregate_edge_attr(G, 'weight', sum)
    6
    >>> aggregate_edge_attr(G, 'weight', lambda x: sum(x)/len(x))
    2.0
    >>> aggregate_edge_attr(G, 'weight', lambda x: x)
    [1, 2, 3]
    """
    # Get edge attributes
    edge_attr = get_edge_attributes(graph, key)
    # Check if there are any
    if not bool(edge_attr):
        raise KeyError(
            f"Graph with {len(graph)} node(s) has no edge attributes for "
            f"the key '{key}'."
        )
    # Dismiss None values
    if dismiss_none:
        edge_attr = {k: v for k, v in edge_attr.items() if v is not None}
    # Aggregate
    return func(list(edge_attr.values()))
