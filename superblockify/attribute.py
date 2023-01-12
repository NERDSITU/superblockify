"""Work with graph attributes."""
from networkx import get_edge_attributes, set_edge_attributes


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
    """Return subgraph of edges with a given attribute value.

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
