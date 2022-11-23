"""Plotting functions."""
import networkx as nx
import osmnx as ox

from superblockify import attribute


def paint_streets(
    graph, cmap="hsv", edge_linewidth=1, node_alpha=0, edge_color=None, **pg_kwargs
):
    """Plot a graph with (cyclic) colormap related to street direction.

    Color will be chosen based on edge bearing, cyclic in 90 degree.
    Function is a wrapper around `osmnx.plot_graph`.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    cmap : string, optional
        name of a matplotlib colormap
    edge_linewidth : float, optional
        width of the edges' lines
    node_alpha : float, optional
        Opacity of the nodes
    edge_color : None, optional
        Do not pass this attribute, as it is set by the bearing direction.
    pg_kwargs
        keyword arguments to pass to `osmnx.plot_graph`.

    Raises
    ------
    ValueError
        If edge_color was set to anything but None.
    ValueError
        If `edge_linewidth` and `node_size` both <= 0, otherwise the plot will be empty.
    ValueError
        If graph is DiGraph? TODO

    Examples
    --------
    _See example in `scripts/TestingNotebooks/20221122-painting_grids.py`._

    """

    if edge_color is not None:
        raise ValueError(
            f"The `edge_color` attribute was set to {edge_color}, "
            f"it will be overwritten by the colors determined with the "
            f"bearings and colormap."
        )

    # Calculate bearings if no edge has `bearing` attribute.
    if not bool(nx.get_edge_attributes(graph, "bearing")):
        graph = ox.add_edge_bearings(graph)

    # Write attribute where bearings are baked down modulo 90 degrees.
    attribute.new_edge_attribute_by_function(
        graph, lambda bear: bear % 90, "bearing", "bearing_90"
    )

    # Make series of edge colors, labels are edge IDs (u, v, key) and values are colors
    e_c = ox.plot.get_edge_colors_by_attr(graph, attr="bearing_90", cmap=cmap)

    # `e_c` only contains colors for edges with the bearing(_90) attribute, but needs
    # colors for every edge. Find edges without bearings (like loops) and set their
    # color transparent.
    # Get all edges
    edges_wo_bearings = list(nx.edges(graph))
    for node in nx.get_edge_attributes(graph, "bearing_90").keys():
        edges_wo_bearings.remove(node[:2])  # remove edges where `bearing_90` is set
    for edge in edges_wo_bearings:
        e_c[(*edge, 0)] = (0, 0, 0, 0)  # transparent color for remaining edges

    # Plot graph with osmnx's function, pass further attributes
    return ox.plot_graph(
        graph,
        node_alpha=node_alpha,
        edge_color=e_c,
        edge_linewidth=edge_linewidth,
        **pg_kwargs,
    )
