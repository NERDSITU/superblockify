"""Plotting functions."""
import logging

import networkx as nx
import osmnx as ox
from matplotlib import pyplot as plt
from numpy import amin, amax

from superblockify import attribute

logger = logging.getLogger("superblockify")


def paint_streets(graph, cmap="hsv", **pg_kwargs):
    """Plot a graph with (cyclic) colormap related to edge direction.

    Color will be chosen based on edge bearing, cyclic in 90 degree.
    Function is a wrapper around `osmnx.plot_graph`.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    cmap : string, optional
        name of a matplotlib colormap
    pg_kwargs
        keyword arguments to pass to `osmnx.plot_graph`.

    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis

    Examples
    --------
    _See example in `scripts/TestingNotebooks/20221122-painting_grids.py`._

    """

    # Calculate bearings if no edge has `bearing` attribute.
    if not bool(nx.get_edge_attributes(graph, "bearing")):
        graph = ox.add_edge_bearings(graph)

    # Write attribute where bearings are baked down modulo 90 degrees.
    attribute.new_edge_attribute_by_function(
        graph, lambda bear: bear % 90, "bearing", "bearing_90"
    )

    return plot_by_attribute(graph, "bearing_90", cmap, **pg_kwargs)


def plot_by_attribute(
    graph,
    attr,
    cmap="hsv",
    edge_linewidth=1,
    node_alpha=0,
    minmax_val=None,
    **pg_kwargs,
):  # noqa: too-many-arguments
    """Plot a graph based on an edge attribute and colormap.

    Color will be chosen based on the specified edge attribute passed to a colormap.
    Function is a direct wrapper around `osmnx.plot_graph`.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Input graph
    attr : string
        Graph's attribute to select colors by
    cmap : string, optional
        Name of a matplotlib colormap
    edge_linewidth : float, optional
        Width of the edges' lines
    node_alpha : float, optional
        Opacity of the nodes
    edge_color : None, optional
        Do not pass this attribute, as it is set by the bearing direction.
    minmax_val : tuple, optional
        Tuple of (min, max) values of the attribute to be plotted
        (default: min and max of attr)
    pg_kwargs
        Keyword arguments to pass to `osmnx.plot_graph`.

    Raises
    ------
    ValueError
        If edge_color was set to anything but None.
    ValueError
        If `edge_linewidth` and `node_size` both <= 0, otherwise the plot will be empty.
    ValueError
        If `minmax_val` is not a tuple of length 2 or None.
    ValueError
        If `minmax_val[0]` is not smaller than `minmax_val[1]`.

    Returns
    -------
    fig, ax : tuple
        matplotlib figure, axis

    """

    if ("edge_color" in pg_kwargs) and (pg_kwargs["edge_color"] is not None):
        raise ValueError(
            f"The `edge_color` attribute was set to {pg_kwargs['edge_color']}, "
            f"it will be overwritten by the colors determined with the "
            f"bearings and colormap."
        )

    if minmax_val is not None and (
        not isinstance(minmax_val, tuple) or len(minmax_val) != 2
    ):
        raise ValueError(
            f"The `minmax_val` attribute was set to {minmax_val}, "
            f"it should be a tuple of length 2 or None."
        )

    # Make list of edge colors, order is the same as in graph.edges()
    # Determine min and max values of the attribute
    logger.debug("Given minmax_val for attribute %s: %s", attr, minmax_val)
    if minmax_val is None or minmax_val[0] is None or minmax_val[1] is None:
        # Min and max of the attribute, ignoring `None` values
        minmax = (
            amin([v for v in nx.get_edge_attributes(graph, attr).values() if v]),
            amax([v for v in nx.get_edge_attributes(graph, attr).values() if v]),
        )
        if minmax_val is None:
            minmax_val = minmax
        elif minmax_val[0] is None:
            minmax_val = (minmax[0], minmax_val[1])
        else:
            minmax_val = (minmax_val[0], minmax[1])
        logger.debug("Determined minmax_val for attribute %s: %s", attr, minmax_val)

    if minmax_val[0] >= minmax_val[1]:
        raise ValueError(
            f"The `minmax_val` attribute is {minmax_val}, "
            f"but the first value must be smaller than the second."
        )

    # Choose the color for each edge based on the edge's attribute value,
    # if `None`, set to gray.
    colormap = plt.get_cmap(cmap)
    e_c = [
        colormap((attr_val - minmax_val[0]) / (minmax_val[1] - minmax_val[0]))
        if attr_val is not None
        else (0.5, 0.5, 0.5, 1) # gray
        for u, v, k, attr_val in graph.edges(keys=True, data=attr)
    ]

    # Print list of unique colors in the colormap, with a set comprehension
    logger.debug(
        "Unique colors in the colormap %s: %s",
        cmap,
        {tuple(c) for c in e_c},
    )

    # Plot graph with osmnx's function, pass further attributes
    return ox.plot_graph(
        graph,
        node_alpha=node_alpha,
        edge_color=e_c,
        edge_linewidth=edge_linewidth,
        **pg_kwargs,
    )
