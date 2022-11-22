"""Plotting functions."""


def paint_streets(graph, edge_linewidth=1, node_alpha=0, edge_color=None, **pg_kwargs):
    """Plot a graph with cyclic colormap related to street direction.

    Color will be chosen based on edge bearing, cyclic in 90 degree.
    Function is a wrapper around `osmnx.plot_graph`.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph
    edge_linewidth : float
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
