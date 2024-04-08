"""Speed module for superblockify, used to add speed limits to the edges of a graph."""

from ..config import Config


def add_edge_travel_times_restricted(
    graph,
    sparsified,
    v_s=Config.V_MAX_SPARSE,
    v_ltn=Config.V_MAX_LTN,
):
    r"""Add edge travel times (in seconds) to a graph.

    The max speed :math:`v_{\mathrm{s}}` is used for the edges of the sparsified graph,
    and the max speed :math:`v_{\mathrm{ltn}}` is used for the remaining edges.

    .. math::

        v_{ij} = \begin{cases}
            v_{\mathrm{s}} & \text{if } (i, j) \in E_{\mathrm{sparsified}} \\
            v_{\mathrm{ltn}} & \text{otherwise}
        \end{cases}

    Then the travel time :math:`t_{ij} = \frac{l_{ij}}{v_{ij}}` is calculated for each
    edge where :math:`l_{ij}` is the length of the edge.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to add the travel times to.
    sparsified : networkx.Graph
        The sparsified graph, optimally a view of the original graph.
    v_s : float
        The max speed for the edges of the sparsified graph.
    v_ltn : float
        The max speed for the remaining edges.

    Notes
    -----
    The travel times are added as an edge attribute ``travel_time_restricted``.

    This function modifies the graph in-place, no value is returned.

    :math:`v_{\mathrm{s}}` and :math:`v_{\mathrm{ltn}}` are read from the
    :mod:`superblockify.config` module and are handled as km/h.
    """

    # units: `length` in m, `v_s` and `v_ltn` in km/h, `travel_time_restricted` in s
    for edge in graph.edges:
        if edge in sparsified.edges:
            graph.edges[edge]["travel_time_restricted"] = (
                graph.edges[edge]["length"] / float(v_s)
            ) * 3.6
        else:
            graph.edges[edge]["travel_time_restricted"] = (
                graph.edges[edge]["length"] / float(v_ltn)
            ) * 3.6
