"""Spatial graph order measures for the superblockify package."""

from numpy import log
from osmnx import orientation_entropy
from osmnx.projection import is_projected, project_graph
from osmnx.stats import basic_stats
from scipy.stats import entropy

from .population.approximation import get_population_area
from .config import Config


def basic_graph_stats(graph, area=None):
    """Calculate several basic graph stats.

    This function calculates the stats from :func:`osmnx.stats.basic_stats` and adds
    the street count per node and the street orientation order
    (:func:`street_orientation_order`).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to calculate the stats of.
    area : float or None
        The area of the graph in square meters. This is used to calculate density
        measures.
    """
    stats = basic_stats(graph, area=area)
    stats["street_orientation_order"] = street_orientation_order(graph, Config.NUM_BINS)
    return stats


def street_orientation_order(graph, num_bins):
    r"""Calculate the street orientation order of a graph.

    .. math::
        \phi=1-\left(\frac{H_o-H_g}{H_{\max}-H_g}\right)^2

    Where :math:`H_o` is the orientation entropy of the graph, :math:`H_g` is the
    minimal plausible entropy of a grid like city :math:`H_g \approx 1.386`, and
    :math:`H_{\max}` is the maximal entropy of a city where the streets are
    uniformly distributed :math:`H_{\max} = \log(\text{num_bins=36}) \approx 3.584`.
    To make an order parameter of this from total disorder to order, :math:`\phi` is
    designed to reach from 0 to 1. [1]_ [2]_

    Calculations are done on the undirected graph, and if the graph is projected, it
    is first unprojected to WGS84.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to calculate the street orientation order of.
    num_bins : int
        The number of bins to use for the orientation histogram in the entropy
        calculation. Positive integer.

    Returns
    -------
    float
        The street orientation order of the graph as defined by [1]_.

    Raises
    ------
    ValueError
        If `num_bins` is not a positive integer.

    References
    ----------
    .. [1] Boeing, G. Urban spatial order: street network orientation, configuration,
           and entropy. Appl Netw Sci **4**, 67 (2019).
           https://doi.org/10.1007/s41109-019-0189-1
    .. [2] Boeing, G. Street Network Models and Indicators for Every Urban Area in the
           World. Geographical Analysis 54, 519–535 (2022).
           https://doi.org/10.1111/gean.12281
    """
    if not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError(
            f"num_bins must be a non-negative integer, not {num_bins} of type "
            f"{type(num_bins)}."
        )

    graph_unprojected = graph.copy()
    if is_projected(graph_unprojected.graph["crs"]):
        # logger.debug(
        #     "Orientation order: Unprojecting graph from %s to 2D coordinates ("
        #     "epsg:4326).",
        #     graph_unprojected.graph["crs"],
        # )
        graph_unprojected = project_graph(graph_unprojected, to_crs="epsg:4326")
    # orientation_entropy requires an undirected graph
    graph_unprojected = graph_unprojected.to_undirected()

    min_entropy_bins = 4  # perfect grid
    perfect_grid = [1] * min_entropy_bins + [0] * (num_bins - min_entropy_bins)
    perfect_grid_entropy = entropy(perfect_grid)
    max_entropy = log(num_bins)

    o_entropy = orientation_entropy(graph_unprojected, num_bins=num_bins)

    return (
        1
        - ((o_entropy - perfect_grid_entropy) / (max_entropy - perfect_grid_entropy))
        ** 2
    )


def calculate_component_metrics(components):
    """Calculate metrics for the components.

    Calculates the metrics for the components and writes them to each component
    dictionary.



    Parameters
    ----------
    components : list of dicts
        List of dictionaries containing the components.

    Notes
    -----
    Works in-place on the components if they are defined, otherwise on the
    partitions.
    """
    for part in components:
        part["population"], part["area"] = get_population_area(part["subgraph"])
        part["population_density"] = part["population"] / part["area"]

        # Add basic_stats to the component
        part.update(
            basic_graph_stats(part["subgraph"], area=part["area"]),
            population=part["population"],
            population_density=part["population_density"],
        )
