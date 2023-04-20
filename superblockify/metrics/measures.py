"""Measures for the metrics module."""
import numpy as np
from numpy import mean, sum as npsum, fill_diagonal, logical_and, isfinite


def calculate_directness(distance_matrix, measure1, measure2):
    r"""Calculate the directness for the given network measures.

    The directness in the mean of the ratios between the distances of the two
    network measures.

    If any of the distances is 0 or infinite, it is ignored in the calculation.

    Parameters
    ----------
    distance_matrix : dict
        The distance matrix for the network measures, as returned by instance attribute
        :attr:`superblockify.metrics.metric.Metric.distance_matrix`
    measure1 : str
        The first network measure
    measure2 : str
        The second network measure

    Returns
    -------
    float
        The directness of the network measures

    Notes
    -----
    .. math:: D_{E/S}=\left\langle\frac{d_E(i, j)}{d_S(i, j)}\right\rangle_{i\neq j}
    """

    dist1, dist2 = _network_measures_filtered_flattened(
        distance_matrix, measure1, measure2
    )

    # Calculate the directness as the mean of the ratios
    return mean(dist1 / dist2)


def calculate_global_efficiency(distance_matrix, measure1, measure2):
    r"""Calculate the global efficiency for the given network measures.

    The global efficiency is the ratio between the sums of the inverses of the
    distances of the two network measures.

    If any of the distances is 0 or infinite, it is ignored in the calculation.

    Parameters
    ----------
    distance_matrix : dict
        The distance matrix for the network measures, as returned by instance attribute
        :attr:`superblockify.metrics.metric.Metric.distance_matrix`
    measure1 : str
        The first network measure
    measure2 : str
        The second network measure

    Returns
    -------
    float
        The global efficiency of the network measures

    Notes
    -----
    .. math::

         E_{\text{glob},S/E}=\frac{\sum_{i \neq j}\frac{1}{d_S(i, j)}}
         {\sum_{i \neq j} \frac{1}{d_E(i, j)}}
    """

    dist1, dist2 = _network_measures_filtered_flattened(
        distance_matrix, measure1, measure2
    )

    # Calculate the global efficiency as the ratio between the sums of the inverses
    return npsum(1 / dist1) / npsum(1 / dist2)


def _network_measures_filtered_flattened(distance_matrix, measure1, measure2):
    """Return the two network measures filtered and flattened.

    The diagonal is set to 0 and the matrix is flattened. We use flattening as it
    preserves the order of the distances and makes the calculation faster.

    Parameters
    ----------
    distance_matrix : dict
        The distance matrix for the network measures, as returned by instance attribute
        :attr:`superblockify.metrics.metric.Metric.distance_matrix`
    measure1 : str
        The first network measure
    measure2 : str
        The second network measure

    Returns
    -------
    1d ndarray
        The first network measure
    1d ndarray
        The second network measure
    """

    # Get the distance matrix for the two network measures
    dist1 = distance_matrix[measure1]
    dist2 = distance_matrix[measure2]
    # Set the diagonal to 0 so that it is not included in the calculation
    fill_diagonal(dist1, 0)
    fill_diagonal(dist2, 0)
    # Flatten the distance matrices
    dist1 = dist1.flatten()
    dist2 = dist2.flatten()

    # Drop the pairs of distances where at least one is 0 or infinite
    mask = logical_and(dist1 != 0, dist2 != 0)
    mask = logical_and(mask, isfinite(dist1), isfinite(dist2))
    dist1 = dist1[mask]
    dist2 = dist2[mask]

    return dist1, dist2


def calculate_coverage(partitioner, weight):
    """Calculate the coverage of the partitioner.

    Calculates the coverage of the partitions weighted by the edge attribute
    self.weight. The coverage is the sum of the weights of the edges between


    Parameters
    ----------
    partitioner : Partitioner
        The partitioner to calculate the coverage for
    weight : str
        The edge attribute to use as weight.
    """

    return 1 - npsum(
        d[weight] for u, v, d in partitioner.sparsified.edges(data=True)
    ) / npsum(d[weight] for u, v, d in partitioner.graph.edges(data=True))


def rel_increase(value_i, value_j):
    """Calculate the relative increase of matrix value_i and value_j.

    Ignore np.inf values and 0 values in the denominator.
    """
    # Use double precision to avoid overflow
    value_i = value_i.astype(np.double)
    value_j = value_j.astype(np.double)
    return np.where(
        (value_i == np.inf) | (value_j == np.inf) | (value_j == 0) | (value_i == 0),
        np.inf,
        value_i / value_j,
    )


def write_relative_increase_to_edges(
    graph, distance_matrix, node_list, measure1, measure2
):
    """Write the relative increase of the distance matrix to the edges of the graph.

    For each edge the relative increases to and from every node to the two nodes
    of the edge are averaged and written to the edge attribute "rel_increase".

    Parameters
    ----------
    graph : nx.Graph
        The graph to write the relative increase to
    distance_matrix : dict
        The distance matrix for the network measures, as returned by instance attribute
        :attr:`superblockify.metrics.metric.Metric.distance_matrix`
    node_list : list
        Indicating the order of the nodes in the distance matrix
    measure1 : str
        The first network measure
    measure2 : str
        The second network measure
    """

    rel_inc = rel_increase(distance_matrix[measure1], distance_matrix[measure2])

    for node1, node2, key in graph.edges(keys=True):
        # All distances to and from u and v
        rel_inc_uv = np.concatenate(
            (
                rel_inc[node_list.index(node1), :],
                rel_inc[:, node_list.index(node1)],
                rel_inc[node_list.index(node2), :],
                rel_inc[:, node_list.index(node2)],
            )
        )
        # Remove np.inf values and average the remaining values
        graph.edges[node1, node2, key]["rel_increase"] = np.mean(
            rel_inc_uv[rel_inc_uv != np.inf]
        )
