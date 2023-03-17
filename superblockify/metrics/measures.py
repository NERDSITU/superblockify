"""Measures for the metrics module."""
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

    subgraph_edges = [
        part["subgraph"].edges(data=True) for part in partitioner.get_partition_nodes()
    ]

    return npsum(d[weight] for u, v, d in subgraph_edges) / npsum(
        d[weight] for u, v, d in partitioner.graph.edges(data=True)
    )
