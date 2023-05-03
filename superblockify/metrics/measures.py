"""Measures for the metrics module."""
from datetime import timedelta
from time import time

import numpy as np
from numpy import mean, sum as npsum, fill_diagonal, logical_and, isfinite, zeros_like


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
    .. math:: D_{E/S}=\left\langle\frac{d_1(i, j)}{d_2(i, j)}\right\rangle_{i\neq j}
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
    mask = logical_and(mask, isfinite(dist1))
    mask = logical_and(mask, isfinite(dist2))
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

    Returns
    -------
    float
        The coverage of the partitioner, between 0 and 1.

    Raises
    ------
    ValueError
        If the partitioner has an empty graph.
    """

    # if there are no edges in the graph, raise an error
    if partitioner.graph.number_of_edges() == 0:
        raise ValueError("The graph is empty.")

    # if there are no edges in the sparsified graph, return 1
    if partitioner.sparsified.number_of_edges() == 0:
        return 1
    if partitioner.graph.number_of_edges() == partitioner.sparsified.number_of_edges():
        return 0

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


def betweenness_centrality(dist_matrix, predecessors, node_list):
    """Calculate the betweenness centrality of the nodes and edges.

    Uses the predecessors to calculate the betweenness centrality of the nodes and
    edges. [1]_ [2]_ [3]_

    Parameters
    ----------
    dist_matrix : np.ndarray
        The distance matrix for the network measures, as returned by
        :func:`superblockify.metrics.distances.calculate_path_distance_matrix`
    predecessors : np.ndarray
        Predecessors matrix of the graph, as returned by
        :func:`superblockify.metrics.distances.calculate_path_distance_matrix`
    node_list : list
        Indicating the order of the nodes in the distance matrix

    Returns
    -------
    np.ndarray, np.ndarray
        The betweenness centrality of the nodes, ordered according to node_list, and
        the betweenness centrality of the edges, ordered according to the edges of the
        graph.

    Notes
    -----
    Does not include endpoints.

    References
    ----------
    .. [1] Linton C. Freeman: A Set of Measures of Centrality Based on Betweenness.
       Sociometry, Vol. 40, No. 1 (Mar., 1977), pp. 35-41
       https://doi.org/10.2307/3033543
    .. [2] Brandes, U. (2001). A faster algorithm for betweenness centrality. Journal of
       Mathematical Sociology, 25(2), 163–177.
       https://doi.org/10.1080/0022250X.2001.9990249
    .. [3] Brandes, U. (2008). On variants of shortest-path betweenness centrality and
       their generic computation. Social Networks, 30(2), 136–145.
       https://doi.org/10.1016/j.socnet.2007.11.001
    """

    # Iterate through predecessors matrix and count the times
    # - each node is used intermediately
    node_freq = zeros_like(node_list, dtype=np.int64)
    # - each edge is used intermediately (sparse matrix)
    edge_freq = zeros_like(dist_matrix, dtype=np.int64)

    # Zip the above loop
    time0 = time()
    for start, end in zip(*np.where(predecessors != -9999)):
        if start == end:
            continue
        prev = predecessors[start]
        curr = prev[end]
        while curr != start:
            node_freq[curr] += 1
            edge_freq[prev[curr], curr] += 1
            curr = prev[curr]
    print(f"Time for zip: {timedelta(seconds=time() - time0)}")

    # Norm by dividing by total amount of shortest paths
    # - using where predectessors != -9999
    total_shortest_paths = npsum(np.where(predecessors != -9999, 1, 0))

    return node_freq / total_shortest_paths, edge_freq / total_shortest_paths
