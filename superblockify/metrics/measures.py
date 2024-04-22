"""Measures for the metrics module."""

from datetime import timedelta
from time import time

import networkx as nx
import numpy as np
from numba import njit, prange, int32, int64, float32, float64
from numpy import sum as npsum

from ..attribute import aggregate_edge_attr
from ..config import logger
from ..utils import __edges_to_1d, __edge_to_1d, percentual_increase


def calculate_directness(distance_matrix, measure1, measure2):
    r"""Calculate the directness for the given network measures.

    The directness is the mean of the ratios between the distances of the two
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
    return np.mean(dist1 / dist2)


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
    np.fill_diagonal(dist1, 0)
    np.fill_diagonal(dist2, 0)
    # Flatten the distance matrices
    dist1 = dist1.flatten()
    dist2 = dist2.flatten()

    # Drop the pairs of distances where at least one is 0 or infinite
    mask = np.logical_and(dist1 != 0, dist2 != 0)
    mask = np.logical_and(mask, np.isfinite(dist1))
    mask = np.logical_and(mask, np.isfinite(dist2))
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
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (value_i == np.inf) | (value_j == np.inf) | (value_j == 0) | (value_i == 0),
            np.inf,
            value_i / value_j,
        )


def write_relative_increase_to_edges(
    graph, distance_matrix, node_list, measure1, measure2
):
    """Write the relative increase of the distance matrix to the edges of the graph.

    For each edge, the relative increases to and from every node to the two nodes
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


@nx.utils.py_random_state(7)
def betweenness_centrality(
    graph,
    node_order,
    dist_matrix,
    predecessors,
    weight="length",
    attr_suffix=None,
    k=None,
    seed=None,
    max_range=None,
):
    """Calculate several types of betweenness centrality for the nodes and edges.

    Uses the predecessors to calculate the betweenness centrality of the nodes and
    edges. The normalized betweenness centrality is calculated, length-scaled, and
    linearly scaled betweenness centrality is calculated for the nodes and edges. When
    passing a k, the summation is only done over k random nodes. [1]_ [2]_ [3]_

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The graph to calculate the betweenness centrality for, distances and
        predecessors must be calculated for this graph
    node_order : list
        Indicating the order of the nodes in the distance matrix
    dist_matrix : np.ndarray
        The distance matrix for the network measures, as returned by
        :func:`superblockify.metrics.distances.calculate_path_distance_matrix`
    predecessors : np.ndarray
        Predecessors matrix of the graph, as returned by
        :func:`superblockify.metrics.distances.calculate_path_distance_matrix`
    weight : str, optional
        The edge attribute to use as weight to decide which multi-edge to attribute the
        betweenness centrality to, by default "length". If None, the first edge of the
        multi-edge is used.
    attr_suffix : str, optional
        The suffix to append to the attribute names, by default None
    k : int, optional
        The number of nodes to calculate the betweenness centrality for, by default
        None
    seed : int, random_state, or None (default)
        Indicator of random number generation state. See :ref:`Randomness<randomness>`
        for additional details.
    max_range : float, optional
        The maximum path length to consider, by default None, which means no maximum
        path length. It is measured in unit of the weight attribute.

    Raises
    ------
    ValueError
        If weight is not None, and the graph does not have the weight attribute on all
        edges.

    Notes
    -----
    Works in-place on the graph.

    It Does not include endpoints.

    Modified from :mod:`networkx.algorithms.centrality.betweenness`.

    The :attr:`weight` attribute is not used to determine the shortest paths, these are
    taken from the predecessor matrix. It is only used for parallel edges to decide
    which edge to attribute the betweenness centrality to.

    If there are :math:`<=` 2 nodes, node betweenness is 0 for all nodes. If there are
    :math:`<=` 1 edges, edge betweenness is 0 for all edges.

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
    """  # pylint: disable=too-many-locals
    # Check if weight attribute is present on all edges
    if weight and not all(
        weight in data for _, _, _, data in graph.edges(keys=True, data=True)
    ):
        raise ValueError(f"Weight attribute {weight} not found on all edges")

    start_time = time()
    if len(graph) == 1:
        b_c = {"node": {"normal": [0], "length": [0], "linear": [0]}, "edge": {}}
    else:
        # edge_list in numba compatible format, originally a list of tuples
        # (idx_u, idx_v) -> string of concatenated indices, padded with zeros (based
        # on the max value) to ensure that all strings have the same length and don't
        # collide
        edges_uv_id = __edges_to_1d(
            np.array(
                [node_order.index(u) for u, _ in graph.edges(keys=False)],
                dtype=np.int32,
            ),
            np.array(
                [node_order.index(v) for _, v in graph.edges(keys=False)],
                dtype=np.int32,
            ),
            len(str(len(graph))),
        )
        # sort inplace to ensure that the edge indices are in ascending order
        edges_uv_id.sort()

        b_c = _calculate_betweenness(
            edges_uv_id,
            predecessors,
            dist_matrix,
            edge_padding=len(str(len(graph))),
            index_subset=seed.sample(range(len(node_order)), k=k) if k else None,
            max_range=max_range,
        )

    attr_suffix = attr_suffix if attr_suffix else ""

    # Normalize betweenness values and write to graph
    scale = (
        1 / ((len(node_order) - 1) * (len(node_order) - 2))
        if len(node_order) > 2
        else None
    )
    for bc_type, node_bc in b_c["node"].items():
        # Initialize with 0.0 to ensure all edges
        nx.set_node_attributes(
            graph,
            0.0,
            f"node_betweenness_{bc_type}{attr_suffix}",
        )
        # Write non-zero values
        nx.set_node_attributes(
            graph,
            {
                node_order[node_idx]: bc * scale
                for node_idx, bc in enumerate(node_bc)
                if bc != 0.0
            },
            f"node_betweenness_{bc_type}{attr_suffix}",
        )
    # Normalize edge betweenness values and write to graph
    scale = (
        1 / (len(node_order) * (len(node_order) - 1)) if len(node_order) > 1 else None
    )
    for bc_type, edge_bc in b_c["edge"].items():
        # Initialize with 0.0 to ensure all edges
        nx.set_edge_attributes(
            graph,
            0.0,
            f"edge_betweenness_{bc_type}{attr_suffix}",
        )
        # Write non-zero values
        nx.set_edge_attributes(
            graph,
            {
                (
                    u_id,
                    v_id,
                    min(
                        graph.get_edge_data(u_id, v_id).items(),
                        key=lambda item: item[1][weight] if weight else 0,
                    )[0],
                ): edge_bc[
                    np.searchsorted(
                        edges_uv_id,
                        __edge_to_1d(
                            node_order.index(u_id),
                            node_order.index(v_id),
                            len(str(len(graph))),
                        ),
                    )
                ]
                * scale
                for (u_id, v_id) in graph.edges(keys=False)
            },
            f"edge_betweenness_{bc_type}{attr_suffix}",
        )
    logger.debug(
        "Calculated betweenness centrality in %s seconds.",
        timedelta(seconds=time() - start_time),
    )


def _calculate_betweenness(
    edges_uv_id, pred, dist, edge_padding, index_subset=None, max_range=None
):
    """Calculate the betweenness centralities for the nodes and edges.

    Parameters
    ----------
    edges_uv_id : np.ndarray
        List of edges in the graph, as returned by
        :func:`superblockify.metrics.measures.__edges_to_1d`
    pred : np.ndarray
        Predecessors matrix of the graph, as returned by
        :func:`superblockify.metrics.distances.calculate_path_distance_matrix`
    dist : np.ndarray
        Distance matrix of the graph, as returned by
        :func:`superblockify.metrics.distances.calculate_path_distance_matrix`
    edge_padding : int
        Number of digits to pad the edge indices with, :attr:`max_len` of the nodes
    index_subset : list, optional
        List of node indices to calculate the betweenness centrality for, by default
        None. Used to calculate k-betweenness centrality
    max_range : float, optional
        Maximum range to calculate the betweenness centrality for, by default None.
        Used to calculate range dependent betweenness centrality.
        None is interpreted as all nodes.
    """

    node_indices = np.arange(pred.shape[0])
    pred = pred.astype(np.int32)
    dist = dist.astype(np.float32)

    betweennesses = _sum_bc(
        np.array(index_subset if index_subset else node_indices, dtype=np.int32),
        pred,
        dist,
        edges_uv_id,
        int32(edge_padding),
        float32(max_range) if max_range else float32("inf"),
    )

    return {
        "node": {
            "normal": betweennesses[: len(node_indices), 0],
            "length": betweennesses[: len(node_indices), 1],
            "linear": betweennesses[: len(node_indices), 2],
        },
        "edge": {
            "normal": betweennesses[len(node_indices) :, 0],
            "length": betweennesses[len(node_indices) :, 1],
            "linear": betweennesses[len(node_indices) :, 2],
        },
    }


@njit(int32[:](float32[:], float32), parallel=False)
def _single_source_given_paths_simplified(dist_row, max_range):  # pragma: no cover
    """Sort nodes, predecessors and distances by distance.

    Parameters
    ----------
    dist_row : np.ndarray, 1D
        Distance row un-sorted.

    Returns
    -------
    S : np.ndarray, 1D
        List of node indices in order of distance, non-decreasing.

    Notes
    -----
    Does not include endpoints.
    """
    dist_order = np.argsort(dist_row)
    # Remove unreachable indices (inf), check from back which is the first
    # reachable node
    while dist_row[dist_order[-1]] >= max_range:
        dist_order = dist_order[:-1]
    # Remove immediately reachable nodes with distance 0, including s itself
    while dist_row[dist_order[0]] == 0:
        dist_order = dist_order[1:]
    return dist_order.astype(np.int32)


@njit(float64[:, :](int32, int32[:], float32[:], int64[:], int32, float32))
def __accumulate_bc(
    s_idx,
    pred_row,
    dist_row,
    edges_uv,
    edge_padding,
    max_range,
):  # pragma: no cover
    # pylint: disable=too-many-locals
    """Calculate the betweenness centrality for a single source node.

    Parameters
    ----------
    s_idx : int
        Index of the source node.
    pred_row : np.ndarray
        Predecessors row of the graph.
    dist_row : np.ndarray
        Distance row of the graph.
    edges_uv : np.ndarray, 1D
        Array of concatenated edge indices, sorted in ascending order.
    edge_padding : int
        Number of digits to pad the edge indices with, :attr:`max_len` of the nodes.
    max_range : float
        Maximum range to calculate the betweenness centrality for.

    Returns
    -------
    node_bc : np.ndarray
        Array of node and edge betweenness centralities.
    """
    betweennesses = np.zeros((len(pred_row) + len(edges_uv), 3), dtype=np.float64)

    s_queue_idx = _single_source_given_paths_simplified(dist_row, max_range)
    # delta = dict.fromkeys(node_indices, 0)
    # delta as 1d-ndarray
    delta = np.zeros(len(pred_row))
    delta_len = delta.copy()
    # s_queue_idx is 1d-ndarray, while not empty
    for w_idx in s_queue_idx[::-1]:  # flip the array to loop non-increasingly
        # No while loop over multiple predecessors, only one path per node pair
        pre_w = pred_row[w_idx]  # P[w_idx]
        dist_w = dist_row[w_idx]  # D[w_idx]
        # Calculate dependency contribution
        coeff = 1 + delta[w_idx]
        coeff_len = 1 / dist_w + delta[w_idx]
        # Find concatenated edge index (u, v) for edge (pre_w, w_idx) in presorted
        # edges_uv
        edge_idx = np.searchsorted(edges_uv, __edge_to_1d(pre_w, w_idx, edge_padding))
        # Add edge betweenness contribution
        betweennesses[len(pred_row) + edge_idx, 0] += coeff
        betweennesses[len(pred_row) + edge_idx, 1] += coeff_len
        betweennesses[len(pred_row) + edge_idx, 2] += dist_w * coeff_len
        # Add to dependency for further nodes/loops
        delta[pre_w] += coeff
        delta_len[pre_w] += coeff_len
        # Add node betweenness contribution
        if w_idx != s_idx:
            betweennesses[w_idx, 0] += delta[w_idx]
            betweennesses[w_idx, 1] += delta_len[w_idx]
            betweennesses[w_idx, 2] += dist_w * delta_len[w_idx]
    return betweennesses


@njit(  # return of two 1d float64 arrays including node and edge betweenness
    float64[:, :](int32[:], int32[:, :], float32[:, :], int64[:], int32, float32),
    parallel=True,
    fastmath=False,
)
def _sum_bc(
    loop_indices, pred, dist, edges_uv, edge_padding, max_range
):  # pragma: no cover
    """Calculate the betweenness centrality for a single source node.

    Parameters
    ----------
    loop_indices : np.ndarray
        Array of node indices to loop over.
    pred : np.ndarray
        Predecessors row of the graph.
    dist : np.ndarray
        Distance row of the graph.
    edges_uv : np.ndarray, 1D
        Array of concatenated edge u and v indices, sorted in ascending order.
    edge_padding : int
        Number of digits to pad edge indices with. Used to convert edge indices
        to 1D indices.
    max_range : float
        Maximum path distances to consider for betweenness calculation.
    Returns
    -------
    node_bc : tuple of ndarray
        Tuple of node betweenness and edge betweenness.

    Notes
    -----
    The edges_u and edges_v arrays are sorted together, this way indices can be
    found using a binary search. This is faster than using np.where.
    """

    betweennesses = np.zeros((len(pred) + len(edges_uv), 3), dtype=np.float64)
    # The first len(pred) rows correspond to node betweenness; the rest to edge
    # The 3 layers correspond to normal, length-scaled, and linearly scaled

    # Loop over nodes to collect betweenness using pair-wise dependencies
    for idx in prange(loop_indices.shape[0]):  # pylint: disable=not-an-iterable
        betweennesses += __accumulate_bc(
            loop_indices[idx],
            pred[loop_indices[idx]],
            dist[loop_indices[idx]],
            edges_uv,
            edge_padding,
            max_range,
        )
    return betweennesses


def calculate_high_bc_clustering(node_x, node_y, node_betweenness, percentile):
    """
    Calculate the high betweenness clustering coefficient and anisotropy for a
    given percentile of nodes with the highest betweenness. [1]_

    Parameters
    ----------
    node_x : list
        List of x coordinates of the nodes.
    node_y : list
        List of y coordinates of the nodes, ordered by node index.
    node_betweenness : list
        List of betweenness values for each node, ordered by node index.
    percentile : float
        Percentile of nodes with the highest betweenness to calculate the
        clustering coefficient for. Between 0 and 1.

    Returns
    -------
    high_bc_clustering : float
        Clustering coefficient for the nodes with the highest betweenness.
    high_bc_anisotropy : float
        Anisotropy for the nodes with the highest betweenness.

    Notes
    -----
    The high betweenness clustering coefficient is calculated as the average
    clustering coefficient of the nodes with the highest betweenness. The
    high betweenness anisotropy is calculated as the average anisotropy of the
    nodes with the highest betweenness.

    References
    ----------
    .. [1] Kirkley, A., Barbosa, H., Barthelemy, M. & Ghoshal, G. From the betweenness
           centrality in street networks to structural invariants in random planar
           graphs. Nat Commun 9, 2501 (2018).
           https://www.nature.com/articles/s41467-018-04978-z
    """
    coord_bc = np.array([node_x, node_y, node_betweenness]).T
    # Sort by betweenness
    coord_bc = coord_bc[coord_bc[:, 2].argsort()]
    # Threshold betweenness
    threshold_idx = int(len(coord_bc) * percentile)
    return __calculate_high_bc_clustering(
        coord_bc, threshold_idx
    ), __calculate_high_bc_anisotropy(coord_bc[threshold_idx:, :2])


def __calculate_high_bc_clustering(coord_bc, threshold_idx):
    r"""High betweenness nodes clustering coefficient.

    .. math::
        C_{\theta} =
        \frac{1}{N_{\theta}\left\langle X \right\rangle}
        \sum_{i = 1}^{N_{\theta}} \| x_i - x_{\mathrm{cm}, \theta} \|

    .. math::
        \langle X \rangle = \frac{1}{N}
        \sum_{i = 1}^{N} \| x_i - x_{\mathrm{cm}, \theta} \|

    .. math::
        x_{\mathrm{cm}, \theta} = \frac{1}{N_{\theta}}
        \sum_{i = 1}^{N_{\theta}} x_i

    The distance calculation :math:`\| x_i - x_{\mathrm{cm}, \theta} \|` includes the
    x and y coordinates of the node, and is the Euclidean distance. In this case, it
    is the Frobenius norm of the difference between the node coordinates and the
    center of mass of the high betweenness nodes.

    Parameters
    ----------
    coord_bc : np.ndarray
        Array of node coordinates and betweenness values, sorted by betweenness.
    threshold_idx : int
        Index of the first node to consider as high betweenness.

    Returns
    -------
    high_bc_clustering : float
        Clustering coefficient for the nodes with the highest betweenness.

    Raises
    ------
    ValueError
        If the coordinate array has less than two nodes.
    ValueError
        If the threshold index is greater than the number of nodes.
    """
    if len(coord_bc) < 2:
        raise ValueError("Coordinate array must have at least two nodes.")
    if threshold_idx >= len(coord_bc):
        raise ValueError("Threshold index must be less than the number of nodes.")
    # High betweenness nodes center of mass
    high_bc_cm = np.mean(coord_bc[threshold_idx:, :2], axis=0)
    # Average distance to center of mass
    avg_dist = np.mean(
        np.linalg.norm(coord_bc[threshold_idx:, :2] - high_bc_cm, axis=1)
    )
    # Norm by average distance of all nodes
    return avg_dist / np.mean(np.linalg.norm(coord_bc[:, :2] - high_bc_cm, axis=1))


def __calculate_high_bc_anisotropy(coord_high_bc):
    r"""High betweenness nodes anisotropy.

    The high betweenness anisotropy is the ratio
    :math:`A_{\theta}=\lambda_1/\lambda_2`, where :math:`\lambda_i` are the positive
    eigenvalues of the covariance matrix of the high betweenness nodes, and
    :math:`\lambda_1 \geq \lambda_2`. [1]_

    Parameters
    ----------
    coord_high_bc : np.ndarray
        Array of the high betweenness nodes coordinates.

    Returns
    -------
    high_bc_anisotropy : float
        Anisotropy for the nodes with the highest betweenness.

    Raises
    ------
    ValueError
        If the number of high betweenness nodes is less than 2.

    References
    ----------
    .. [1] Kirkley, A., Barbosa, H., Barthelemy, M. & Ghoshal, G. From the betweenness
           centrality in street networks to structural invariants in random planar
           graphs. Nat Commun 9, 2501 (2018).
           https://www.nature.com/articles/s41467-018-04978-z
    """
    if len(coord_high_bc) < 2:
        raise ValueError(
            "High betweenness nodes must be at least 2, for less the anisotropy is "
            "not defined."
        )
    # Covariance matrix
    cov = np.cov(coord_high_bc.T)
    # Eigenvalues
    eigvals = np.linalg.eigvals(cov)
    # Sort eigenvalues
    eigvals = np.sort(eigvals)[::-1]
    # Anisotropy
    return eigvals[0] / eigvals[1]


def add_ltn_means(components, edge_attr):
    """Add mean of attributes to each Superblock.

    Writes the mean of the specified edge attribute(s) to each Superblock in the list of
    components. The mean is calculated as the mean of each attribute in the Superblock
    subgraph.
    Works in-place and adds `mean_{attr}` to each Superblock.

    Parameters
    ----------
    components : list of dict
        List of dictionaries of Superblock components.
    edge_attr : key or list of keys
        Edge attribute(s) to calculate the mean of.
    """
    # Loop over Superblocks
    for component in components:
        # Loop over attributes
        for attr in edge_attr if isinstance(edge_attr, list) else [edge_attr]:
            # Calculate mean
            component[f"mean_{attr}"] = aggregate_edge_attr(
                component["subgraph"], attr, np.mean, dismiss_none=True
            )


def add_relative_changes(components, attr_pairs):
    """Add relative difference of attributes to each Superblock.

    Measured in terms of percentual increase using
    :func:`superblockify.utils.percentual_increase`.

    Write the relative percentual change of the specified edge attribute(s) to each
    Superblock in the list of components.
    The relative change is the percentual change of the first to the second attribute.
    Works in-place and adds `change_{attr1}` to each Superblock.
    If `attr1` has a value of 2 and `attr2` has a value of 1, the relative change is
    -0.5, a 50% decrease. If `attr1` has a value of 4 and `attr2` has a value of 6,
    the relative change is 0.5, a 50% increase.

    Parameters
    ----------
    components : list of dict
        List of dictionaries of Superblock components.
    attr_pairs : list of tuples with two keys
        List of attribute pairs to calculate the relative change of.

    Raises
    ------
    KeyError
        If any key cannot be found in the Superblocks.
    """
    # Loop over Superblocks
    for component in components:
        # Loop over attribute pairs
        for attr1, attr2 in (
            attr_pairs if isinstance(attr_pairs, list) else [attr_pairs]
        ):
            # Calculate relative change
            try:
                component[f"change_{attr1}"] = percentual_increase(
                    component[attr1], component[attr2]
                )
            except KeyError as err:
                raise KeyError(f"Key {err} not found in Superblocks.") from err
