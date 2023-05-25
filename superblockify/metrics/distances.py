"""Distance calculation for the network metrics."""
from datetime import timedelta
from itertools import combinations
from multiprocessing import cpu_count, Pool
from time import time

import numpy as np
from networkx import to_scipy_sparse_array
from osmnx.projection import is_projected
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from .plot import plot_distance_distributions
from ..config import logger
from ..utils import has_pairwise_overlap

_AVG_EARTH_RADIUS_M = 6.3781e6  # in meters, arXiv:1510.07674 [astro-ph.SR]


def calculate_path_distance_matrix(
    graph,
    weight=None,
    unit_symbol=None,
    node_order=None,
    plot_distributions=False,
    log_debug=True,
):
    """Calculate the distance matrix for the partitioning.

    Use cythonized scipy.sparse.csgraph functions to calculate the distance matrix.

    Generally Dijkstra's algorithm with a Fibonacci heap is used. It's approximate
    computational cost is ``O[N(N*k + N*log(N))]`` where ``N`` is the number of
    nodes and ``k`` is the average number of edges per node. We use this,
    because our graphs are usually sparse. For dense graphs, the Floyd-Warshall
    algorithm can be implemented with ``O[N^3]`` computational cost. [1]_

    Runtime comparison:

    - Scheveningen, NL (N = 1002, E = 2329):
        - Dijkstra: 172ms
        - Floyd-Warshall: 193ms
    - Liechtenstein, LI (N = 1797, E = 4197):
        - Dijkstra: 498ms
        - Floyd-Warshall: 917ms
    - Copenhagen, DK (N = 7806, E = 19565):
        - Dijkstra: 14.65s
        - Floyd-Warshall: 182.03s
    - Barcelona, ES (N = 8812, E = 16441):
        - Dijkstra: 18.21s
        - Floyd-Warshall: 134.69s
    (simple, one-time execution)

    The input graph will be converted to a scipy sparse matrix in CSR format.
    Compressed Sparse Row format is a sparse matrix format that is efficient for
    arithmetic operations. [2]_

    Parameters
    ----------
    graph : networkx.Graph
        The graph to calculate the distance matrix for
    weight : str, optional
        The edge attribute to use as weight. If None, all edge weights are 1.
    unit_symbol : str, optional
        The unit symbol to use for the distance matrix, like 's' for seconds.
    node_order : list, optional
        The order of the nodes in the distance matrix. If None, the ordering is
        produced by graph.nodes().
    plot_distributions : bool, optional
        If True, a histogram of the distribution of the shortest path lengths is
        plotted.
    log_debug : bool, optional
        If True, log runtime and graph information at debug level.

    Raises
    ------
    ValueError
        If the graph has negative edge weights.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the shortest
        path length from node i to node j.
    predecessors : ndarray
        The predecessors for the distance matrix. predecessors[i, j] is the node
        before node j on the shortest path from node i to node j. [1]_

    References
    ----------
    .. [1] SciPy 1.10.0 Reference Guide, scipy.sparse.csgraph.shortest_path
       https://docs.scipy.org/doc/scipy-1.10.0/reference/generated/scipy.sparse.csgraph.shortest_path.html
       (accessed February 21, 2023)
    .. [2] SciPy 1.10.0 Reference Guide, scipy.sparse.csr_matrix
       https://docs.scipy.org/doc/scipy-1.10.0/reference/generated/scipy.sparse.csr_matrix.html
       (accessed February 21, 2023)

    """

    if weight is not None and any(
        w < 0 for (u, v, w) in graph.edges.data(weight, default=0)
    ):
        # For this case Johnson's algorithm could be used, but none of our graphs
        # should have negative edge weights.
        raise ValueError("Graph has negative edge weights.")

    # First get N x N array of distances representing the input graph.
    graph_matrix = to_scipy_sparse_array(
        graph, weight=weight, format="csr", nodelist=node_order
    )
    start_time = time()
    dist_full_graph, predecessors = dijkstra(
        graph_matrix, directed=True, return_predecessors=True, unweighted=False
    )

    # Convert to half-precision to save memory
    dist_full_graph = dist_full_graph.astype(np.half)

    if log_debug:
        logger.debug(
            "All-pairs shortest path lengths for graph with %s nodes and %s edges "
            "calculated in %s.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            timedelta(seconds=time() - start_time),
        )
    if plot_distributions:
        if node_order is None:
            node_order = graph.nodes()
        # Where `dist_full_graph` is inf, replace with 0
        plot_distance_distributions(
            dist_full_graph[dist_full_graph != np.inf],
            dist_title="Distribution of shortest path lengths on full graph",
            coords=(
                [graph.nodes[node]["x"] for node in node_order],
                [graph.nodes[node]["y"] for node in node_order],
            ),
            coord_title="Coordinates of nodes",
            labels=("x", "y"),
            distance_unit=unit_symbol,
        )

    return dist_full_graph, predecessors


def calculate_euclidean_distance_matrix_projected(
    graph, node_order=None, plot_distributions=False
):
    """Calculate the Euclidean distances between all nodes in the graph.

    Uses the x and y coordinates of the nodes of a projected graph. The coordinates
    are in meters.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to calculate the distance matrix for. The graph should be
        projected.
    node_order : list, optional
        The order of the nodes in the distance matrix. If None, the ordering is
        produced by graph.nodes().
    plot_distributions : bool, optional
        If True, plot the distributions of the Euclidean distances and coordinates.
        Sanity check for the coordinate values.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the Euclidean
        distance between node i and node j.

    Raises
    ------
    ValueError
        If the graph is not projected.

    """

    # Find CRS from graph's metadata
    if "crs" not in graph.graph or not is_projected(graph.graph["crs"]):
        raise ValueError("Graph is not projected.")

    # Get the node order
    if node_order is None:
        node_order = list(graph.nodes())

    # Get the coordinates of the nodes
    x_coord = np.array([graph.nodes[node]["x"] for node in node_order])
    y_coord = np.array([graph.nodes[node]["y"] for node in node_order])

    # Check that all values are float or int and not inf or nan
    if not np.issubdtype(x_coord.dtype, np.number) or not np.issubdtype(
        y_coord.dtype, np.number
    ):
        raise ValueError("Graph has non-numeric coordinates.")
    if np.any(np.isinf(x_coord)) or np.any(np.isinf(y_coord)):
        raise ValueError("Graph has infinite coordinates.")

    # Calculate the Euclidean distances between all nodes
    dist_matrix = np.sqrt(
        np.square(x_coord[:, np.newaxis] - x_coord[np.newaxis, :])
        + np.square(y_coord[:, np.newaxis] - y_coord[np.newaxis, :])
    )
    # Convert to half-precision to save memory
    dist_matrix = dist_matrix.astype(np.half)

    if plot_distributions:
        plot_distance_distributions(
            dist_matrix,
            dist_title="Distribution of Euclidean distances",
            coords=(x_coord, y_coord),
            coord_title="Scatter plot of projected coordinates",
            labels=("x", "y"),
        )

    return dist_matrix


def calculate_euclidean_distance_matrix_haversine(
    graph, node_order=None, plot_distributions=False
):
    """Calculate the Euclidean distances between all nodes in the graph.

    Uses the **Haversine formula** to calculate the distances between all nodes in
    the graph. The coordinates are in degrees.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to calculate the distance matrix for
    node_order : list, optional
        The order of the nodes in the distance matrix. If None, the ordering is
        produced by graph.nodes().
    plot_distributions : bool, optional
        If True, plot the distributions of the Euclidean distances and coordinates.
        Sanity check for the coordinate values.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the Euclidean
        distance between node i and node j.

    Raises
    ------
    ValueError
        If coordinates are not numeric or not in the range [-90, 90] for latitude
        and [-180, 180] for longitude.

    """

    if node_order is None:
        node_order = list(graph.nodes())

    start_time = time()

    # Calculate the Euclidean distances between all nodes
    # Do vectorized calculation for all nodes
    lat = np.array([graph.nodes[node]["lat"] for node in node_order])
    lon = np.array([graph.nodes[node]["lon"] for node in node_order])

    # Check that all values are float or int and proper lat/lon values
    if not np.issubdtype(lat.dtype, np.number) or not np.issubdtype(
        lon.dtype, np.number
    ):
        raise ValueError("Latitude and longitude values must be numeric.")
    if np.any(lat > 90) or np.any(lat < -90):
        raise ValueError("Latitude values are not in the range [-90, 90].")
    if np.any(lon > 180) or np.any(lon < -180):
        raise ValueError("Longitude values are not in the range [-180, 180].")

    node1_lat = np.expand_dims(lat, axis=0)
    node1_lon = np.expand_dims(lon, axis=0)
    node2_lat = np.expand_dims(lat, axis=1)
    node2_lon = np.expand_dims(lon, axis=1)

    # Calculate haversine distance,
    # see https://en.wikipedia.org/wiki/Haversine_formula
    # and https://github.com/mapado/haversine/blob/master/haversine/haversine.py
    lat = node2_lat - node1_lat
    lon = node2_lon - node1_lon
    hav = (
        np.sin(lat / 2) ** 2
        + np.cos(node1_lat) * np.cos(node2_lat) * np.sin(lon / 2) ** 2
    )
    dist_matrix = 2 * _AVG_EARTH_RADIUS_M * np.arcsin(np.sqrt(hav))
    logger.debug(
        "Euclidean distances for graph with %s nodes and %s edges "
        "calculated in %s. "
        "Min/max lat/lon values: %s, %s, %s, %s; Difference: %s, %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        timedelta(seconds=time() - start_time),
        np.min(node1_lat),
        np.max(node1_lat),
        np.min(node1_lon),
        np.max(node1_lon),
        np.max(node1_lat) - np.min(node1_lat),
        np.max(node1_lon) - np.min(node1_lon),
    )

    if plot_distributions:
        # Plot distribution of distances and scatter plot of lat/lon
        plot_distance_distributions(
            dist_matrix,
            dist_title="Distribution of Euclidean distances",
            coords=(node1_lon, node1_lat),
            coord_title="Scatter plot of lat/lon",
            labels=("Longitude", "Latitude"),
        )

    return dist_matrix


def calculate_partitioning_distance_matrix(
    partitioner,
    weight=None,
    unit_symbol=None,
    node_order=None,
    num_workers=None,
    chunk_size=1,
    plot_distributions=False,
    check_overlap=True,
):
    """Calculate the distance matrix for the partitioning.

    This is the pairwise distance between all pairs of nodes, where the shortest
    paths are only allowed to traverse edges in the start and goal partitions and
    unpartitioned edges.
    For this we calculate the distances and predecessors on the sparsified graph and the
    subgraphs separately. Then we combine the distances and predecessors to get a full
    distance matrix.
    We cannot do one big calculation, because the restrictions, to only enter/leave,
    are dynamic and depend on the start and goal node.

    Parameters
    ----------
    partitioner : BasePartitioner
        The partitioner to calculate the distance matrix for
    weight : str, optional
        The edge attribute to use as weight. If None, all edges have weight 1.
    unit_symbol : str, optional
        The unit symbol to use for the weight.
    node_order : list, optional
        The order of the nodes in the distance matrix. If None, the ordering is
        produced by graph.nodes().
    num_workers : int, optional
        The maximal number of workers used to process distance matrices. If None,
        the number of workers is set to ``min(32, cpu_count() // 2)``.
        Choose this number carefully, as it can lead to memory errors if too high,
        if the graph has partitions. In this case another partitioner approach
        might yield better results.
    chunk_size : int, optional
        The chunk-size to use for the multiprocessing pool. This is the number of
        partitions for which the distance matrix is calculated in one go (thread).
        Keep this low if the graph is big or has many partitions. We suggest to
        keep this at 1.
    plot_distributions : bool, optional
        If True, plot the distributions of the Euclidean distances and coordinates.
    check_overlap : bool, optional
        If True, check that the partitions do not overlap node-wise.

    Raises
    ------
    ValueError
        If partitions don't have unique names.
    ValueError
        If the partitions overlap node-wise. For nodes considered to be in the
        partition, see :func:`BasePartitioner.get_partition_nodes()
        <superblockify.partitioning.partitioner.BasePartitioner.get_partition_nodes()>`.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the distance
        between node i and node j for the given rules of the partitioning.
        In the order of `node_order` if given, otherwise as produced by `graph.nodes()`.
    predecessors : ndarray
        The predecessor matrix for the partitioning. predecessors[i, j] is the
        predecessor of node j on the shortest path from node i for the given rules
        of the partitioning.
    """

    if node_order is None:
        node_order = list(partitioner.graph.nodes)

    if num_workers is None:
        num_workers = min(32, cpu_count() // 2)
        logger.debug("No number of workers specified, using %s.", num_workers)

    partitions = partitioner.get_partition_nodes()
    # Check that the partitions have unique names
    if len(partitions) != len({part["name"] for part in partitions}):
        raise ValueError("Partitions must have unique names.")

    partitions = {
        part["name"]: {
            "subgraph": part["subgraph"],
            "nodes": list(part["nodes"]),  # exclusive nodes inside the subgraph
            "nodelist": list(part["subgraph"]),  # also nodes shared with the
            # sparsified graph or on partition boundaries
        }
        for part in partitions
    }

    # Check that none of the partitions overlap by checking that the intersection
    # of the nodes in each partition is empty.
    if check_overlap:
        pairwise_overlap = has_pairwise_overlap(
            [list(part["nodes"]) for part in partitions.values()]
        )
        # Check if any element in the pairwise overlap matrix is True, except the
        # diagonal
        if np.any(pairwise_overlap[np.triu_indices_from(pairwise_overlap, k=1)]):
            raise ValueError("The partitions overlap node-wise. This is not allowed.")

    partitions["sparsified"] = {
        "subgraph": partitioner.sparsified,
        "nodes": list(partitioner.sparsified.nodes),
        "nodelist": list(partitioner.sparsified.nodes),
    }

    dist_matrix, pred_matrix = shortest_paths_restricted(
        partitioner.graph, partitions, weight, node_order, num_workers, chunk_size
    )

    if plot_distributions:
        # Where `dist_full_graph` is inf, replace with 0
        plot_distance_distributions(
            dist_matrix[dist_matrix != np.inf],
            dist_title="Distribution of shortest path distances for the "
            "partitioning",
            coords=(
                [partitioner.graph.nodes[node]["x"] for node in node_order],
                [partitioner.graph.nodes[node]["y"] for node in node_order],
            ),
            coord_title="Coordinates of nodes",
            labels=("x", "y"),
            distance_unit=unit_symbol,
        )

    return dist_matrix, pred_matrix


def shortest_paths_restricted(
    graph, partitions, weight, node_order, num_workers=2, chunk_size=1
):
    """Calculate restricted shortest paths.

    Shortest distances and predecessors with restrictions of not passing through
    partitions. The passed partitions is a dictionary with the partition names as
    keys and the partition as value. The partition is a dictionary with the keys
    `subgraph`, `nodes` and `nodelist`. The dict needs to have a special partition,
    called `sparsified`, which is the sparsified graph. The `nodes` key is a list
    of nodes that are exclusive to the partition, i.e. nodes that are not shared
    with the sparsified graph or on partition boundaries. The `nodelist` key is a
    list of nodes that are exclusive to the partition and nodes that are shared
    with the sparsified graph. The subgraphs all need to be graphs views of a shared
    graph. The Partitioner class produces such partitions, after passing the
    integrated :func:`is_valid_partitioning()
    <superblockify.partitioning.checks.is_valid_partitioning>` checks.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to calculate the shortest paths for.
    partitions : dict
        The partitions to calculate the shortest paths for. The keys are the
        partition names and the values are dictionaries with the keys `subgraph`,
        `nodes` and `nodelist`.
    weight : str or None, optional
        The edge attribute to use as weight. If None, all edge weights are 1.
    node_order : list
        The order of the nodes in the distance matrix.
    num_workers : int, optional
        The number of workers to use for the multiprocessing pool. Defaults to 2.
    chunk_size : int, optional
        The chunk size for the multiprocessing pool. Defaults to 1.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning.
    pred_matrix : ndarray
        The predecessor matrix for the partitioning.

    Notes
    -----
    For usage with a :class:`Partitioner
    <superblockify.partitioning.partitioner.BasePartitioner>`,
    see :func:`calculate_partitioning_distance_matrix()
    <superblockify.metrics.distances.calculate_partitioning_distance_matrix>`.
    """
    # pylint: disable=too-many-locals, unused-argument

    # node_order indices
    # filtered indices: sparse/partition
    n_sparse_indices = [node_order.index(n) for n in partitions["sparsified"]["nodes"]]
    part_name_order = [name for name in partitions.keys() if name != "sparsified"]
    n_partition_indices_separate = [
        [node_order.index(n) for n in partitions[name]["nodes"]]
        for name in part_name_order
    ]
    n_partition_indices = [n for part in n_partition_indices_separate for n in part]

    # Semipermeable graphs
    g_leaving = graph.copy()
    # Construct Compressed Sparse Row matrix
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    g_leaving = to_scipy_sparse_array(
        g_leaving, nodelist=node_order, weight=weight, format="coo"
    )
    data, row, col = g_leaving.data, g_leaving.row, g_leaving.col

    # Remove edges between different partitions
    # (col, row in separate n_partition_indices_separate)
    for n_ind_1, n_ind_2 in combinations(n_partition_indices_separate, 2):
        mask = np.logical_and(np.isin(row, n_ind_1), np.isin(col, n_ind_2))
        mask = np.logical_or(
            mask, np.logical_and(np.isin(row, n_ind_2), np.isin(col, n_ind_1))
        )
        data, row, col = data[~mask], row[~mask], col[~mask]
    g_entering = (data, (row, col))

    # Remove edges from partition to sparse
    mask = np.logical_and(
        np.isin(row, n_partition_indices), np.isin(col, n_sparse_indices)
    )
    data, row, col = data[~mask], row[~mask], col[~mask]
    g_leaving = csr_matrix((data, (row, col)), shape=(len(node_order), len(node_order)))
    # Remove edges from sparse to partition
    data, (row, col) = g_entering
    mask = np.logical_and(
        np.isin(row, n_sparse_indices), np.isin(col, n_partition_indices)
    )
    data, row, col = data[~mask], row[~mask], col[~mask]
    g_entering = csr_matrix(
        (data, (row, col)), shape=(len(node_order), len(node_order))
    )

    # Calculate the combinations in parallel
    logger.debug(
        "Calculating 2 semipermeable distance matrices with %d workers.",
        min(2, num_workers),
    )
    start_time = time()
    # Parallelized calculation with `p.imap_unordered`
    with Pool(processes=min(2, num_workers)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    dijkstra_settings,
                    [g_leaving, g_entering],
                    chunksize=1,
                ),
                desc="Calculating distance matrices",
                total=2,
                unit_scale=1,
            )
        )
    logger.debug(
        "Finished calculating distance matrices in %s. Combining the results.",
        timedelta(seconds=time() - start_time),
    )
    min_mask = results[0][0] < results[1][0]
    dist_le, pred_le = results[1]
    dist_le[min_mask] = results[0][0][min_mask]
    pred_le[min_mask] = results[0][1][min_mask]

    # Fill up paths
    n_partition_intersect_indices = [
        list(
            set(partitions["sparsified"]["nodelist"]).intersection(
                partitions[name]["nodelist"]
            )
        )
        for name in part_name_order
    ]
    n_partition_intersect_indices = [
        [node_order.index(n) for n in part_indices]
        for part_indices in n_partition_intersect_indices
    ]

    start_time = time()

    dist_step = np.full_like(dist_le, np.inf)
    pred_step = np.full_like(pred_le, -9999)

    for part_idx, part_intersect in zip(
        n_partition_indices_separate, n_partition_intersect_indices
    ):
        # Add distances from i in part_idx to all j in n_partition_indices
        # to the same j to all k in n_partition_indices
        # There is a different amount of i-j than j-k, calculate all combinations
        # and then mask out the invalid ones
        # add to a new dimension
        dists = (
            dist_le[np.ix_(part_idx, part_intersect)][:, :, np.newaxis]
            + dist_le[np.ix_(part_intersect, n_partition_indices)][np.newaxis, :, :]
        )
        # get index of minimum distance for predecessor, use new axis
        min_idx = np.argmin(dists, axis=1)

        # write minima into dist_step
        dist_step[np.ix_(part_idx, n_partition_indices)] = dists[
            np.arange(dists.shape[0])[:, np.newaxis],
            min_idx,
            np.arange(dists.shape[-1]),
        ]

        # write predecessors into pred_step
        # predecessor i-k is the same as predecessor k-j
        # into pred_step, write the predecessor of k-j
        for n_p, i in enumerate(part_idx):
            for m_p, j in enumerate(n_partition_indices):
                if i == j:
                    continue
                pred_step[i, j] = pred_le[part_intersect[min_idx[n_p, m_p]], j]
        # might be vectorized

    mask = dist_step < dist_le
    dist_le = np.where(mask, dist_step, dist_le)
    pred_le = np.where(mask, pred_step, pred_le)

    logger.debug(
        "Finished filling up paths in %s.",
        timedelta(seconds=time() - start_time),
    )

    return dist_le, pred_le


def dijkstra_settings(sparse_matrix):
    """Wrapper for the dijkstra function.

    Fixes keyword arguments for the dijkstra function.
    """
    return dijkstra(
        sparse_matrix,
        directed=True,
        indices=None,  # all nodes, sorted as in sparse_matrix
        return_predecessors=True,
        unweighted=False,
    )
