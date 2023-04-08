"""Distance calculation for the network metrics."""
import logging
from datetime import timedelta
from multiprocessing import cpu_count, Pool
from time import time

import numpy as np
from networkx import to_scipy_sparse_array
from osmnx.projection import is_projected
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from .plot import plot_distance_distributions
from ..utils import has_pairwise_overlap

logger = logging.getLogger("superblockify")

_AVG_EARTH_RADIUS_M = 6.3781e6  # in meters, arXiv:1510.07674 [astro-ph.SR]


def calculate_distance_matrices(
    node_list, partitioner, weight, chunk_size, make_plots, num_workers
):
    """Calculate the distance matrices for the partitioning.

    Parameters
    ----------
    node_list : list
        The list of nodes to calculate the distance matrices for
    partitioner : superblockify.Partitioner
        The partitioner to calculate the distance matrices for
    weight : str, optional
        The edge attribute to use as weight, if None count hops
    chunk_size : int
        The chunk size for the multiprocessing pool
    make_plots : bool
        If True, the distance distributions are plotted
    num_workers : int
        The number of workers to use for the multiprocessing pool

    Returns
    -------
    dict
        The distance matrices for the partitioning. The keys are the distance
        matrix types, the values are the distance matrices, corresponding to the
        node order in ``node_list``.
    dict
        The predecessors for the distance matrices. The keys correspond to the
        keys of the distance matrices, the values are the predecessors for the
        distance matrices.
    """

    dist_matrix = {
        # Euclidean distances (E)
        "E": calculate_euclidean_distance_matrix_projected(
            partitioner.graph,
            node_order=node_list,
            plot_distributions=make_plots,
        )
    }

    predecessors = {}

    # On the full graph (S)
    dist_matrix["S"], predecessors["S"] = calculate_path_distance_matrix(
        partitioner.graph,
        weight=weight,
        node_order=node_list,
        plot_distributions=make_plots,
    )
    # On the partitioning graph (N)
    dist_matrix["N"], predecessors["N"] = calculate_partitioning_distance_matrix(
        partitioner,
        weight=weight,
        node_order=node_list,
        num_workers=num_workers,
        chunk_size=chunk_size,
        plot_distributions=make_plots,
    )

    return dist_matrix, predecessors


def calculate_path_distance_matrix(
    graph,
    weight=None,
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
        before node j on the shortest path from node i to node j. [3]_

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
            distance_unit="khops"
            if weight is None
            else "km"
            if weight == "length"
            else f"k{weight}",
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
    node_order=None,
    num_workers=None,
    chunk_size=1,
    plot_distributions=False,
    check_overlap=True,
):  # pylint: disable=too-many-locals
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
    node_order : list, optional
        The order of the nodes in the distance matrix. If None, the ordering is
        produced by graph.nodes().
    num_workers : int, optional
        The maximal number of workers used to process distance matrices. If None,
        the number of workers is set to min(32, cpu_count() // 2).
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
        If the partitions overlap node-wise. For nodes considered to be in the
        partition, see `BasePartitioner.get_partition_nodes()`.

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

    partitions = {
        part["name"]: {
            "subgraph": part["subgraph"],
            "nodes": list(part["nodes"]),  # exclusive nodes inside the subgraph
            "nodelist": list(part["subgraph"]),  # also nodes shared with the
            # sparsified graph or
        }
        for part in partitioner.get_partition_nodes()
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

    logger.debug("Preparing combinations for processing.")
    start_time = time()

    combs = (
        (
            name,
            to_scipy_sparse_array(
                part["subgraph"],
                weight=weight,
                format="csr",
                nodelist=part["nodelist"],
            ),
        )
        for name, part in partitions.items()
    )

    # Calculate the combinations in parallel
    logger.debug(
        "Calculating %d separate distance matrices in parallel, "
        "with %d workers and chunk-size %d.",
        len(partitions),
        num_workers,
        chunk_size,
    )
    # Parallelized calculation with `p.imap_unordered`
    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    dijkstra_param,
                    combs,
                    chunksize=chunk_size,
                ),
                desc="Calculating distance matrices",
                total=len(partitions),
                unit_scale=1,
            )
        )
    logger.debug(
        "Finished calculating %d separate distance matrices in %s. "
        "Combining the results.",
        len(partitions),
        timedelta(seconds=time() - start_time),
    )

    # Construct the distance matrix for the partitioning, half-precision float
    dist_matrix = np.full((len(node_order), len(node_order)), np.inf, dtype=np.half)
    pred_matrix = np.full((len(node_order), len(node_order)), -9999, dtype=np.int32)
    node_order_indices = list(range(len(node_order)))

    # Construct matrices for the separate distances and predecessors
    # sort `results` to ensure that the sparsified partition is last
    # as the shared nodes should be determined by the sparsified graph
    for name, (dist, pred) in sorted(
        results, key=lambda x: x[0] != "sparsified", reverse=True
    ):
        partitions[name]["node_order_idx"] = [
            node_order.index(n) for n in partitions[name]["nodelist"]
        ]
        dist_matrix[
            np.ix_(
                partitions[name]["node_order_idx"],
                partitions[name]["node_order_idx"],
            )
        ] = dist

        def predecessors_vectorized(p_simple):
            if p_simple != -9999:
                return node_order_indices[
                    partitions[name]["node_order_idx"][p_simple]
                ]  # pylint: disable=cell-var-from-loop
            return p_simple

        pred_matrix[
            np.ix_(
                partitions[name]["node_order_idx"],
                partitions[name]["node_order_idx"],
            )
        ] = np.vectorize(predecessors_vectorized)(pred)

    logger.debug("Constructing csr matrix for simplified calculation.")

    ## Fill up distances
    # Construct simplified graph
    graph_restricted = dist_matrix.copy()
    n_partition_indices = [
        i
        for i in node_order_indices
        if i not in partitions["sparsified"]["node_order_idx"]
    ]
    graph_restricted[np.ix_(n_partition_indices, n_partition_indices)] = np.inf

    # Construct Compressed Sparse Row matrix
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    data = graph_restricted.flatten()
    row, col = np.indices(graph_restricted.shape)
    row, col = row.flatten(), col.flatten()
    # remove the diagonal
    data = data[row != col]
    row, col = row[row != col], col[row != col]
    # remove inf values
    data, row, col = data[data != np.inf], row[data != np.inf], col[data != np.inf]

    graph_restricted = csr_matrix(
        (data, (row, col)), shape=(len(node_order), len(node_order))
    )
    logger.debug(
        "Constructed csr matrix of sparsity %.2f%% (%d/%d) for simplified "
        "calculation. Running Dijkstra's algorithm...",
        100 * (1 - graph_restricted.nnz / len(node_order) ** 2),
        graph_restricted.nnz,
        len(node_order) ** 2,
    )
    start_time = time()
    dist_simple, pred_simple = dijkstra(
        graph_restricted, directed=True, return_predecessors=True
    )
    logger.debug(
        "Restricted all-pairs shortest path lengths for graph with $d partitions, "
        " %d nodes and %d edges calculated in %s. Reconstructing predecessors...",
        len(partitions) - 1,
        len(node_order),
        time() - start_time,
    )
    min_mask = dist_simple < dist_matrix
    dist_matrix = np.where(min_mask, dist_simple, dist_matrix)
    pred_matrix[min_mask] = pred_matrix[pred_simple[min_mask], np.where(min_mask)[1]]

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
            distance_unit="hops"
            if weight is None
            else "km"
            if weight == "length"
            else weight,
        )

    return dist_matrix, pred_matrix


def dijkstra_param(comb):
    """Wrapper for the dijkstra function.

    Fixes keyword arguments for the dijkstra function.
    """
    name, sparse_matrix = comb

    return name, dijkstra(
        sparse_matrix,
        directed=True,
        indices=None,  # all nodes, sorted as in sparse_matrix
        return_predecessors=True,
        unweighted=False,
    )
