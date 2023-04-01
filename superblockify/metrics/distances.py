"""Distance calculation for the network metrics."""
import logging
from datetime import timedelta
from itertools import combinations, chain
from multiprocessing import cpu_count, Pool
from time import time

import numpy as np
from networkx import to_scipy_sparse_array
from osmnx.projection import is_projected
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm

from .plot import plot_distance_distributions
from ..utils import has_pairwise_overlap

logger = logging.getLogger("superblockify")

_AVG_EARTH_RADIUS_M = 6.3781e6  # in meters, arXiv:1510.07674 [astro-ph.SR]


def calculate_distance_matrices(
    node_list, partitioner, weight, approach, chunk_size, make_plots, num_workers
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
    approach : str
        The approach to use for calculating the distances, by default "rep_nodes".
        Can be "rep_nodes" or "full".
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
    """

    if approach not in ["rep_nodes", "full"]:
        raise ValueError(
            f"The approach must be 'rep_nodes' or 'full', got '{approach}'."
        )

    dist_matrix = {
        # Euclidean distances (E)
        "E": calculate_euclidean_distance_matrix_projected(
            partitioner.graph,
            node_order=node_list,
            plot_distributions=make_plots,
        ),
        # On the full graph (S)
        "S": calculate_path_distance_matrix(
            partitioner.graph,
            weight=weight,
            node_order=node_list,
            plot_distributions=make_plots,
        ),
        # On the partitioning graph (N)
        "N": calculate_partitioning_distance_matrix(
            partitioner,
            weight=weight,
            approach=approach,
            node_order=node_list,
            num_workers=num_workers,
            chunk_size=chunk_size,
            plot_distributions=make_plots,
        ),
    }
    return dist_matrix


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
    dist_full_graph = dijkstra(
        graph_matrix, directed=True, return_predecessors=False, unweighted=False
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

    return dist_full_graph


def calculate_euclidean_distance_matrix_projected(
    graph, node_order=None, plot_distributions=False
):
    """Calculate the euclidean distances between all nodes in the graph.

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
        If True, plot the distributions of the euclidean distances and coordinates.
        Sanity check for the coordinate values.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the euclidean
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

    # Calculate the euclidean distances between all nodes
    dist_matrix = np.sqrt(
        np.square(x_coord[:, np.newaxis] - x_coord[np.newaxis, :])
        + np.square(y_coord[:, np.newaxis] - y_coord[np.newaxis, :])
    )
    # Convert to half-precision to save memory
    dist_matrix = dist_matrix.astype(np.half)

    if plot_distributions:
        plot_distance_distributions(
            dist_matrix,
            dist_title="Distribution of euclidean distances",
            coords=(x_coord, y_coord),
            coord_title="Scatter plot of projected coordinates",
            labels=("x", "y"),
        )

    return dist_matrix


def calculate_euclidean_distance_matrix_haversine(
    graph, node_order=None, plot_distributions=False
):
    """Calculate the euclidean distances between all nodes in the graph.

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
        If True, plot the distributions of the euclidean distances and coordinates.
        Sanity check for the coordinate values.

    Returns
    -------
    dist_matrix : ndarray
        The distance matrix for the partitioning. dist_matrix[i, j] is the euclidean
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

    # Calculate the euclidean distances between all nodes
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
            dist_title="Distribution of euclidean distances",
            coords=(node1_lon, node1_lat),
            coord_title="Scatter plot of lat/lon",
            labels=("Longitude", "Latitude"),
        )

    return dist_matrix


def calculate_partitioning_distance_matrix(
    partitioner,
    weight=None,
    approach="rep_nodes",
    node_order=None,
    num_workers=None,
    chunk_size=1,
    plot_distributions=False,
    check_overlap=True,
):  # pylint: disable=too-many-locals
    """Calculate the distance matrix for the partitioning.

    This is the pairwise distance between all pairs of nodes/representative nodes, where
    the shortest paths are only allowed to traverse edges in the start and goal
    partitions and unpartitioned edges. The whole sparsified graph is always computed.
    For this, for each combination of start and goal partitions, the shortest
    paths are calculated using `calculate_path_distance_matrix()`, as well as for the
    unpartitioned edges.
    Finally, a big distance matrix is constructed, where the distances for the
    edges in the start and goal partitions are taken from the distance matrix for
    the corresponding partition, and the distances for the unpartitioned edges are
    taken from the distance matrix for the unpartitioned edges.

    Parameters
    ----------
    partitioner : BasePartitioner
        The partitioner to calculate the distance matrix for
    weight : str, optional
        The edge attribute to use as weight. If None, all edges have weight 1.
    approach : str, optional
        The approach to use for the distance matrix calculation. Can be one of
        "full" and "rep_nodes". "full" calculates the distance matrix for all nodes
        in the graph. "rep_nodes" calculates the distance matrix for the representative
        nodes of each partition. The whole sparsified graph is always computed.
    node_order : list, optional
        The order of the nodes in the distance matrix. If None, the ordering is
        produced by graph.nodes() if approach is "full", or by the order of the
        representative nodes of `get_partition_nodes()` followed by
        list(partitioner.sparsified.nodes) if approach is "rep_nodes".
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
        If True, plot the distributions of the euclidean distances and coordinates.
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
        In the order
          - of `node_order`
          - produced by `graph.nodes()` if `approach` is "full"
          - of the representative nodes of `get_partition_nodes()`, followed by the
            nodes `partitioner.sparsified.nodes` if `approach` is "rep_nodes"
    """

    if num_workers is None:
        num_workers = min(32, cpu_count() // 2)
        logger.debug("No number of workers specified, using %s.", num_workers)

    partitions = partitioner.get_partition_nodes()

    # Check that none of the partitions overlap by checking that the intersection
    # of the nodes in each partition is empty.
    if check_overlap:
        pairwise_overlap = has_pairwise_overlap(
            [list(part["nodes"]) for part in partitions]
        )
        # Check if any element in the pairwise overlap matrix is True, except the
        # diagonal
        if np.any(pairwise_overlap[np.triu_indices_from(pairwise_overlap, k=1)]):
            raise ValueError("The partitions overlap node-wise. This is not allowed.")

    if node_order is None and approach == "full":
        node_order = list(partitioner.get_sorted_node_list())
    elif node_order is None and approach == "rep_nodes":
        node_order = [part["rep_node"] for part in partitions] + list(
            partitioner.sparsified.nodes
        )

    # Preparing the combinations for processing. Generator of tuples of the form:
    # (name, sparse_matrix, node_id_order, from_indices, to_indices)
    # name: name of the processing combination, not used in computation
    # sparse_matrix: the sparse matrix to calculate the distances for
    # node_id_order: the order of the nodes in the sparse matrix which should be
    #                fully calculated
    # from_indices: the indices of the dijkstra result to save the distances from
    # to_indices: the indices of the distance matrix to save the distances to
    #             (the indices are the indices in the node_id_order)
    # This is done so not every combination also calculates the distances for the
    # unpartitioned edges, but only the ones that need it.

    logger.debug("Preparing combinations for processing.")
    start_time = time()

    # Start <> Goal
    combs = (
        (
            _comb_differing_partitions(
                partitioner, weight, approach, node_order, start, goal
            )
        )
        for (start, goal) in combinations(partitions, 2)
    )

    # Start == Goal + Sparsified (unpartitioned edges)
    combs = chain(
        combs,
        (
            _comb_one_partition(partitioner, weight, approach, node_order, part)
            for part in partitions
        ),
    )

    # Only Sparsified (unpartitioned edges)
    combs = chain(combs, _comb_unpartitioned(partitioner, weight, node_order))

    # Calculate the combinations in parallel
    # We expect comb to be a generator of length binom(n, 2) + n + 1 = (n^2 + n) / 2
    # + 1
    logger.debug(
        "Calculating distance matrices for %s partitions, %d combinations, "
        "with %d workers and chunk-size %d.",
        len(partitions),
        (len(partitions) / 2 + 1 / 2) * len(partitions) + 1,
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
                total=(len(partitions) / 2 + 1 / 2) * len(partitions) + 1,
                unit_scale=1,
            )
        )
    # results = []
    # for comb in combs:
    #     results.append(dijkstra_param(comb))

    # Construct the distance matrix for the partitioning, half-precision float
    dist_matrix = np.full((len(node_order), len(node_order)), np.inf, dtype=np.half)
    for part_combo_dist_matrix, from_indices, to_indices in results:
        # Fill the distance matrix with the distances for the nodes in this pair
        for from_index, to_index in zip(from_indices, to_indices):
            dist_matrix[to_index] = part_combo_dist_matrix[from_index]

    logger.debug(
        "Calculated distance matrices for all combinations of partitions in %s "
        "seconds.",
        time() - start_time,
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
            distance_unit="hops"
            if weight is None
            else "km"
            if weight == "length"
            else weight,
        )

    return dist_matrix


def dijkstra_param(comb):
    """Wrapper for the dijkstra function.

    Fixes keyword arguments for the dijkstra function.
    """
    _, sparse_matrix, node_id_order, from_indices, to_indices = comb

    return (
        dijkstra(
            sparse_matrix,
            directed=True,
            indices=node_id_order,
            return_predecessors=False,
            unweighted=False,
        ),
        from_indices,
        to_indices,
    )


def _comb_differing_partitions(partitioner, weight, approach, node_order, start, goal):
    """Combination of arguments for the dijkstra function for differing partitions."""
    if approach == "full":
        start_idx = [node_order.index(n) for n in start["nodes"]]
        goal_idx = [node_order.index(n) for n in goal["nodes"]]
    nodelist = list(start["nodes"]) + list(goal["nodes"])
    nodelist += [n for n in partitioner.sparsified.nodes if n not in nodelist]
    return (
        f"{start['name']}<>{goal['name']}",
        to_scipy_sparse_array(
            partitioner.graph,
            weight=weight,
            format="csr",
            nodelist=nodelist,  # nodes allowed to traverse
        ),
        np.arange(len(start["nodes"]) + len(goal["nodes"]))  # numerating all
        if approach == "full"  # indices of rep_nodes in nodelist
        else np.array(
            [nodelist.index(start["rep_node"]), nodelist.index(goal["rep_node"])]
        ),
        [
            np.ix_(
                range(len(start["nodes"])),
                range(
                    len(start["nodes"]),
                    len(start["nodes"]) + len(goal["nodes"]),
                ),
            ),
            np.ix_(
                range(
                    len(start["nodes"]),
                    len(start["nodes"]) + len(goal["nodes"]),
                ),
                range(len(start["nodes"])),
            ),
        ]
        if approach == "full"
        else [
            (np.array([[0]]), np.array([[1]])),  # np.ix_(range(1), range(1, 2)),
            (np.array([[1]]), np.array([[0]])),  # np.ix_(range(1, 2), range(1))
        ],
        [np.ix_(start_idx, goal_idx), np.ix_(goal_idx, start_idx)]
        if approach == "full"
        else [
            np.ix_(
                [node_order.index(start["rep_node"])],
                [node_order.index(goal["rep_node"])],
            ),
            np.ix_(
                [node_order.index(goal["rep_node"])],
                [node_order.index(start["rep_node"])],
            ),
        ],
    )


def _comb_one_partition(partitioner, weight, approach, node_order, part):
    """Combination of arguments for the dijkstra function for one partition."""
    nodelist = list(part["nodes"]) + [
        n for n in partitioner.sparsified.nodes if n not in part["nodes"]
    ]
    len_all = (len(part["nodes"]) if approach == "full" else 1) + len(
        partitioner.sparsified.nodes
    )
    to_indices = [
        node_order.index(n)
        for n in (
            (list(part["nodes"]) if approach == "full" else [part["rep_node"]])
            + list(partitioner.sparsified.nodes)
        )
    ]
    return (
        f"{part['name']}+Sparsified",
        to_scipy_sparse_array(
            partitioner.graph,
            weight=weight,
            format="csr",
            nodelist=nodelist,
        ),
        np.arange(len(part["nodes"]) + len(partitioner.sparsified.nodes))
        if approach == "full"
        else np.append(
            nodelist.index(part["rep_node"]),
            np.arange(
                len(part["nodes"]),
                len(part["nodes"]) + len(partitioner.sparsified.nodes),
            ),
        ),
        # Index with all items, don't need to split
        [np.ix_(range(len_all), range(len_all))],
        [np.ix_(to_indices, to_indices)],
    )


def _comb_unpartitioned(partitioner, weight, node_order):
    """Combination of arguments for the dijkstra function for sparsified graph."""
    return (
        (
            "unp",
            to_scipy_sparse_array(
                partitioner.graph,
                weight=weight,
                format="csr",
                nodelist=list(partitioner.sparsified.nodes),
            ),
            np.arange(len(partitioner.sparsified.nodes)),
            # Index with all items, don't need to split
            [
                np.ix_(
                    range(len(partitioner.sparsified.nodes)),
                    range(len(partitioner.sparsified.nodes)),
                )
            ],
            [
                np.ix_(
                    [node_order.index(n) for n in partitioner.sparsified.nodes],
                    [node_order.index(n) for n in partitioner.sparsified.nodes],
                ),
            ],
        ),
    )
