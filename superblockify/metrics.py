"""Metric object for the superblockify package."""
import logging
from datetime import timedelta
from time import time

import numpy as np
from networkx import to_scipy_sparse_array
from osmnx.projection import is_projected
from scipy.sparse.csgraph import dijkstra

from superblockify.plot import plot_distance_distributions

logger = logging.getLogger("superblockify")

_AVG_EARTH_RADIUS_M = 6.3781e6  # in meters, arXiv:1510.07674 [astro-ph.SR]


class Metric:
    """Metric object to be used with partitioners.

    A metric object is used to calculate the quality of a partitioning.
    It holds the information on several network metrics, which can be read,
    and can be used to calculate them when passing a Partitioner object.

    There are different network measures
    - d_E(i, j): Euclidean
    - d_S(i, j): Shortest path on full graph
    - d_N(i, j): Shortest path with ban through LTNs

    Attributes
    ----------
    coverage : float
        The coverage of the partitioning takes of the whole graph
    num_components : int
        The number of components in the graph
    avg_path_length : dict
        The average path length of the graph for each network measure
        {"E": float, "S": float, "N": float}
    directness : dict
        The directness of the graph for the network measure ratios
        {"ES": float, "EN": float, "SN": float}
    global_efficiency : dict
        The global efficiency of the graph for each network measure
        {"SE": float, "NE": float, "NS": float}
    local_efficiency : dict
        The local efficiency of the graph for each network measure
        {"SE": float, "NE": float, "NS": float}

    """

    def __init__(self):
        """Construct a metric object."""

        self.coverage = None
        self.num_components = None
        self.avg_path_length = {"E": None, "S": None, "N": None}
        self.directness = {"ES": None, "EN": None, "SN": None}
        self.global_efficiency = {"SE": None, "NE": None, "NS": None}
        self.local_efficiency = {"SE": None, "NE": None, "NS": None}

        self.distance_matrix = None

    def __str__(self):
        """Return a string representation of the metric object.

        Only returns the attributes that are not None or for a dict the
        attributes that are not None for each key. If all attributes in a dict are None,
        it is not returned.
        If no attributes are not None, an empty string is returned.
        """
        string = ""
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, dict):
                    if all(v is None for v in value.values()):
                        continue
                    string += f"{key}: "
                    for key2, value2 in value.items():
                        if value2 is not None:
                            string += f"{key2}: {value2}, "
                    string = string[:-2] + "; "
                else:
                    string += f"{key}: {value}; "
        return string

    def __repr__(self):
        """Return a string representation of the metric object.

        Additional to the __str__ method, it also returns the class name.
        """
        return f"{self.__class__.__name__}({self.__str__()})"

    def calculate_all(self, partitioner):
        """Calculate all metrics for the partitioning.

        `self.distance_matrix` is used to save the distances for the metrics and should
        is set to None after calculating the metrics.

        Parameters
        ----------
        partitioner : BasePartitioner
            The partitioner object to calculate the metrics for

        """

        # Get node list for fixed order
        node_list = list(partitioner.graph.nodes())

        # Euclidean distances (E)
        dist_euclidean = self.calculate_euclidean_distance_matrix_projected(
            partitioner.graph, node_order=node_list
        )

        # On the full graph (S)
        dist_full_graph = self.calculate_distance_matrix(
            partitioner.graph, weight="length", node_order=node_list
        )

        # On the partitioning graph (N)
        dist_partitioning_graph = self.calculate_partitioning_distance_matrix(
            partitioner, weight="length", node_order=node_list
        )

        self.distance_matrix = {
            "E": dist_euclidean,
            "S": dist_full_graph,
            "N": dist_partitioning_graph,
        }

    def calculate_distance_matrix(
        self, graph, weight=None, node_order=None, plot_distributions=False
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
                distance_unit="hops"
                if weight is None
                else "km"
                if weight == "length"
                else weight,
            )

        return dist_full_graph

    def calculate_euclidean_distance_matrix_projected(
        self, graph, node_order=None, plot_distributions=False
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
        self, graph, node_order=None, plot_distributions=False
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
        self, partitioner, weight=None, node_order=None, plot_distributions=False
    ):
        """Calculate the distance matrix for the partitioning.

        This is the pairwise distance between all pairs of nodes, where the shortest
        paths are only allowed to traverse edges in the start and goal partitions and
        unpartitioned edges.
        For this, for each combination of start and goal partitions, the shortest
        paths are calculated using `calculate_distance_matrix()`, as well as for the
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
        node_order : list, optional
            The order of the nodes in the distance matrix. If None, the ordering is
            produced by graph.nodes().
        plot_distributions : bool, optional
            If True, plot the distributions of the euclidean distances and coordinates.

        Returns
        -------
        dist_matrix : ndarray
            The distance matrix for the partitioning. dist_matrix[i, j] is the distance
            between node i and node j for the given rules of the partitioning.

        """
