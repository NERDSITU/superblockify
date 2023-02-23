"""Metric object for the superblockify package."""
import logging
from datetime import timedelta
from time import time

from networkx import to_scipy_sparse_array
from scipy.sparse.csgraph import dijkstra

logger = logging.getLogger("superblockify")


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
        partitioner : Partitioner
            The partitioner object to calculate the metrics for

        """

        # On the full graph (S)
        dist_full_graph = self.calculate_distance_matrix(
            partitioner.graph, weight="length", node_order=node_list
        )

        # On the partitioning graph (N)

        self.distance_matrix = {"S": dist_full_graph}

    def calculate_distance_matrix(self, graph, weight=None, node_order=None):
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
        return dist_full_graph
