"""Metric object for the superblockify package."""
import logging
from datetime import timedelta
from time import time

from networkx import floyd_warshall_numpy

logger = logging.getLogger("superblockify")


class Metric:
    """Metric object to beused with partitioners.

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
        save_distances : bool, optional
            Whether to save the distances for the metrics, by default True
            If there are too many nodes, this can take a lot of memory, but saves
            time when calculating the distances for the metrics again.

        """

        # Calculate all-pairs shortest path lengths with the Floyd-Warshall algorithm
        # On the full graph (S)
        start_time = time()
        floyd_warshall = floyd_warshall_numpy(partitioner.graph, weight="length")
        logger.debug(
            "Calculated all-pairs shortest path lengths with the Floyd-Warshall "
            "algorithm on the full graph (S) in %s",
            timedelta(seconds=time() - start_time),
        )
        # On the partitioning graph (N)

        self.distance_matrix = {"S": floyd_warshall}
