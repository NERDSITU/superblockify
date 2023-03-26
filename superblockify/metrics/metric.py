"""Metric object for the superblockify package."""
import logging
import pickle
from configparser import ConfigParser
from os import path

from .distances import (
    calculate_euclidean_distance_matrix_projected,
    calculate_distance_matrix,
    calculate_partitioning_distance_matrix,
)
from .measures import (
    calculate_global_efficiency,
    calculate_directness,
    write_relative_increase_to_edges,
)
from ..metrics import (
    plot_distance_matrices,
    plot_distance_matrices_pairwise_relative_difference,
)
from ..metrics.plot import (
    plot_component_wise_travel_increase,
    plot_relative_difference,
    plot_relative_increase_on_graph,
)
from ..plot import save_plot
from ..utils import compare_dicts

logger = logging.getLogger("superblockify")

config = ConfigParser()
config.read("config.ini")
RESULTS_DIR = config["general"]["results_dir"]


class Metric:
    """Metric object to be used with partitioners.

    A metric object is used to calculate the quality of a partitioning.
    It holds the information on several network metrics, which can be read,
    and can be used to calculate them when passing a Partitioner object.

    There are different network measures
    - d_E(i, j): Euclidean
    - d_S(i, j): Shortest path on full graph
    - d_N(i, j): Shortest path with ban through LTNs

    We define several types of combinations of these metrics:
    (i, j are nodes in the graph)

    The network metrics are the following:

    - Coverage (fraction of network covered by a partition):
      C = sum(1 if i in partition else 0) / len(graph.nodes)

    - Components (number of connected components):
      C = len(graph.components)

    - Average path length:
        - A(E) = mean(d_E(i, j)) where i <> j
        - A(S) = mean(d_S(i, j)) where i <> j
        - A(N) = mean(d_N(i, j)) where i <> j

    - Directness:
        - D(E, S) = mean(d_E(i, j) / d_S(i, j)) where i <> j
        - D(E, N) = mean(d_E(i, j) / d_N(i, j)) where i <> j
        - D(S, N) = mean(d_S(i, j) / d_N(i, j)) where i <> j

    - Global efficiency:
        - G(i; S/E) = sum(1/d_S(i, j)) / sum(1/d_E(i, j)) where for each sum i <> j
        - G(i; N/E) = sum(1/d_N(i, j)) / sum(1/d_E(i, j)) where for each sum i <> j
        - G(i; N/S) = sum(1/d_N(i, j)) / sum(1/d_S(i, j)) where for each sum i <> j

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

    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        """Construct a metric object."""

        self.coverage = None
        self.num_components = None
        self.avg_path_length = {"E": None, "S": None, "N": None}
        self.directness = {"ES": None, "EN": None, "SN": None}
        self.global_efficiency = {"SE": None, "NE": None, "NS": None}

        self.distance_matrix = None
        self.weight = None
        self.node_list = None

    def calculate_all(
        self,
        partitioner,
        weight="length",
        num_workers=None,
        chunk_size=1,
        make_plots=False,
    ):
        """Calculate all metrics for the partitioning.

        `self.distance_matrix` is used to save the distances for the metrics and should
        is set to None after calculating the metrics.

        Parameters
        ----------
        partitioner : BasePartitioner
            The partitioner object to calculate the metrics for
        weight : str, optional
            The edge attribute to use as weight, by default "length", if None count hops
        num_workers : int, optional
            The number of workers to use for multiprocessing. If None, use
            min(32, os.cpu_count() + 4), by default None
        chunk_size : int, optional
            The chunk size to use for multiprocessing, by default 1
        make_plots : bool, optional
            Whether to make plots of the distributions of the distances for each
            network measure, by default False

        """
        # pylint: disable=unused-argument

        # Set weight attribute
        self.weight = weight

        self.node_list = partitioner.get_sorted_node_list()

        # Euclidean distances (E)
        dist_euclidean = calculate_euclidean_distance_matrix_projected(
            partitioner.graph,
            node_order=self.node_list,
            plot_distributions=make_plots,
        )

        # On the full graph (S)
        dist_full_graph = calculate_distance_matrix(
            partitioner.graph,
            weight="length",
            node_order=self.node_list,
            plot_distributions=make_plots,
        )

        # On the partitioning graph (N)
        dist_partitioning_graph = calculate_partitioning_distance_matrix(
            partitioner,
            weight="length",
            node_order=self.node_list,
            num_workers=num_workers,
            chunk_size=chunk_size,
            plot_distributions=make_plots,
        )

        self.distance_matrix = {
            "E": dist_euclidean,
            "S": dist_full_graph,
            "N": dist_partitioning_graph,
        }

        self.calculate_all_measure_sums()

        write_relative_increase_to_edges(
            partitioner.graph, self.distance_matrix, self.node_list, "N", "S"
        )

        if make_plots:
            fig, _ = plot_distance_matrices(
                self, name=f"{partitioner.name} - {partitioner.__class__.__name__}"
            )
            save_plot(
                partitioner.results_dir,
                fig,
                f"{partitioner.name}_distance_matrices.pdf",
            )
            fig.show()
            fig, _ = plot_distance_matrices_pairwise_relative_difference(
                self, name=f"{partitioner.name} - {partitioner.__class__.__name__}"
            )
            save_plot(
                partitioner.results_dir,
                fig,
                f"{partitioner.name}_distance_matrices_"
                f"pairwise_relative_difference.pdf",
            )
            fig.show()

            fig, _ = plot_relative_difference(
                self, "S", "N", title=f"{partitioner.name} - {self.__class__.__name__}"
            )
            save_plot(
                partitioner.results_dir,
                fig,
                f"{partitioner.name}_relative_difference_SN.pdf",
            )
            fig.show()

            fig, _ = plot_component_wise_travel_increase(
                partitioner,
                self.distance_matrix,
                self.node_list,
                measure1="S",
                measure2="N",
            )
            save_plot(
                partitioner.results_dir,
                fig,
                f"{partitioner.name}_component_wise_travel_increase.pdf",
            )

            fig, _ = plot_relative_increase_on_graph(partitioner.graph)
            save_plot(
                partitioner.results_dir,
                fig,
                f"{partitioner.name}_relative_increase_on_graph.pdf",
            )

        # self.coverage = self.calculate_coverage(partitioner)
        # logger.debug("Coverage: %s", self.coverage)

    def calculate_all_measure_sums(self):
        """Based on the distance matrix, calculate the network measures.

        Calculate the directness, global and local efficiency for each network measure
        and write them to the corresponding attributes.

        """

        # Directness
        for key in self.directness:
            self.directness[key] = calculate_directness(
                self.distance_matrix, key[0], key[1]
            )
            logger.debug("Directness %s: %s", key, self.directness[key])

        # Global efficiency
        for key in self.global_efficiency:
            self.global_efficiency[key] = calculate_global_efficiency(
                self.distance_matrix, key[0], key[1]
            )
            logger.debug("Global efficiency %s: %s", key, self.global_efficiency[key])

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

    def __eq__(self, other):
        """Return True if the two objects are equal.

        Tests the equality of the attributes of the objects.
        Used in input-output tests.
        """
        return compare_dicts(self.__dict__, other.__dict__)

    def save(self, name):
        """Save the metric to a file.

        Will be saved as a pickle file at RESULTS_DIR/name.metrics.

        Parameters
        ----------
        name : str
            The name of the file to save the metric to.

        """

        metrics_path = path.join(RESULTS_DIR, name, name + ".metrics")
        # Check if metrics already exist
        if path.exists(metrics_path):
            logger.debug("Metrics already exist, overwriting %s", metrics_path)
        else:
            logger.debug("Saving metrics to %s", metrics_path)
        with open(metrics_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, name):
        """Load a partitioning from a file.

        Parameters
        ----------
        path : str
            The path to the file to load the partitioning from.

        Returns
        -------
        partitioning : Partitioning
            The loaded partitioning.

        """

        metrics_path = path.join(RESULTS_DIR, name, name + ".metrics")
        logger.debug("Loading metrics from %s", metrics_path)
        with open(metrics_path, "rb") as file:
            metrics = pickle.load(file)

        return metrics
