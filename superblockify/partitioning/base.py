"""BasePartitioner parent and dummy."""
import logging
import pickle
from abc import ABC, abstractmethod
from configparser import ConfigParser
from os import makedirs
from os.path import join, exists, dirname
from typing import final, Final, List

import osmnx as ox
from matplotlib import pyplot as plt
from networkx import (
    weakly_connected_components,
    set_edge_attributes,
    MultiDiGraph,
)
from osmnx.stats import edge_length_total

from .checks import is_valid_partitioning
from .plot import (
    plot_partition_graph,
    plot_subgraph_component_size,
    plot_component_rank_size,
    plot_component_graph,
)
from .representative import set_representative_nodes
from .speed import add_edge_travel_times_restricted
from .utils import (
    show_highway_stats,
    remove_dead_ends_directed,
    split_up_isolated_edges_directed,
)
from .. import attribute
from ..metrics.metric import Metric
from ..plot import save_plot
from ..utils import load_graph_from_place

logger = logging.getLogger("superblockify")

config = ConfigParser()
config.read(join(dirname(__file__), "..", "..", "config.ini"))
GRAPH_DIR = config["general"]["graph_dir"]
RESULTS_DIR = config["general"]["results_dir"]


class BasePartitioner(ABC):
    """Parent class for partitioning graphs.

    Notes
    -----
    This class is an abstract base class and should not be instantiated directly.

    Examples
    --------
    >>> from superblockify.partitioning import BasePartitioner
    >>> import osmnx as ox
    >>> name, search_str = "Resistencia", "Resistencia, Chaco, Argentina"
    >>> graph = ox.graph_from_place(search_str, network_type="drive")
    >>> part = BasePartitioner(graph=graph, name=name)
    >>> part.run(make_plots=True)

    >>> from superblockify.partitioning import BasePartitioner
    >>> import osmnx as ox
    >>> part = BasePartitioner(
    ...     name="Resistencia", search_str="Resistencia, Chaco, Argentina"
    ... )
    >>> part.run(approach=True, make_plots=True, num_workers=6)

    """

    # pylint: disable=too-many-instance-attributes, too-many-lines

    def __init__(
        self,
        name="unnamed",
        city_name=None,
        search_str=None,
        graph=None,
        reload_graph=False,
    ):
        """Constructing a BasePartitioner

        Parameters
        ----------
        name : str, optional
            Name of the graph's city. Will be used as folder name for results and
            plot titles. Default is "unnamed".
        city_name : str
            Name of the city, for looking up the graph. If graph is not found,
            it will be downloaded from OSMnx using the `search_str`. Default is None.
            Used so multiple `Partitioner` don't need to download the same city graph.
        search_str : str or list of str, optional
            Search string for OSMnx to download a graph, default is None. Only used if
            no graph is found under GRAPH_DIR/city_name.graphml.
            For composite cities, a list of search strings can be provided.
        graph : networkx.MultiDiGraph, optional
            Graph to partition. Used if no graph is found under GRAPH_DIR/name.graphml
            and `search_str` is None. Default is None.
        reload_graph : bool, optional
            If True, reload the graph from OSMnx, even if a graph with the name
            `name.graphml` is found in the working directory. Default is False.

        Raises
        ------
        ValueError
            If neither graph nor search_str are provided.
        ValueError
            If name is not a string or empty.
        ValueError
            If city_name is not a string or empty.
        KeyError
            If search_str is given, it must be a string or list of non-empty strings.

        Notes
        -----
        GRAPH_DIR is set in the `config.ini` file.

        """

        if not isinstance(name, str) or name == "":
            raise ValueError("Name must be a non-empty string.")
        if not isinstance(city_name, str) or city_name == "":
            raise ValueError("City name must be a non-empty string.")
        if not (
            search_str is None
            or (isinstance(search_str, str) and search_str != "")
            or (
                isinstance(search_str, list)
                and all(isinstance(s, str) and s != "" for s in search_str)
                and search_str != []
            )
        ):
            raise KeyError(
                "Search_str must be a non-empty string or a list of non-empty strings."
            )

        # First check weather a graph is found under GRAPH_DIR/city_name.graphml
        graph_path = join(GRAPH_DIR, f"{city_name}.graphml")
        if exists(graph_path):
            self.graph = self.load_or_find_graph(city_name, search_str, reload_graph)
        elif search_str is not None:
            self.graph = self.load_or_find_graph(city_name, search_str)
        elif graph is not None:
            self.graph = graph
        else:
            raise ValueError("Either graph or search_str must be provided.")

        remove_dead_ends_directed(self.graph)
        show_highway_stats(self.graph)

        # Create results directory
        self.results_dir = join(RESULTS_DIR, name)
        if not exists(self.results_dir):
            makedirs(self.results_dir)

        # Set Instance variables
        self.name: Final[str] = name
        self.city_name: str = city_name
        self.partitions: List[dict] | None = None
        self.components: List[dict] | None = None
        self.sparsified = None
        self.attribute_label: str | None = None
        self.attr_value_minmax: tuple | None = None
        self.metric = Metric()

        # Log initialization
        logger.info(
            "Initialized %s(%s) with %d nodes and %d edges.",
            self.name,
            self.__class__.__name__,
            len(self.graph.nodes),
            len(self.graph.edges),
        )

    @final
    def run(self, calculate_metrics=True, make_plots=False, **kwargs):
        """Run partitioning.

        Parameters
        ----------
        calculate_metrics : str or bool, optional
            If True, calculate metrics for the partitioning. If False, don't.
            Save them to self.results_dir/metrics. Default is True.
        make_plots : bool, optional
            If True, make plots of the partitioning and save them to
            self.results_dir/figures. Default is False.
        """

        self.partition_graph(make_plots=make_plots, **kwargs)

        # If partitioner has only set the partitions, but not the sparsified graph,
        # make it from the edges not in the partitions
        if self.sparsified is None:
            logger.debug(
                "Sparsified graph not set. "
                "Making it from the edges not in the partitions."
            )
            self.set_sparsified_from_components()

        # Set representative nodes
        set_representative_nodes(
            self.components if self.components else self.partitions
        )

        # Check that the partitions and sparsified graph satisfy the requirements
        is_valid_partitioning(self)

        if make_plots:
            if self.partitions:
                fig, _ = plot_partition_graph(self)
                save_plot(self.results_dir, fig, f"{self.name}_partition_graph.pdf")
                plt.show()
            fig, _ = plot_subgraph_component_size(self, "length")
            save_plot(self.results_dir, fig, f"{self.name}_subgraph_component_size.pdf")
            plt.show()
            fig, _ = plot_component_rank_size(self, "length")
            save_plot(self.results_dir, fig, f"{self.name}_component_rank_size.pdf")
            plt.show()
            if self.components:
                fig, _ = plot_component_graph(self)
                save_plot(self.results_dir, fig, f"{self.name}_component_graph.pdf")
                plt.show()

        if calculate_metrics:
            self.calculate_metrics(make_plots=make_plots, **kwargs)

    @abstractmethod
    def partition_graph(self, make_plots=False, **kwargs):
        """Partition the graph.

        Parameters
        ----------
        make_plots : bool, optional
            If True, make plots of the partitioning and save them to
            self.results_dir/figures. Default is False.

        Notes
        -----
        This method should be implemented by the child class, kwargs are handed down
        from the `run` method.
        """

        self.attribute_label = "example_label"
        # Define partitions
        self.partitions = [
            {"name": "zero", "value": 0.0},
            {"name": "one", "value": 1.0},
        ]

    def calculate_metrics(self, make_plots=False, num_workers=None, chunk_size=1):
        """Calculate metrics for the partitioning.

        Calculates the metrics for the partitioning and writes them to the
        metrics dictionary. It includes the network metrics for the partitioned graph.

        There are different network measures
        - d_E(i, j): Euclidean
        - d_S(i, j): Shortest path on full graph
        - d_N(i, j): Shortest path with ban through LTNs

        Parameters
        ----------
        make_plots : bool, optional
            If True show visualization graphs of the approach. If False only print
            into console. Default is False.
        num_workers : int, optional
            Number of workers to use for parallel processing. Default is None, which
            uses min(32, os.cpu_count() + 4) workers.
        chunk_size : int, optional
            Size of chunks to split the graph into for parallel processing. Default is
            1, which means no chunking. A chunking over 3 seems to not be beneficial.

        """

        # Log calculating metrics
        logger.info("Calculating metrics for %s", self.name)
        self.metric.calculate_all(
            partitioner=self,
            make_plots=make_plots,
            num_workers=num_workers,
            chunk_size=chunk_size,
        )

        logger.debug("Metrics for %s: %s", self.name, self.metric)

    def make_subgraphs_from_attribute(
        self, split_disconnected=False, min_edge_count=0, min_length=0
    ):
        """Make component subgraphs from attribute.

        Method for child classes to make subgraphs from the attribute
        `self.attribute_label`, to analyze (dis-)connected components.
        For each partition makes a subgraph with the edges that have the
        attribute value of the partition.
        Writes them to `self.component[i]["subgraph"]` with the name of the
        partition+`_component_`+`j`. Where `j` is the index of the component.

        Parameters
        ----------
        split_disconnected : bool, optional
            If True, split the disconnected components into separate subgraphs.
            Default is False.
        min_edge_count : int, optional
            If split_disconnected is True, minimal size of a component to be
            considered as a separate subgraph. Default is 0.
        min_length : int, optional
            If split_disconnected is True, minimal length (in meters) of a component to
            be considered as a separate subgraph. Default is 0.

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).

        """

        self.check_has_been_run()

        # Log making subgraphs
        logger.info(
            "Making subgraphs for %s with attribute %s",
            self.name,
            self.attribute_label,
        )

        found_disconnected = False
        num_partitions = len(self.partitions)

        # Make component subgraphs from attribute
        for part in self.partitions:
            logger.debug("Making subgraph for partitions %s", part)
            part["subgraph"] = attribute.get_edge_subgraph_with_attribute_value(
                self.graph, self.attribute_label, part["value"]
            )
            part["num_edges"] = len(part["subgraph"].edges)
            part["num_nodes"] = len(part["subgraph"].nodes)
            part["length_total"] = sum(
                d["length"] for u, v, d in part["subgraph"].edges(data=True)
            )

        if split_disconnected:
            self.components = []

        for part in self.partitions:
            # Split disconnected components
            connected_components = weakly_connected_components(part["subgraph"])
            # Make list of generator of connected components
            connected_components = list(connected_components)
            logger.debug(
                "Partition %s has %d conn. comp. In total %d nodes and %d edges.",
                part["name"],
                len(list(connected_components)),
                len(part["subgraph"].nodes),
                len(part["subgraph"].edges),
            )
            if split_disconnected:
                found_disconnected = True
                # Add partitions for each connected component
                for i, component in enumerate(connected_components):
                    attribute_edge_subgraph = self.graph.edge_subgraph(
                        (u, v, k)
                        for u, v, k, d in self.graph.subgraph(component).edges(
                            data=True, keys=True
                        )
                        if d.get(self.attribute_label) == part["value"]
                    )
                    self.components.append(
                        {
                            "name": f"{part['name']}_component_{i}",
                            "value": part["value"],
                            "subgraph": attribute_edge_subgraph,
                            "num_edges": len(attribute_edge_subgraph.edges),
                            "num_nodes": len(attribute_edge_subgraph.nodes),
                            "length_total": sum(
                                d["length"]
                                for u, v, d in attribute_edge_subgraph.edges(data=True)
                            ),
                        }
                    )
                    # Add 'ignore' attribute, based on min_edge_count and min_length
                    self.components[-1]["ignore"] = (
                        self.components[-1]["num_edges"] < min_edge_count
                        or self.components[-1]["length_total"] < min_length
                    )

        # Log status about disconnected components
        found_disconnected = (
            f"Found disconnected components in %s, splitting them. "
            f"There are {num_partitions} partitions, "
            f"and {len(self.partitions)} components. "
            f"Thereof are {len([c for c in self.components if not c['ignore']])} "
            f"components with more than {min_edge_count} edges and "
            f"more than {min_length} meters."
            if found_disconnected
            else "No disconnected components found in %s, nothing to split."
        )
        logger.debug(found_disconnected, self.name)

        if split_disconnected:
            self.overwrite_attributes_of_ignored_components(
                attribute_name=self.attribute_label, attribute_value=None
            )

    def set_components_from_sparsified(self):
        """Set components from sparsified graph.

        Method for child classes to set the components from the sparsified graph.
        The components are the connected components of the rest graph without the
        sparsified subgraph.
        The components are set to `self.components`, also overwriting the
        `self.partitions` attribute.
        """

        split_up_isolated_edges_directed(self.graph, self.sparsified)

        # Find difference, edgewise, between the graph and the sparsified subgraph
        rest = self.graph.edge_subgraph(
            [
                (u, v, k)
                for u, v, k in self.graph.edges(keys=True, data=False)
                if (u, v, k) not in self.sparsified.edges(keys=True)
            ]
        ).copy()

        rest.remove_nodes_from(self.sparsified.nodes())
        logger.debug(
            "Making components from sparsified graph, "
            "rest graph has %d nodes and %d edges.",
            len(rest.nodes),
            len(rest.edges),
        )

        # Find weakly connected components on the rest graph with split nodes
        wc_components = list(weakly_connected_components(rest))

        self.attr_value_minmax = (0, len(wc_components))
        self.partitions = []
        undirected = self.graph.to_undirected()
        for i, component in enumerate(wc_components):
            # Find edges that are connected to the component nodes, on undirected graph
            edges = undirected.edges(component, keys=True)
            if not edges:
                logger.debug("Skipping empty component %d", i)
                continue
            subgraph = self.graph.edge_subgraph(
                # forward edges + backward edges - span directed graph from undirected
                list(edges)
                + [(v, u, k) for u, v, k in edges]
            )
            self.partitions.append(
                {
                    "name": f"residential_{i}",
                    "value": i,
                    "subgraph": subgraph,
                    "num_edges": subgraph.number_of_edges(),
                    "num_nodes": subgraph.number_of_nodes(),
                    "length_total": edge_length_total(subgraph),
                }
            )

        self.components = self.partitions
        for component in self.components:
            component["ignore"] = False

    def set_sparsified_from_components(self):
        """Set sparsified graph from components.

        Method for child classes to set the sparsified graph from the components.
        The sparsified graph is the graph with all edges that are not in the
        components. The sparsified graph is set to `self.sparsified`.
        """
        # list of edges in partitions, use components if not None, else partitions
        edges_in_partitions = {
            edge
            for component in (self.components if self.components else self.partitions)
            for edge in component["subgraph"].edges(keys=True)
        }
        self.sparsified = self.graph.edge_subgraph(
            # set of all edges - set of edges in partitions
            set(list(self.graph.edges(keys=True)))
            - edges_in_partitions
        )

    def overwrite_attributes_of_ignored_components(
        self, attribute_name, attribute_value=None
    ):
        """Overwrite attributes of ignored components.

        Method for child classes to overwrite the subgraph's edge attributes
        of ignored components. Overwrites the attribute `attribute_name` with
        `attribute_value` for all components that have the attribute `ignore` set to
        True.
        This is useful for example to overwrite the `self.attribute_label` attribute
        with `None` to make the subgraph invisible in the network plot
        (`self.plot_partition_graph()`).

        Also it will affect `self.graph`, as the component's subgraph is a view of the
        original graph.

        Parameters
        ----------
        attribute_name : str
            Name of the attribute to overwrite.
        attribute_value : str, optional
            Value to overwrite the attribute with. Default is None.

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).
        AssertionError
            If `self.components` is not defined (the subgraphs have not been split
            into components).

        """

        self.check_has_been_run()

        if self.components is None:
            raise AssertionError(
                f"Components have not been defined for {self.name}. "
                f"Run `make_subgraphs_from_attribute` with `split_disconnected` "
                f"set to True."
            )

        # Log overwriting attributes
        logger.info(
            "Overwriting attributes of ignored components for attribute %s "
            "with value %s",
            attribute_name,
            attribute_value,
        )

        # Overwrite attributes of ignored components
        if self.components:
            for component in self.components:
                if component["ignore"]:
                    set_edge_attributes(
                        component["subgraph"], attribute_value, attribute_name
                    )

    def get_partition_nodes(self):
        """Get the nodes of the partitioned graph.

        Returns list of dict with name of partition and list of nodes in partition.
        If partitions were split up into components with `make_subgraphs_from_attribute`
        with `split_disconnected` set to True, the nodes of the components are returned.

        Per default, nodes are considered to be inside a partition if they are in the
        subgraph of the partition and not in the sparsified subgraph. Also, `ignored`
        components are left out.

        Nodes inside the sparsified graph are not considered to be inside a partition.

        Method can be overwritten by child classes to change the definition of
        which nodes are considered to be inside a partition.

        Returns
        -------
        list of dict
            List of dict with `name` of partition, `subgraph` of partition and set of
            `nodes` in partition.

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).

        """

        self.check_has_been_run()

        # List of partitions /unignored components
        # Only take `name`, `subgraph` and `representative_node_id` from the components
        partitions = [
            {
                "name": part["name"],
                "subgraph": part["subgraph"],
                "rep_node": part["representative_node_id"],
            }
            for part in (self.components if self.components else self.partitions)
            # if there is the key "ignore" and it is True, ignore the partition
            if not (part.get("ignore") is True)
        ]

        # Add list of nodes "inside" each partitions
        #  - nodes that are not shared with the sparsified graph are considered to be
        #    inside the partition
        #  - from these the distances are calculated
        #  - the nodes not in any partitions are considered as the unpartitioned nodes
        for part in partitions:
            part["nodes"] = {
                node
                for node in part["subgraph"].nodes()
                if part["subgraph"].degree(node) >= 1
                and node not in self.sparsified.nodes
            }

        return partitions

    def get_sorted_node_list(self):
        """Get sorted list of nodes.

        Sorted after the amount of nodes in the partition, followed by the
        unpartitioned nodes.
        Uses `get_partition_nodes` to return a list of nodes.

        Returns
        -------
        list of nodes
            List of nodes sorted after the name of the partition, followed by the
            unpartitioned nodes.
        """

        # Get node list for fixed order - sorted by partition name
        node_list = self.get_partition_nodes()
        # node_list is list of dicts, which all have a "name" and "nodes" key

        # Make one long list out of all the nodes,
        # sorted by number of nodes in "subgraph"
        # node_list = sorted(node_list, key=lambda x: x["name"])
        node_list = sorted(node_list, key=lambda x: len(x["nodes"]), reverse=True)
        node_list = [node for partition in node_list for node in partition["nodes"]]
        # Throw out duplicates, started from the back
        node_list = list(dict.fromkeys(node_list[::-1]))[::-1]  # Should not change
        # anything if requirements are met
        # Add nodes that are not in a partition, only the key of nodes is needed
        node_list += [node for node in self.graph.nodes if node not in node_list]

        return node_list

    def check_has_been_run(self):
        """Check if the partitioner has ran.

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).

        """

        if self.partitions is None:
            raise AssertionError(
                f"{self.__class__.__name__} has no partitions, "
                f"run before plotting graph."
            )
        if self.attribute_label is None:
            raise AssertionError(
                f"{self.__class__.__name__} has no `attribute_label` yet, "
                f"run before plotting graph."
            )

    def load_or_find_graph(self, city_name, search_str, reload_graph=False):
        """Load or find graph if it exists.

        If graph GRAPH_DIR/name.graphml exists, load it. Else, find it using
        `search_str` and save it to GRAPH_DIR/name.graphml.

        Parameters
        ----------
        city_name : str
            Name of the graph. Can be the name of the place and also be descriptive.
            Not to confuse with the name of an instance of the class.
        search_str : str or list of str
            String to search for in OSM. Can be a list of strings to combine multiple
            search terms. Use nominatim to find the right search string.
        reload_graph : bool, optional
            If True, reload the graph even if it already exists.

        Returns
        -------
        graph : networkx.MultiDiGraph
            Graph.

        Notes
        -----
        GRAPH_DIR is set in the `config.ini` file.
        """

        # Check if graph already exists
        graph_path = join(GRAPH_DIR, city_name + ".graphml")
        if exists(graph_path) and not reload_graph:
            logger.debug("Loading graph from %s", graph_path)
            graph = ox.load_graphml(graph_path)
        else:
            logger.debug("Finding graph with search string %s", search_str)
            graph = load_graph_from_place(
                save_as=graph_path,
                search_string=search_str,
                custom_filter=config["graph"]["network_filter"],
                simplify=True,
            )
            logger.debug("Saving graph to %s", graph_path)
            ox.save_graphml(graph, graph_path)
        return graph

    # IO methods
    def save(self, save_graph_copy=False):
        """Save the partitioner.

        Pickle the partitioner and save it to file. Metric object will be saved in
        separate file.

        Parameters
        ----------
        save_graph_copy : bool, optional
            If True, save the graph to a file. In the case the partitioner was
            initialized with a name and/or search string, the underlying graph is
            at GRAPH_DIR/name.graphml already. Only use this if you want to save a
            copy of the graph that has been modified by the partitioner.
            This is necessary for later plotting partitions, but not for component
            plots.

        Notes
        -----
        `graph_dir` is set in the `config.ini` file.
        """

        # Save graph
        if save_graph_copy:
            graph_path = join(self.results_dir, self.name + ".graphml")
            logger.debug("Saving graph copy to %s", graph_path)
            ox.save_graphml(self.graph, filepath=graph_path)

        # Save metrics
        self.metric.save(self.results_dir, self.name)

        # Save partitioner, with self.graph = None
        partitioner_path = join(self.results_dir, self.name + ".partitioner")
        # Check if partitioner already exists
        if exists(partitioner_path):
            logger.debug("Partitioner already exists, overwriting %s", partitioner_path)
        else:
            logger.debug("Saving partitioner to %s", partitioner_path)
        with open(partitioner_path, "wb") as file:
            # Convert subgraph views to MultiDiGraphs for pickling, if they exist
            if self.partitions is not None:
                for i, partition in enumerate(self.partitions):
                    self.partitions[i]["subgraph"] = MultiDiGraph(partition["subgraph"])
            if self.components is not None:
                for i, component in enumerate(self.components):
                    self.components[i]["subgraph"] = MultiDiGraph(component["subgraph"])
            # Convert self.sparsified to MultiDiGraph for pickling
            if self.sparsified is not None:
                self.sparsified = MultiDiGraph(self.sparsified)
            # Remove graph from partitioner
            graph = self.graph
            self.graph = None
            # Remove metric from partitioner
            metric = self.metric
            self.metric = None
            pickle.dump(self, file)
            # Restore graph
            self.graph = graph
            # Restore metric
            self.metric = metric

    @classmethod
    def load(cls, name):
        """Load a partitioner.

        Parameters
        ----------
        name : str
            Name of the partitioner. This is the name of the folder in which the
            partitioner is saved. Also, the name of the graph and the metrics.

        Returns
        -------
        partitioner : BasePartitioner
            Partitioner.

        Raises
        ------
        FileNotFoundError
            If the partitioner cannot be found.

        Notes
        -----
        The directories RESULTS_DIR and GRAPH_DIR are set in the `config.ini` file.

        """

        # Load partitioner
        partitioner_path = join(RESULTS_DIR, name, name + ".partitioner")
        logger.debug("Loading partitioner from %s", partitioner_path)
        with open(partitioner_path, "rb") as file:
            partitioner = pickle.load(file)

        # Load metric
        metric_path = join(RESULTS_DIR, name, name + ".metrics")
        if exists(metric_path):
            logger.debug("Loading metric from %s", metric_path)
            partitioner.metric = Metric.load(name)
        else:
            logger.debug("Metric not found in %s, keeping empty", metric_path)

        return cls._load_graph(partitioner, name)

    @classmethod
    def _load_graph(cls, partitioner, name):
        """Load the graph of a partitioner.

        Parameters
        ----------
        partitioner : BasePartitioner
            Partitioner with graph not loaded.
        name : str
            Name of the partitioner. This is the name of the folder in which the
            partitioner is saved. Also, the name of the graph and the metrics.

        Returns
        -------
        partitioner : BasePartitioner
            Partitioner with graph loaded.
        """

        # Load graph - if possible from RESULTS_DIR, else from GRAPH_DIR
        graph_path = join(RESULTS_DIR, name, name + ".graphml")
        if exists(graph_path):
            logger.debug("Loading graph from %s", graph_path)
            partitioner.graph = ox.load_graphml(graph_path)
        else:
            graph_path = join(GRAPH_DIR, partitioner.city_name + ".graphml")
            if exists(graph_path):
                logger.debug("Loading graph from %s", graph_path)
                partitioner.graph = ox.load_graphml(graph_path)
            else:
                logger.debug("Graph not found in %s, keeping empty", graph_path)
                return partitioner
        # Only if self.graph is not None, not if it could not be loaded.
        # Graphs of self.components need to be converted to be subgraphs of self.graph.
        if partitioner.components is not None:
            for i, component in enumerate(partitioner.components):
                if "subgraph" in component:
                    partitioner.components[i][
                        "subgraph"
                    ] = partitioner.graph.edge_subgraph(component["subgraph"].edges)
        # Graphs of self.partitions need to be converted to be subgraphs of self.graph.
        if partitioner.partitions is not None:
            for i, partition in enumerate(partitioner.partitions):
                if "subgraph" in partition:
                    partitioner.partitions[i][
                        "subgraph"
                    ] = partitioner.graph.edge_subgraph(partition["subgraph"].edges)
        # Convert self.sparsified to subgraph of self.graph.
        if partitioner.sparsified is not None:
            partitioner.sparsified = partitioner.graph.edge_subgraph(
                partitioner.sparsified.edges
            )
        return partitioner
