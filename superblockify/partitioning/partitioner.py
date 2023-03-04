"""BasePartitioner parent and dummy."""
import logging
from abc import ABC, abstractmethod
from configparser import ConfigParser
from os import path, makedirs
from random import choice

import networkx as nx
import osmnx as ox
from numpy import linspace

from .. import attribute, plot, metrics
from ..utils import load_graph_from_place

logger = logging.getLogger("superblockify")

config = ConfigParser()
config.read("config.ini")
GRAPH_DIR = config["general"]["graph_dir"]
RESULTS_DIR = config["general"]["results_dir"]


class BasePartitioner(ABC):
    """Parent class for partitioning graphs.

    Notes
    -----
    This class is an abstract base class and should not be instantiated directly.

    Examples
    --------
    >>> from superblockify.partitioning import DummyPartitioner
    >>> import osmnx as ox
    >>> name, search_str = "Resistencia", "Resistencia, Chaco, Argentina"
    >>> graph = ox.graph_from_place(search_str, network_type="drive")
    >>> part = DummyPartitioner(graph)
    >>> part.run(make_plots=True)

    >>> from superblockify.partitioning import DummyPartitioner
    >>> import osmnx as ox
    >>> part = DummyPartitioner(
    ...     name="Resistencia", search_str="Resistencia, Chaco, Argentina"
    ... )
    >>> part.run()
    >>> part.calculate_metrics(make_plots=True, num_workers=6)

    """

    def __init__(self, graph=None, name="unnamed", search_str=None, reload_graph=False):
        """Constructing a BasePartitioner

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            Input graph
        name : str, optional
            Name of the graph's city, default is 'unnamed'.
        search_str : str or list of str, optional
            Search string for OSMnx to download a graph, default is None. Only used if
            graph is None. If there can be found a graph at
            GRAPH_DIR/name.graphml it will be loaded instead. Otherwise,
            it will be downloaded from OSMnx and saved there.
        reload_graph : bool, optional
            If True, reload the graph from OSMnx, even if a graph with the name
            `name.graphml` is found in the working directory. Default is False.

        Raises
        ------
        ValueError
            If neither graph nor search_str are provided.
        ValueError
            If name is not a string or empty.

        Notes
        -----
        GRAPH_DIR is set in the `config.ini` file.

        """

        if not isinstance(name, str) or name == "":
            raise ValueError("Name must be a non-empty string.")

        # Set Instance variables
        if graph is None:
            if search_str is None:
                raise ValueError("Either graph or search_str must be provided.")
            self.graph = self.load_or_find_graph(name, search_str, reload_graph)
        else:
            self.graph = graph
            # Make folder for graph output
            result_dir = path.join(RESULTS_DIR, name)
            if not path.exists(result_dir):
                makedirs(result_dir)

        self.name = name
        self.partitions = None
        self.components = None
        self.attribute_label = None
        self.attr_value_minmax = None
        self.metric = metrics.Metric()

        # Log initialization
        logger.info(
            "Initialized %s(%s) with %d nodes and %d edges.",
            self.name,
            self.__class__.__name__,
            len(self.graph.nodes),
            len(self.graph.edges),
        )

    @abstractmethod
    def run(self, make_plots=False, **kwargs):
        """Run partitioning.

        Parameters
        ----------
        make_plots : bool, optional
            If True, make plots of the partitioning and save them to
            RESULTS_DIR/self.name/figures. Default is False.
        """

        self.attribute_label = "example_label"
        # Define partitions
        self.partitions = [
            {"name": "zero", "value": 0.0},
            {"name": "one", "value": 1.0},
        ]

    def calculate_metrics(self, make_plots=False, num_workers=None):
        """Calculate metrics for the partitioning.

        Calculates the metrics for the partitioning and writes them to the
        metrics dictionary. It includes the network metrics for the partitioned graph.

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

        - Local efficiency:
            - L(S/E) = sum(G(i; S/E)) / N where i = 1..N
            - L(N/E) = sum(G(i; N/E)) / N where i = 1..N
            - L(N/S) = sum(G(i; N/S)) / N where i = 1..N

        Parameters
        ----------
        make_plots : bool, optional
            If True show visualization graphs of the approach. If False only print
            into console. Default is False.
        num_workers : int, optional
            Number of workers to use for parallel processing. Default is None, which
            uses min(32, os.cpu_count() + 4) workers.

        """

        # Log calculating metrics
        logger.debug("Calculating metrics for %s", self.name)
        self.metric.calculate_all(
            partitioner=self,
            make_plots=make_plots,
            num_workers=num_workers,
        )
        if make_plots:
            fig, _ = self.metric.plot_distance_matrices(
                name=f"{self.name} - {self.__class__.__name__}"
            )
            self.save_plot(fig, f"{self.name}_distance_matrices.pdf")

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

        self.__check_has_been_run()

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
            connected_components = nx.weakly_connected_components(part["subgraph"])
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
                    self.components.append(
                        {
                            "name": f"{part['name']}_component_{i}",
                            "value": part["value"],
                            "subgraph": self.graph.subgraph(component),
                            "num_edges": len(self.graph.subgraph(component).edges),
                            "num_nodes": len(self.graph.subgraph(component).nodes),
                            "length_total": sum(
                                d["length"]
                                for u, v, d in self.graph.subgraph(component).edges(
                                    data=True
                                )
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

        self.__check_has_been_run()

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
                    nx.set_edge_attributes(
                        component["subgraph"], attribute_value, attribute_name
                    )

    def get_partition_nodes(self):
        """Get the nodes of the partitioned graph.

        Returns list of dict with name of partition and list of nodes in partition.
        If partitions were split up into components with `make_subgraphs_from_attribute`
        with `split_disconnected` set to True, the nodes of the components are returned.

        Per default, nodes are considered to be inside a partition if they are in the
        subgraph of the partition and have a degree of at least 2. Also, `ignored`
        components are left out.

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

        self.__check_has_been_run()

        # List of partitions /unignored components
        # Only take `name` and `subgraph` from the components
        if self.components:
            partitions = [
                {"name": comp["name"], "subgraph": comp["subgraph"]}
                for comp in self.components
                if not comp["ignore"]
            ]
        else:
            partitions = [
                {"name": part["name"], "subgraph": part["subgraph"]}
                for part in self.partitions
            ]

        # Add list of nodes "inside" each partitions
        #  - nodes that have at least a degree of 2
        #  - from these the distances are calculated
        #  - the nodes not in any partitions are considered as the unpartitioned nodes
        for part in partitions:
            part["nodes"] = {
                node
                for node in part["subgraph"].nodes()
                if part["subgraph"].degree(node) >= 2
            }

        return partitions

    def get_sorted_node_list(self):
        """Get sorted list of nodes.

        Sorted after the name of the partition, followed by the unpartitioned nodes.
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

        # Make one long list out of all the nodes, sorted by partition name
        node_list = sorted(node_list, key=lambda x: x["name"])
        node_list = [node for partition in node_list for node in partition["nodes"]]
        # Throw out duplicates, started from the back
        node_list = list(dict.fromkeys(node_list[::-1]))[::-1]
        # Add nodes that are not in a partition, only the key of nodes is needed
        node_list += [node for node in self.graph.nodes if node not in node_list]

        return node_list

    def plot_partition_graph(self, **pba_kwargs):
        """Plotting the partitions with color on graph.

        Plots the partitioned graph, just like `plot.paint_streets` but that the
        *partitions* have a uniform color.

        Parameters
        ----------
        pba_kwargs
            Keyword arguments to pass to `superblockify.plot_by_attribute`.

        Returns
        -------
        fig, axe : tuple
            matplotlib figure, axis

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).

        """

        self.__check_has_been_run()

        # Log plotting
        logger.info(
            "Plotting partitions graph for %s with attribute %s",
            self.name,
            self.attribute_label,
        )
        return plot.plot_by_attribute(
            self.graph,
            self.attribute_label,
            minmax_val=self.attr_value_minmax,
            **pba_kwargs,
        )

    def plot_component_graph(self, **pba_kwargs):
        """Plotting the components with color on graph.

        Plots the graph with the components, just like `plot.paint_streets` but that
        the *components* have a uniform color.

        Parameters
        ----------
        pba_kwargs
            Keyword arguments to pass to `superblockify.plot_by_attribute`.

        Returns
        -------
        fig, axe : tuple
            matplotlib figure, axis

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).
        AssertionError
            If `self.components` is not defined (the subgraphs have not been split
            into components).

        """

        self.__check_has_been_run()

        if self.components is None:
            raise AssertionError(
                f"Components have not been defined for {self.name}. "
                f"Run `make_subgraphs_from_attribute` with `split_disconnected` "
                f"set to True."
            )

        # Log plotting
        logger.info(
            "Plotting component graph for %s with attribute %s",
            self.name,
            self.attribute_label,
        )
        # Bake component labels into graph
        for component in self.components:
            if not component["ignore"]:
                nx.set_edge_attributes(
                    component["subgraph"],
                    component["name"],
                    "component_name",
                )
        return plot.plot_by_attribute(
            self.graph,
            attr="component_name",
            attr_types="categorical",
            cmap="prism",
            minmax_val=None,
            **pba_kwargs,
        )

    def plot_subgraph_component_size(self, measure, xticks=None, **pcs_kwargs):
        """Plot the size of the subgraph components of the partitions.

        Scatter plot of the size of the subgraph components of each partition type.

        Parameters
        ----------
        measure : str, optional
            Way to measure component size. Can be 'edges', 'length' or 'nodes'.
        pcs_kwargs
            Keyword arguments to pass to `superblockify.plot.plot_component_size`.

        Returns
        -------
        fig, axe : tuple
            matplotlib figure, axis

        Raises
        ------
        AssertionError
            If BasePartitioner has not been run yet (the partitions are not defined).
        ValueError
            If measure is not 'edges', 'length' or 'nodes'.

        """

        self.__check_has_been_run()

        if measure not in ["edges", "length", "nodes"]:
            raise ValueError(
                f"Measure '{measure}' is not supported, "
                f"use 'edges', 'length' or 'nodes'."
            )

        # Find number of edges in each component for each partition
        key_name = "length_total" if measure == "length" else f"num_{measure}"
        component_size = []
        component_values = []
        ignore = []

        # If subgraphs were split, use components
        if self.components:
            logger.debug("Using components for plotting.")
            for comp in self.components:
                component_size.append(comp[key_name])
                component_values.append(comp["value"])
                ignore.append(comp["ignore"])
        # Else use partitions
        else:
            logger.debug("Using partitions for plotting.")
            for part in self.partitions:
                component_size.append(part[key_name])
                component_values.append(part["value"])
                ignore = None

        if xticks is None:
            xticks = list(linspace(*self.attr_value_minmax, 7))

        # Plot
        return plot.plot_component_size(
            graph=self.graph,
            attr=self.attribute_label,
            component_size=component_size,
            component_values=component_values,
            size_measure_label=f"Component size ({measure})",
            ignore=ignore,
            title=self.name,
            minmax_val=self.attr_value_minmax,
            xticks=xticks,
            **pcs_kwargs,
        )

    def __check_has_been_run(self):
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
                f"{self.__class__.__name__} has no `attribute_label` yet,"
                f"run before plotting graph."
            )

    def save_plot(self, fig, filename, **sa_kwargs):
        """Save the plot `fig` to file.

        Saved in the RESULTS_DIR/self.name/filename.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save.
        filename : str
            Filename to save to.
        sa_kwargs
            Keyword arguments to pass to `matplotlib.pyplot.savefig`.

        Notes
        -----
        RESULTS_DIR is set in the `config.ini` file.
        """

        filename = path.join(RESULTS_DIR, self.name, filename)
        # Log saving
        logger.debug(
            "Saving plot (%s) to %s",
            fig.axes[0].get_title(),
            filename,
        )

        # Save
        fig.savefig(filename, **sa_kwargs)

    def load_or_find_graph(self, name, search_str, reload_graph=False):
        """Load or find graph if it exists.

        If graph GRAPH_DIR/name.graphml exists, load it. Else, find it using
        `search_str` and save it to GRAPH_DIR/name.graphml.

        Parameters
        ----------
        name : str
            Name of the graph. Can be the name of the place and also be descriptive.
            Will be used for naming files and plot titles.
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
        graph_path = path.join(GRAPH_DIR, name + ".graphml")
        if path.exists(graph_path) and not reload_graph:
            logger.debug("Loading graph from %s", graph_path)
            graph = ox.load_graphml(graph_path)
        else:
            logger.debug("Finding graph with search string %s", search_str)
            graph = load_graph_from_place(
                save_as=graph_path,
                search_string=search_str,
                network_type="drive",
                simplify=True,
            )
            logger.debug("Saving graph to %s", graph_path)
            ox.save_graphml(graph, graph_path)
        return graph


class DummyPartitioner(BasePartitioner):
    """Dummy partitioner.

    Partitions randomly.
    """

    def run(self, make_plots=False, **kwargs):
        """Run method. Must be overridden.

        Assign random partitions to edges.
        """

        # The label under which the partition attribute is saved in the `self.graph`.
        self.attribute_label = "dummy_attribute"

        # Somehow determining the partition of edges
        # - edges also may not be included in any partition and miss the label
        values = list(range(3))
        attribute.new_edge_attribute_by_function(
            self.graph, lambda bear: choice(values), "osmid", self.attribute_label
        )

        # A List of the existing partitions, the 'value' attribute should be equal to
        # the edge attributes under the instances `attribute_label`, which belong to
        # this partition
        self.partitions = [
            {
                "name": str(num),
                "value": num,
                "subgraph": attribute.get_edge_subgraph_with_attribute_value(
                    self.graph, self.attribute_label, num
                ),
                "num_edges": num,
                "num_nodes": num,
                "length_total": num,
            }
            for num in values
        ]
