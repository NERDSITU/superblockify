"""BasePartitioner parent and dummy."""
import logging
from abc import ABC, abstractmethod
from random import choice

import networkx as nx

from .. import attribute, plot, metrics

logger = logging.getLogger("superblockify")


class BasePartitioner(ABC):
    """Parent class for partitioning graphs."""

    def __init__(self, graph, name="unnamed"):
        """Constructing a BasePartitioner

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            Input graph
        name : str, optional
            Name of the graph's city, default is 'unnamed'.

        """

        # Set Instance variables
        self.graph = graph
        self.name = name
        self.partition = None
        self.components = None
        self.attribute_label = None
        self.attr_value_minmax = None
        self.metric = metrics.Metric()

        # Log initialization
        logger.info(
            "Initialized %s(%s) with %d nodes and %d edges.",
            self.name,
            self.__class__.__name__,
            len(graph.nodes),
            len(graph.edges),
        )

    @abstractmethod
    def run(self, **kwargs):
        """Run partitioning.

        Parameters
        ----------
        show_analysis_plots : bool, optional
            If True show visualization graphs of the approach, if implemented.
        """

        self.attribute_label = "example_label"
        # Define partitions
        self.partition = [{"name": "zero", "value": 0.0}, {"name": "one", "value": 1.0}]

    def calculate_metrics(self):
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

        """

        # Log calculating metrics
        logger.debug("Calculating metrics for %s", self.name)
        self.metric.calculate_all(self)
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
            If BasePartitioner has not been runned yet (the partitions are not defined).

        """

        self.__check_has_been_runned()

        # Log making subgraphs
        logger.info(
            "Making subgraphs for %s with attribute %s",
            self.name,
            self.attribute_label,
        )

        found_disconnected = False
        num_partitions = len(self.partition)

        # Make component subgraphs from attribute
        for part in self.partition:
            logger.debug("Making subgraph for partition %s", part)
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

        for part in self.partition:
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
            f"and {len(self.partition)} components."
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
            If BasePartitioner has not been runned yet (the partitions are not defined).
        AssertionError
            If `self.components` is not defined (the subgraphs have not been split
            into components).

        """

        self.__check_has_been_runned()

        if self.components is None:
            raise AssertionError(
                f"Components have not been defined for {self.name}. "
                f"Run `make_subgraphs_from_attribute` first."
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

    def plot_partition_graph(self, **pba_kwargs):
        """Plotting the partition with color on graph.

        Plots the partitioned graph, just like `plot.paint_streets` but that the
        partitions have a uniform color.

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
            If BasePartitioner has not been runned yet (the partitions are not defined).

        """

        self.__check_has_been_runned()

        # Log plotting
        logger.info(
            "Plotting partition graph for %s with attribute %s",
            self.name,
            self.attribute_label,
        )
        return plot.plot_by_attribute(
            self.graph,
            self.attribute_label,
            minmax_val=self.attr_value_minmax,
            **pba_kwargs,
        )

    def plot_subgraph_component_size(self, measure, **pcs_kwargs):
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
            If BasePartitioner has not been runned yet (the partitions are not defined).
        ValueError
            If measure is not 'edges', 'length' or 'nodes'.

        """

        self.__check_has_been_runned()

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
            for part in self.partition:
                component_size.append(part[key_name])
                component_values.append(part["value"])
                ignore = None

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
            **pcs_kwargs,
        )

    def __check_has_been_runned(self):
        """Check if the partitioner has runned.

        Raises
        ------
        AssertionError
            If BasePartitioner has not been runned yet (the partitions are not defined).

        """

        if self.partition is None:
            raise AssertionError(
                f"{self.__class__.__name__} has no partitions, "
                f"run before plotting graph."
            )
        if self.attribute_label is None:
            raise AssertionError(
                f"{self.__class__.__name__} has no `attribute_label` yet,"
                f"run before plotting graph."
            )


class DummyPartitioner(BasePartitioner):
    """Dummy partitioner.

    Partitions randomly.
    """

    def run(self, **kwargs):
        """Run method. Must be overridden.

        Assign random partitions to edges.
        """

        # The label under which partition attribute is saved in the `self.graph`.
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
        self.partition = [
            {
                "name": str(num),
                "value": num,
                "num_edges": num,
                "num_nodes": num,
                "length_total": num,
            }
            for num in values
        ]
