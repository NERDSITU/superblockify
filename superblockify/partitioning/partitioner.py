"""BasePartitioner parent and dummy."""
import logging
from abc import ABC, abstractmethod
from random import choice

import networkx as nx

from .. import attribute, plot

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

    def make_subgraphs_from_attribute(self, split_disconnected=False):
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
                        }
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

    def plot_subgraph_component_size(self, **pcs_kwargs):
        """Plot the size of the subgraph components of the partitions.

        Scatter plot of the size of the subgraph components of each partition type.

        Parameters
        ----------
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

        """

        self.__check_has_been_runned()

        # Find number of edges in each component for each partition
        num_edges = []
        component_values = []

        # If subgraphs were split, use components
        if self.components:
            logger.debug("Using components for plotting.")
            for comp in self.components:
                num_edges.append(comp["num_edges"])
                component_values.append(comp["value"])
        # Else use partitions
        else:
            logger.debug("Using partitions for plotting.")
            for part in self.partition:
                num_edges.append(part["num_edges"])
                component_values.append(part["value"])

        # Plot
        return plot.plot_component_size(
            graph=self.graph,
            attr=self.attribute_label,
            num_edges=num_edges,
            component_values=component_values,
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
            {"name": str(num), "value": num, "num_edges": num} for num in values
        ]
