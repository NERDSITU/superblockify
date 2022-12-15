"""BasePartitioner parent and dummy."""
from abc import ABC, abstractmethod
from random import choice

from .. import attribute, plot


class BasePartitioner(ABC):
    """Parent class for partitioning graphs."""

    def __init__(self, graph, name="unnamed"):
        """Constructing a BasePartitioner

        Parameters
        ----------
        graph : networkx.Graph
            Input graph
        name : str, optional
            Name of the graph's city, default is 'unnamed'.

        """

        # Set Instance variables
        self.graph = graph
        self.name = name
        self.partition = None
        self.attribute_label = None

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
        fig, ax : tuple
            matplotlib figure, axis

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

        return plot.plot_by_attribute(self.graph, self.attribute_label, **pba_kwargs)


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
        self.partition = [{"name": str(num), "value": num} for num in values]
