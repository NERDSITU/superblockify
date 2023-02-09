"""Tests for the partitioner module."""
import networkx as nx
import pytest
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, Axes

from superblockify.partitioning import BasePartitioner


class TestBasePartitioner:
    """Class to test the BasePartitioner and its dummy class."""

    # pylint: disable=abstract-class-instantiated
    def test_instantiate_abstract_class(self, test_city_bearing):
        """Test instantiating the abstract base class itself."""
        _, graph = test_city_bearing
        with pytest.raises(TypeError):
            BasePartitioner(graph)

    def test_abstract_class_run_not_overridden(self):
        """Test instantiating a child without overriding abstract methods."""

        class ChildPartitioner(BasePartitioner):
            """Child instance without overriding `run` method."""

        with pytest.raises(TypeError):
            empty_graph = nx.empty_graph()
            ChildPartitioner(empty_graph)

    # pylint: enable=abstract-class-instantiated


class TestPartitioners:
    """Standard tests all classes of BasePartitioner need to suffice."""

    def test_run(self, test_city_bearing, partitioner_class):
        """Test run/partitioning method by design."""
        city_name, graph = test_city_bearing
        part = partitioner_class(graph, name=city_name)
        part.run()
        assert part.graph is not None
        assert part.attribute_label is not None
        assert part.partition is not None

    def test_plot_partition_graph(self, test_city_bearing, partitioner_class):
        """Test `plot_partition_graph` by design."""
        city_name, graph = test_city_bearing
        part = partitioner_class(graph, name=city_name)
        part.run(show_analysis_plots=True)
        fig, axe = part.plot_partition_graph()
        assert isinstance(fig, Figure)
        assert isinstance(axe, Axes)

    def test_plot_partitions_unpartitioned(self, test_city_bearing, partitioner_class):
        """Test `plot_partition_graph` exception handling."""
        city_name, graph = test_city_bearing
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
        part.run()
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.plot_partition_graph()

    def test_make_subgraphs_from_attribute(self, test_city_bearing, partitioner_class):
        """Test `make_subgraphs_from_attribute` by design."""
        city_name, graph = test_city_bearing
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.make_subgraphs_from_attribute()
        part.run()
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.make_subgraphs_from_attribute()

    def test_plot_subgraph_component_size(self, test_city_bearing, partitioner_class):
        """Test `plot_subgraph_component_size` by design."""
        city_name, graph = test_city_bearing
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.plot_subgraph_component_size()
        part.run()
        fig, _ = part.plot_subgraph_component_size()
        fig.show()
        part.components = None
        fig, _ = part.plot_subgraph_component_size()
        fig.show()
        plt.close()
