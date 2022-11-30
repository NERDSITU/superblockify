"""Tests for the partitioner module."""
from configparser import ConfigParser

from matplotlib.pyplot import Figure, Axes
import pytest
import networkx as nx

from superblockify.partitioning import BasePartitioner, DummyPartitioner

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]


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


class TestDummyPartitioner:
    """Tests for the dummy class of BasePartitioner."""

    def test_run(self, test_city_bearing):
        """Test run/partitioning method by design."""
        _, graph = test_city_bearing
        part = DummyPartitioner(graph)
        part.run()
        assert part.graph is not None
        assert part.attribute_label is not None
        assert part.partition is not None

    def test_plot_partition_graph(self, test_city_bearing):
        """Test `plot_partition_graph` by design."""
        _, graph = test_city_bearing
        part = DummyPartitioner(graph)
        part.run()
        fig, axe = part.plot_partition_graph()
        assert isinstance(fig, Figure)
        assert isinstance(axe, Axes)

    def test_plot_partitions_unpartitioned(self, test_city_bearing):
        """Test `plot_partition_graph` exception handling."""
        _, graph = test_city_bearing
        part = DummyPartitioner(graph)
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
        part.run()
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
