"""Tests for the partitioner module."""
from configparser import ConfigParser
from os import path, remove
from shutil import rmtree

import networkx as nx
import pytest
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, Axes
from osmnx import load_graphml

from superblockify.partitioning import BasePartitioner

config = ConfigParser()
config.read("config.ini")
TEST_DATA = config["tests"]["test_data_path"]
GRAPH_DIR = config["general"]["graph_dir"]
RESULTS_DIR = config["general"]["results_dir"]


class TestBasePartitioner:
    """Class to test the BasePartitioner and its dummy class."""

    # pylint: disable=abstract-class-instantiated
    def test_instantiate_abstract_class(self, test_city_all):
        """Test instantiating the abstract base class itself."""
        _, graph = test_city_all
        with pytest.raises(TypeError):
            BasePartitioner(graph)

    def test_abstract_class_run_not_overridden(self):
        """Test instantiating a child without overriding abstract methods."""

        class ChildPartitioner(BasePartitioner):
            """Child instance without overriding `run` method."""

        with pytest.raises(TypeError):
            empty_graph = nx.empty_graph()
            ChildPartitioner(empty_graph)


class TestPartitioners:
    """Standard tests all classes of BasePartitioner need to suffice."""

    def test_run(self, test_city_all, partitioner_class):
        """Test run/partitioning method by design."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        part.run()
        assert part.graph is not None
        assert part.attribute_label is not None
        assert part.partitions is not None

    def test_plot_partition_graph(self, test_city_all, partitioner_class):
        """Test `plot_partition_graph` by design."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        part.run(show_analysis_plots=True)
        fig, axe = part.plot_partition_graph()
        assert isinstance(fig, Figure)
        assert isinstance(axe, Axes)
        plt.close("all")

    def test_plot_component_graph(self, test_city_all, partitioner_class):
        """Test `plot_component_graph` by design."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        part.run(show_analysis_plots=False)
        if part.components is not None:
            fig, axe = part.plot_component_graph()
            assert isinstance(fig, Figure)
            assert isinstance(axe, Axes)
            plt.close("all")

    def test_plot_partition_graph_unpartitioned(self, test_city_all, partitioner_class):
        """Test `plot_partition_graph` exception handling."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
        part.run()
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.plot_partition_graph()

    def test_plot_partitions_unpartitioned(self, test_city_all, partitioner_class):
        """Test `plot_partition_graph` exception handling."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
        part.run()
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.plot_partition_graph()

    def test_make_subgraphs_from_attribute(self, test_city_all, partitioner_class):
        """Test `make_subgraphs_from_attribute` by design."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.make_subgraphs_from_attribute()
        part.run()
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.make_subgraphs_from_attribute()

    def test_plot_subgraph_component_size(self, test_city_all, partitioner_class):
        """Test `plot_subgraph_component_size` by design."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        with pytest.raises(AssertionError):
            part.plot_subgraph_component_size("nodes")
        part.run()
        fig, _ = part.plot_subgraph_component_size("nodes")
        fig, _ = part.plot_subgraph_component_size("edges")
        fig, _ = part.plot_subgraph_component_size("length")
        fig.show()
        part.components = None
        fig, _ = part.plot_subgraph_component_size("nodes")
        fig.show()
        plt.close("all")

    @pytest.mark.parametrize(
        "invalid_measure",
        ["", "invalid", "node", None, 10, 1.0, True, False],
    )
    def test_plot_subgraph_component_size_invalid_measure(
        self, test_city_small, partitioner_class, invalid_measure
    ):
        """Test `plot_subgraph_component_size` with unavailable measure."""
        city_name, graph = test_city_small
        part = partitioner_class(graph, name=city_name)
        part.run()
        with pytest.raises(ValueError):
            part.plot_subgraph_component_size(invalid_measure)

    def test_overwrite_attributes_of_ignored_components_unpartitioned(
        self, test_city_small, partitioner_class
    ):
        """Test `overwrite_attributes_of_ignored_components` exception handling."""
        city_name, graph = test_city_small
        part = partitioner_class(graph, name=city_name)
        part.run()
        part.components = None
        with pytest.raises(AssertionError):
            part.overwrite_attributes_of_ignored_components(
                attribute_name=part.attribute_label
            )

    def test_get_sorted_node_list(self, test_city_all, partitioner_class):
        """Test `get_sorted_node_list` by design."""
        city_name, graph = test_city_all
        part = partitioner_class(graph, name=city_name)
        part.run()
        sorted_nodes = part.get_sorted_node_list()
        assert len(sorted_nodes) == len(graph.nodes)

    @pytest.mark.parametrize(
        "graph,name,search_str,reload_graph",
        [
            (
                load_graphml(path.join(TEST_DATA, "cities", "Adliswil_small.graphml")),
                "Adliswil_tmp",
                "Adliswil",
                False,
            ),
            (
                None,
                "Adliswil_tmp",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                False,
            ),
            (
                None,
                "Adliswil_tmp",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                True,
            ),
        ],
    )
    def test_graph_loading_and_finding(
        self,
        partitioner_class,
        graph,
        name,
        search_str,
        reload_graph,
        _teardown_graph_loading_and_finding,
    ):
        """Test loading and finding of graph files.
        Initialization of partitioner class and `self.load_or_find_graph`."""
        part = partitioner_class(graph, name, search_str, reload_graph)
        assert part.graph is not None
        assert part.name is not None

    @pytest.mark.parametrize(
        "graph,name,search_str",
        [
            (None, None, None),
            (None, "Adliswil", None),
            (None, None, "Adliswil, Bezirk Horgen, Zürich, Switzerland"),
        ],
    )
    def test_graph_loading_and_finding_invalid(
        self, partitioner_class, graph, name, search_str
    ):
        """Test loading and finding of graph files with invalid input."""
        with pytest.raises(ValueError):
            partitioner_class(graph, name, search_str)


@pytest.fixture(scope="class")
def _teardown_graph_loading_and_finding():
    """Delete Adliswil_tmp.graphml file and directory."""
    yield None
    test_graph = path.join(GRAPH_DIR, "Adliswil_tmp.graphml")
    if path.exists(test_graph):
        remove(test_graph)
    results_dir = path.join(RESULTS_DIR, "Adliswil_tmp")
    if path.exists(results_dir):
        rmtree(results_dir)
