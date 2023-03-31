"""Tests for the partitioner module."""
import logging
from configparser import ConfigParser
from os import path, remove

import networkx as nx
import pytest
from matplotlib import pyplot as plt
from matplotlib.pyplot import Figure, Axes
from osmnx import load_graphml

from superblockify.partitioning import BasePartitioner
from superblockify.utils import compare_components_and_partitions

logger = logging.getLogger("superblockify")

config = ConfigParser()
config.read("config.ini")


class TestBasePartitioner:
    """Class to test the BasePartitioner and its dummy class."""

    # pylint: disable=abstract-class-instantiated
    def test_instantiate_abstract_class(self, test_city_all_copy):
        """Test instantiating the abstract base class itself."""
        _, graph = test_city_all_copy
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

    def test_run(self, test_city_all_precalculated):
        """Test run/partitioning method by design."""
        part = test_city_all_precalculated
        assert part.graph is not None
        assert part.attribute_label is not None
        assert part.partitions is not None
        assert part.name is not None and part.name != ""
        assert part.city_name is not None and part.city_name != ""

    def test_plot_partition_graph(self, test_city_small_precalculated_copy):
        """Test `plot_partition_graph` by design."""
        part = test_city_small_precalculated_copy
        fig, axe = part.plot_partition_graph()
        assert isinstance(fig, Figure)
        assert isinstance(axe, Axes)
        plt.close("all")

    def test_plot_component_graph(self, test_city_all_precalculated):
        """Test `plot_component_graph` by design."""
        part = test_city_all_precalculated
        if part.components is not None:
            fig, axe = part.plot_component_graph()
            assert isinstance(fig, Figure)
            assert isinstance(axe, Axes)
            plt.close("all")

    def test_plot_partition_graph_unpartitioned(
        self, test_city_all_preloaded, test_city_all_precalculated
    ):
        """Test `plot_partition_graph` exception handling."""
        part = test_city_all_preloaded
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
        part = test_city_all_precalculated
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.plot_partition_graph()

    def test_plot_partitions_unpartitioned(
        self, test_city_all_preloaded, test_city_all_precalculated
    ):
        """Test `plot_partition_graph` exception handling."""
        part = test_city_all_preloaded
        with pytest.raises(AssertionError):
            part.plot_partition_graph()
        part = test_city_all_precalculated
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.plot_partition_graph()

    def test_make_subgraphs_from_attribute(
        self, test_city_all_preloaded, test_city_all_precalculated
    ):
        """Test `make_subgraphs_from_attribute` by design."""
        part = test_city_all_preloaded
        with pytest.raises(AssertionError):
            part.make_subgraphs_from_attribute()
        part = test_city_all_precalculated
        part.attribute_label = None
        with pytest.raises(AssertionError):
            part.make_subgraphs_from_attribute()

    def test_plot_subgraph_component_size(
        self, test_city_all_preloaded, test_city_all_precalculated
    ):
        """Test `plot_subgraph_component_size` by design."""
        part = test_city_all_preloaded
        with pytest.raises(AssertionError):
            part.plot_subgraph_component_size("nodes")
        part = test_city_all_precalculated
        part.plot_subgraph_component_size("nodes")
        part.plot_subgraph_component_size("edges")
        part.plot_subgraph_component_size("length")
        part.components = None
        part.plot_subgraph_component_size("nodes")
        plt.close("all")

    @pytest.mark.parametrize(
        "invalid_measure",
        ["", "invalid", "node", None, 10, 1.0, True, False],
    )
    def test_plot_subgraph_component_size_invalid_measure(
        self, test_city_small_precalculated_copy, invalid_measure
    ):
        """Test `plot_subgraph_component_size` with unavailable measure."""
        part = test_city_small_precalculated_copy
        with pytest.raises(ValueError):
            part.plot_subgraph_component_size(invalid_measure)

    def test_overwrite_attributes_of_ignored_components_unpartitioned(
        self, test_city_small_precalculated_copy
    ):
        """Test `overwrite_attributes_of_ignored_components` exception handling."""
        part = test_city_small_precalculated_copy
        part.components = None
        with pytest.raises(AssertionError):
            part.overwrite_attributes_of_ignored_components(
                attribute_name=part.attribute_label
            )

    def test_get_sorted_node_list(self, test_city_small_precalculated_copy):
        """Test `get_sorted_node_list` by design."""
        part = test_city_small_precalculated_copy
        sorted_nodes = part.get_sorted_node_list()
        assert set(sorted_nodes) == set(part.graph.nodes())

    @pytest.mark.parametrize(
        "name,city_name,search_str,graph,reload_graph",
        [
            (
                "Adliswil_tmp_name",
                "Adliswil_tmp",
                None,
                load_graphml(
                    path.join(
                        config["tests"]["test_data_path"], "cities", "Adliswil.graphml"
                    )
                ),
                False,
            ),
            (
                "Adliswil_tmp_name",
                "Adliswil_tmp",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                False,
            ),
            (
                "Adliswil_tmp_name",
                "Adliswil_tmp",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                True,
            ),
        ],
    )
    def test_graph_loading_and_finding(
        self,
        partitioner_class,
        name,
        city_name,
        search_str,
        graph,
        reload_graph,
        _teardown_test_graph_io,
    ):
        """Test loading and finding of graph files.
        Initialization of partitioner class and `self.load_or_find_graph`."""
        part = partitioner_class(name, city_name, search_str, graph, reload_graph)
        assert part.graph is not None
        assert part.name is not None

    @pytest.mark.parametrize(
        "name,city_name,search_str,graph,error_type",
        [
            (None, None, None, None, ValueError),
            ("Adliswil_name", None, None, None, ValueError),
            (
                None,
                None,
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                ValueError,
            ),
            (
                None,
                "Adliswil",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                ValueError,
            ),
            (
                "",
                "Adliswil",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                ValueError,
            ),
            ("", "Adliswil", None, None, ValueError),
            (
                "Adliswil_name",
                None,
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                ValueError,
            ),
            (
                "Adliswil_name",
                "",
                "Adliswil, Bezirk Horgen, Zürich, Switzerland",
                None,
                ValueError,
            ),
            ("Adliswil_name", "Adliswil", "", None, KeyError),
            ("Adliswil_name", "Adliswil", [], None, KeyError),
            ("Adliswil_name", "Adliswil", [""], None, KeyError),
        ],
    )
    def test_graph_loading_and_finding_invalid(
        self, partitioner_class, name, city_name, search_str, graph, error_type
    ):
        """Test loading and finding of graph files with invalid input."""
        with pytest.raises(error_type):
            partitioner_class(name, city_name, search_str, graph)

    @pytest.mark.parametrize(
        "save_graph_copy,delete_before_load",
        [(False, False), (True, False), (False, True)],
    )
    def test_saving_and_loading(
        self,
        partitioner_class,
        save_graph_copy,
        delete_before_load,
        _teardown_test_graph_io,
    ):
        """Test saving and loading of partitioner."""
        # Prepare
        part = partitioner_class(
            name="Adliswil_tmp_save_load_name",
            city_name="Adliswil_tmp_save_load",
            search_str="Adliswil, Bezirk Horgen, Zürich, Switzerland",
        )
        part.run(calculate_metrics=False)

        # Save
        part.save(save_graph_copy)
        if delete_before_load:
            # Delete graph at GRAPH_DIR/Adliswil_tmp_save_load.graphml
            remove(
                path.join(
                    config["general"]["graph_dir"], "Adliswil_tmp_save_load.graphml"
                )
            )
            # Delete metrics at GRAPH_DIR/Adliswil_tmp_save_load.metric
            remove(
                path.join(
                    config["general"]["results_dir"],
                    "Adliswil_tmp_save_load_name",
                    "Adliswil_tmp_save_load_name.metrics",
                )
            )

        # Load
        part_loaded = partitioner_class.load(part.name)
        # Check if all instance keys are equal
        assert part.__dict__.keys() == part_loaded.__dict__.keys()
        # Check if all instance attributes are equal (except graph if deleted)
        for attr in part.__dict__:
            if attr == "metric" or (attr == "graph" and delete_before_load):
                continue
            if isinstance(getattr(part, attr), nx.Graph):
                # For the graph only check equality of the nodes and edges, not the
                # node and edge attributes as the modifications are not saved.
                logger.debug(
                    "Comparing instance attribute %s of partitioner %s.",
                    attr,
                    part.name,
                )
                assert set(getattr(part, attr).nodes) == set(
                    getattr(part_loaded, attr).nodes
                ) and set(getattr(part, attr).edges) == set(
                    getattr(part_loaded, attr).edges
                )
            elif (
                attr in ["components", "partitions"] and getattr(part, attr) is not None
            ):
                assert compare_components_and_partitions(
                    getattr(part, attr), getattr(part_loaded, attr)
                )
            elif all(
                isinstance(elem, dict)
                for elem in [getattr(part, attr), getattr(part_loaded, attr)]
            ):
                # Compare two dicts only by their keys
                assert getattr(part, attr).keys() == getattr(part_loaded, attr).keys()
            else:
                assert getattr(part, attr) == getattr(part_loaded, attr)

    def test_load_file_not_found(self, partitioner_class):
        """Test loading of partitioner with file not found."""
        with pytest.raises(FileNotFoundError):
            partitioner_class.load("file_not_found")
