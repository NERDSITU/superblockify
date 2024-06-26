"""Tests for the partitioner module."""

from os import path, remove
from os.path import exists, join

import networkx as nx
import pytest
from osmnx import load_graphml

from superblockify.config import logger, Config
from superblockify.partitioning import BasePartitioner
from superblockify.utils import compare_components_and_partitions
from tests.conftest import mark_xfail_flaky_download


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

    def test_run_reduced_graph(self, test_city_all_reduced_precalculated):
        """Test run/partitioning method by design with reduced graphs."""
        part = test_city_all_reduced_precalculated
        assert part.attribute_label is not None
        assert part.partitions is not None
        assert part.name is not None and part.name != ""
        assert part.city_name is not None and part.city_name != ""
        assert "reduced_population" in part.graph.graph
        assert "reduced_area" in part.graph.graph
        assert "reduced_street_orientation_order" in part.graph.graph
        assert "reduced_circuity_avg" in part.graph.graph
        assert "reduced_n" in part.graph.graph
        assert "reduced_m" in part.graph.graph

    @pytest.mark.parametrize("replace_max_speeds", [False, True])
    def test_run_make_plots(self, test_city_all_preloaded, replace_max_speeds):
        """Test plotting of partitioning results by design."""
        test_city_all_preloaded.run(
            calculate_metrics=False,
            make_plots=True,
            replace_max_speeds=replace_max_speeds,
        )

    @pytest.mark.parametrize(
        "unit,replace_max_speeds",
        [
            ("distance", False),
            ("distance", True),
            (None, False),
        ],
    )
    def test_ignore_replace_max_speeds(
        self,
        caplog,
        test_one_city_preloaded_copy,
        unit,
        replace_max_speeds,
    ):
        """Test ignoring and replacing max_speeds by design.
        When unit is not "time", ``replace_max_speeds`` is ignored, warn if
        ``ignore_max_speeds`` is set anyway.
        """
        caplog.clear()
        test_one_city_preloaded_copy.metric.unit = unit
        test_one_city_preloaded_copy.run(
            calculate_metrics=False,
            make_plots=False,
            replace_max_speeds=replace_max_speeds,
        )
        assert (
            caplog.records[0].message
            == "replace_max_speeds is ignored when unit is not 'time'."
        )

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
                    path.join(Config.TEST_DATA_PATH, "cities", "Adliswil.graphml")
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
        part = partitioner_class(
            name, city_name, search_str, graph=graph, reload_graph=reload_graph
        )
        assert part.graph is not None
        assert part.name is not None
        if reload_graph:
            test_graph = join(Config.GRAPH_DIR, "Adliswil_tmp.graphml")
            if exists(test_graph):
                remove(test_graph)

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

    def test_no_search_str_graph_given(self, partitioner_class):
        """Tests error raising when no graph, nor search_str is given."""
        with pytest.raises(ValueError):
            partitioner_class(
                name="test",
                city_name="test",
            )

    @mark_xfail_flaky_download
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
    ):  # pylint: disable=too-many-branches
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
            remove(path.join(Config.GRAPH_DIR, "Adliswil_tmp_save_load.graphml"))
            # Delete metrics at RESULTS_DIR/.../Adliswil_tmp_save_load_name.metrics
            remove(
                path.join(
                    Config.RESULTS_DIR,
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
                if not save_graph_copy:
                    # the edges where the id is looking like uuid4().int must be
                    # ignored. The rest must be the same. Delete nodes from both graphs
                    # delete any id that does not fit into int64
                    for node in list(getattr(part, attr).nodes):
                        if node >= 2**63:
                            getattr(part, attr).remove_node(node)
                    for node in list(getattr(part_loaded, attr).nodes):
                        if node >= 2**63:
                            getattr(part_loaded, attr).remove_node(node)
                assert set(getattr(part, attr).nodes) == set(
                    getattr(part_loaded, attr).nodes
                )
                if save_graph_copy:
                    assert set(getattr(part, attr).edges) == set(
                        getattr(part_loaded, attr).edges
                    )
            elif (
                attr in ["components", "partitions"] and getattr(part, attr) is not None
            ):
                if save_graph_copy:
                    # For not saved graph copy, the components and partitions might
                    # not be reconstructed completely.
                    assert compare_components_and_partitions(
                        getattr(part, attr), getattr(part_loaded, attr)
                    )
                else:  # Therefore, only compare the keys of the dicts in the lists.
                    assert all(
                        set(elem.keys()) == set(elem_loaded.keys())
                        for elem, elem_loaded in zip(
                            getattr(part, attr), getattr(part_loaded, attr)
                        )
                    )
            elif all(
                isinstance(elem, dict)
                for elem in [getattr(part, attr), getattr(part_loaded, attr)]
            ):
                # Compare two dicts only by their keys
                assert getattr(part, attr).keys() == getattr(part_loaded, attr).keys()
            else:
                assert getattr(part, attr) == getattr(part_loaded, attr)

    @pytest.mark.parametrize("other_name", [None, "other_name"])
    def test_save_key_figures(
        self, test_city_small_precalculated_copy, other_name, _teardown_test_folders
    ):
        """Test saving of key figures."""
        part = test_city_small_precalculated_copy
        part.save_key_figures(
            save_dir=join(Config.RESULTS_DIR, "key_figures_test"), name=other_name
        )
        assert path.isfile(
            path.join(
                Config.RESULTS_DIR,
                "key_figures_test",
                f"{other_name or part.name}_key_figures.yml",
            )
        )

    def test_load_file_not_found(self, partitioner_class):
        """Test loading of partitioner with file not found."""
        with pytest.raises(FileNotFoundError):
            partitioner_class.load("file_not_found")
