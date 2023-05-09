"""Tests for the metric class."""
import matplotlib.pyplot as plt
import pytest

from superblockify.metrics.metric import Metric
from tests.conftest import mark_xfail_flaky_download


class TestMetric:
    """Class to test the Metric class."""

    @pytest.mark.parametrize("unit", ["time", "distance", None])
    def test_init(self, unit):
        """Test the init method."""
        metric = Metric(unit=unit)
        assert metric.coverage is None
        assert metric.num_components is None
        assert metric.avg_path_length == {"S": None, "N": None}
        assert metric.directness == {"SN": None}
        assert metric.global_efficiency == {"NS": None}
        assert not metric.distance_matrix
        assert not metric.predecessor_matrix
        assert metric.unit == unit
        assert metric.node_list is None

    @pytest.mark.parametrize(
        "unit,expected_symbol",
        [
            ("time", "s"),
            ("distance", "m"),
            (None, "hops"),
            ("bla", "(bla)"),
            (1, "(1)"),
            (0, "(0)"),
            (True, "(True)"),
            (False, "(False)"),
        ],
    )
    def test_unit_symbol(self, unit, expected_symbol):
        """Test the unit_symbol method."""
        metric = Metric()
        metric.unit = unit
        assert metric.unit_symbol() == expected_symbol

    @pytest.mark.parametrize("unit", ["time", "distance", None])
    def test_str(self, unit):
        """Test the __str__ method."""
        metric = Metric(unit=unit)
        assert str(metric) == f"unit: {unit}; "
        metric.coverage = 0.5
        assert str(metric) == f"coverage: 0.5; unit: {unit}; "
        metric.num_components = 2
        assert str(metric) == f"coverage: 0.5; num_components: 2; unit: {unit}; "
        metric.avg_path_length = {"E": None, "S": 4, "N": 11}
        assert (
            str(metric) == f"coverage: 0.5; num_components: 2; avg_path_length: S: 4, "
            f"N: 11; unit: {unit}; "
        )

    @pytest.mark.parametrize("unit", ["time", "distance", None])
    def test_repr(self, unit):
        """Test the __repr__ method."""
        metric = Metric(unit=unit)
        assert repr(metric) == f"Metric(unit: {unit}; )"
        metric.coverage = 0.5
        assert repr(metric) == f"Metric(coverage: 0.5; unit: {unit}; )"
        metric.num_components = 2
        assert (
            repr(metric) == f"Metric(coverage: 0.5; num_components: 2; unit: {unit}; )"
        )
        metric.avg_path_length = {"E": None, "S": 4, "N": 11}
        assert (
            repr(metric) == "Metric(coverage: 0.5; num_components: 2; "
            f"avg_path_length: S: 4, N: 11; unit: {unit}; )"
        )

    @pytest.mark.parametrize(
        "unit,replace_max_speeds",
        [
            ("time", True),
            ("time", False),
            ("distance", False),
            (None, False),
        ],
    )
    def test_calculate_metrics(
        self, test_city_small_precalculated_copy, unit, replace_max_speeds
    ):
        """Test the calculate_all method for full metrics."""
        part = test_city_small_precalculated_copy
        part.metric.unit = unit
        part.calculate_metrics(make_plots=True, replace_max_speeds=replace_max_speeds)
        plt.close("all")
        for dist_matrix in part.metric.distance_matrix.values():
            assert dist_matrix.shape == (part.graph.number_of_nodes(),) * 2

    @pytest.mark.parametrize("unit", ["time", "distance"])
    def test_calculate_metrics_before(self, test_one_city_precalculated_copy, unit):
        """Test the metric calculation for before partitioning."""
        part = test_one_city_precalculated_copy
        part.metric.unit = unit
        part.calculate_metrics_before(make_plots=True)
        plt.close("all")
        assert part.metric.node_list is not None
        assert (
            part.metric.distance_matrix["S"].shape
            == (part.graph.number_of_nodes(),) * 2
        )
        assert (
            part.metric.predecessor_matrix["S"].shape
            == (part.graph.number_of_nodes(),) * 2
        )
        if unit == "distance":
            assert (
                part.metric.distance_matrix["E"].shape
                == (part.graph.number_of_nodes(),) * 2
            )
        for bc_type in ["normal", "length", "linear"]:
            assert (
                len(part.graph.nodes(data=f"node_betweenness_{bc_type}"))
                == part.graph.number_of_nodes()
            )
            assert (
                len(part.graph.edges(data=f"edge_betweenness_{bc_type}"))
                == part.graph.number_of_edges()
            )

    @mark_xfail_flaky_download
    def test_saving_and_loading(
        self,
        partitioner_class,
        _teardown_test_graph_io,
    ):
        """Test saving and loading of metrics."""
        # Prepare
        part = partitioner_class(
            name="Adliswil_tmp_name",
            city_name="Adliswil_tmp",
            search_str="Adliswil, Bezirk Horgen, Zürich, Switzerland",
        )
        part.run(calculate_metrics=True, make_plots=False)
        # Save
        part.save(save_graph_copy=False)
        # Load
        metric = Metric.load(part.name)
        # Check if metrics are equal
        assert part.metric == metric
