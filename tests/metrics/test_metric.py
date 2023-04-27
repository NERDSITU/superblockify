"""Tests for the metric class."""
import matplotlib.pyplot as plt
import pytest

from superblockify.metrics.metric import Metric
from tests.conftest import mark_xfail_flaky_download


class TestMetric:
    """Class to test the Metric class."""

    def test_init(self):
        """Test the init method."""
        metric = Metric()
        assert metric.coverage is None
        assert metric.num_components is None
        assert metric.avg_path_length == {"E": None, "S": None, "N": None}
        assert metric.directness == {"ES": None, "EN": None, "SN": None}
        assert metric.global_efficiency == {"SE": None, "NE": None, "NS": None}
        assert metric.distance_matrix is None

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

    def test_str(self):
        """Test the __str__ method."""
        metric = Metric()
        assert str(metric) == ""
        metric.coverage = 0.5
        assert str(metric) == "coverage: 0.5; "
        metric.num_components = 2
        assert str(metric) == "coverage: 0.5; num_components: 2; "
        metric.avg_path_length = {"E": None, "S": 4, "N": 11}
        assert (
            str(metric)
            == "coverage: 0.5; num_components: 2; avg_path_length: S: 4, N: 11; "
        )

    def test_repr(self):
        """Test the __repr__ method."""
        metric = Metric()
        assert repr(metric) == "Metric()"
        metric.coverage = 0.5
        assert repr(metric) == "Metric(coverage: 0.5; )"
        metric.num_components = 2
        assert repr(metric) == "Metric(coverage: 0.5; num_components: 2; )"
        metric.avg_path_length = {"E": None, "S": 4, "N": 11}
        assert (
            repr(metric) == "Metric(coverage: 0.5; num_components: 2; "
            "avg_path_length: S: 4, N: 11; )"
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
        part.calculate_metrics(
            make_plots=True, unit=unit, replace_max_speeds=replace_max_speeds
        )
        plt.close("all")
        for dist_matrix in part.metric.distance_matrix.values():
            assert dist_matrix.shape == (part.graph.number_of_nodes(),) * 2

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
