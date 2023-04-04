"""Tests for the metric class."""
import matplotlib.pyplot as plt
import pytest

from superblockify.metrics.metric import Metric


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

    def test_calculate_all_full(self, test_city_small_precalculated_copy):
        """Test the calculate_all method for full metrics."""
        part = test_city_small_precalculated_copy
        part.calculate_metrics(approach="full", make_plots=True)
        plt.close("all")
        for dist_matrix in part.metric.distance_matrix.values():
            assert dist_matrix.shape == (part.graph.number_of_nodes(),) * 2

    def test_calculate_all_rep_nodes(self, test_one_city_precalculated_copy):
        """Test the calculate_all method for representative node metrics."""
        part = test_one_city_precalculated_copy
        part.calculate_metrics(approach="rep_nodes", make_plots=True)
        plt.close("all")

    @pytest.mark.parametrize(
        "approach",  # only 'rep_nodes' and 'full' are valid
        [True, False, "invalid", 1, 0.5, -1, -0.5, None, [], {}, set()],
    )
    def test_calculate_all_invalid_approach(self, approach):
        """Test the calculate_all method for invalid approaches."""
        metric = Metric()
        with pytest.raises(ValueError):
            metric.calculate_all(None, approach=approach)

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
        part.run()
        # Save
        part.save(save_graph_copy=False)
        # Load
        metric = Metric.load(part.name)
        # Check if metrics are equal
        assert part.metric == metric
