"""Tests for the metrics module."""
import pytest

from superblockify.metrics import Metric


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
        assert metric.local_efficiency == {"SE": None, "NE": None, "NS": None}
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

    def test_calculate_all(self, test_city_small, partitioner_class):
        """Test the calculate_all method for full metrics."""
        city_name, graph = test_city_small
        part = partitioner_class(graph, name=city_name)
        part.run()
        part.calculate_metrics()

    @pytest.mark.parametrize("weight", ["length", None])
    def test_calculate_distance_matrix(self, test_city_small, weight):
        """Test calculating all pairwise distances for the full graphs with timeout."""
        _, graph = test_city_small
        metric = Metric()
        metric.calculate_distance_matrix(graph, weight=weight)

    def test_calculate_distance_matrix_negative_weight(self, test_city_small):
        """Test calculating all pairwise distances for the full graphs with negative
        weights.
        """
        _, graph = test_city_small
        # Change the first edge length to -1
        graph.edges[list(graph.edges)[0]]["length"] = -1
        metric = Metric()
        with pytest.raises(ValueError):
            metric.calculate_distance_matrix(graph, weight="length")
