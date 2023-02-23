"""Tests for the metrics module."""
import matplotlib.pyplot as plt
import pytest
from numpy import inf

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
        """Test calculating all pairwise distances for the full graphs."""
        _, graph = test_city_small
        metric = Metric()
        metric.calculate_distance_matrix(graph, weight=weight, plot_distributions=True)
        # With node ordering
        metric.calculate_distance_matrix(
            graph, node_order=list(graph.nodes), plot_distributions=True
        )
        plt.close("all")

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

    def test_calculate_euclidean_distance_matrix_projected(self, test_city_all):
        """Test calculating all pairwise euclidean distances for the full graphs.
        Projected."""
        _, graph = test_city_all
        metric = Metric()
        metric.calculate_euclidean_distance_matrix_projected(
            graph, plot_distributions=True
        )
        # With node ordering
        metric.calculate_euclidean_distance_matrix_projected(
            graph, node_order=list(graph.nodes), plot_distributions=True
        )
        plt.close("all")

    @pytest.mark.parametrize(
        "key,value",
        [
            ("x", None),
            ("y", None),
            ("x", "a"),
            ("y", "a"),
            ("x", inf),
            ("y", inf),
            ("x", -inf),
            ("y", -inf),
        ],
    )
    def test_calculate_euclidean_distance_matrix_projected_faulty_coords(
        self, test_city_small, key, value
    ):
        """Test calculating all pairwise euclidean distances for the full graphs
        with missing coordinates. Projected.
        """
        _, graph = test_city_small
        # Change key attribute of first node
        graph.nodes[list(graph.nodes)[0]][key] = value
        metric = Metric()
        with pytest.raises(ValueError):
            metric.calculate_euclidean_distance_matrix_projected(graph)

    def test_calculate_euclidean_distance_matrix_projected_unprojected_graph(
        self, test_city_small
    ):
        """Test `calculate_euclidean_distance_matrix_projected` exception handling
        unprojected graph."""
        _, graph = test_city_small
        metric = Metric()

        # Pseudo-unproject graph
        graph.graph["crs"] = "epsg:4326"
        with pytest.raises(ValueError):
            metric.calculate_euclidean_distance_matrix_projected(graph)

        # Delete crs attribute
        graph.graph.pop("crs")
        with pytest.raises(ValueError):
            metric.calculate_euclidean_distance_matrix_projected(graph)

    def test_calculate_euclidean_distance_matrix_haversine(self, test_city_small):
        """Test calculating all pairwise euclidean distances for the full graphs.
        Haversine."""
        _, graph = test_city_small
        metric = Metric()
        metric.calculate_euclidean_distance_matrix_haversine(
            graph, plot_distributions=True
        )
        # With node ordering
        metric.calculate_euclidean_distance_matrix_haversine(
            graph, node_order=list(graph.nodes), plot_distributions=True
        )
        plt.close("all")

    @pytest.mark.parametrize(
        "key,value",
        [
            ("lat", None),
            ("lon", None),
            ("lat", "a"),
            ("lon", "a"),
            ("lat", -90.1),
            ("lon", -180.1),
            ("lat", 90.1),
            ("lon", 180.1),
            ("lat", inf),
            ("lon", inf),
            ("lat", -inf),
            ("lon", -inf),
        ],
    )
    def test_calculate_euclidean_distance_matrix_haversine_faulty_coords(
        self, test_city_small, key, value
    ):
        """Test calculating all pairwise euclidean distances for the full graphs
        with missing coordinates. Haversine.
        """
        _, graph = test_city_small
        # Change key attribute of first node
        graph.nodes[list(graph.nodes)[0]][key] = value
        metric = Metric()
        with pytest.raises(ValueError):
            metric.calculate_euclidean_distance_matrix_haversine(graph)
