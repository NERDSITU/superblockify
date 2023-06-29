"""Tests for the betweenness partitioner."""
import pytest
from networkx import get_edge_attributes, set_edge_attributes

from superblockify.partitioning import BetweennessPartitioner


class TestBetweennessPartitioner:
    """Tests for the BetweennessPartitioner"""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("percentile", [99.9, 90.0, 80.0, 10.0, 1e-8])
    @pytest.mark.parametrize("scaling", ["normal", "length", "linear"])
    @pytest.mark.parametrize("max_range", [None, 2000])
    def test_run(self, test_one_city_copy, percentile, scaling, max_range):
        """Test the run method of the BetweennessPartitioner."""
        city_name, graph = test_one_city_copy
        part = BetweennessPartitioner(
            name=city_name + "_test",
            city_name=city_name,
            graph=graph,
        )
        part.run(
            calculate_metrics=False,
            percentile=percentile,
            scaling=scaling,
            max_range=max_range,
        )
        part.attribute_label = "betweenness_percentile"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("percentile", 100.0),
            ("percentile", None),
            ("percentile", 0.0),
            ("percentile", -1.0),
            ("percentile", 101.0),
            ("percentile", "string"),
            ("scaling", "normall"),
            ("scaling", None),
            ("scaling", 1),
        ],
    )
    def test_run_faulty_parameters(self, test_one_city_copy, key, value):
        """Test the run method of the BetweennessPartitioner with faulty parameters."""
        city_name, graph = test_one_city_copy
        part = BetweennessPartitioner(
            name=city_name + "_test",
            city_name=city_name,
            graph=graph,
        )
        with pytest.raises(ValueError):
            part.run(
                calculate_metrics=False,
                **{key: value},
            )

    @pytest.mark.parametrize("percentile", [100.0 - 1e-8, 90.0, 50.0, 10.0, 1e-8])
    @pytest.mark.parametrize(
        "val_distribution",
        [
            lambda x: x**4,  # values around 0 are clustered
            lambda x: 1 - (x - 1) ** 4,  # values around 1 are clustered
        ],
    )
    def test_write_attribute_threshold_adaption(
        self, test_one_city_copy, percentile, val_distribution, monkeypatch
    ):
        """Test adaption of threshold. At least one edge must be outside and at least
        one inside the sparse graph.

        Monkeypatch the calculate_metrics_before method to avoid the calculation of
        the metrics.

        Parameters
        ----------
        test_one_city_copy : tuple
            City name and graph.
        percentile : float
            The percentile to use for determining the high betweenness centrality
            edges.
        val_distribution : function
            Function that maps values [0, 1] to values [0, 1]. This is used to make
            the threshold adaption work.
        """
        _, graph = test_one_city_copy
        edges_total = graph.number_of_edges()
        set_edge_attributes(
            graph,
            {
                edge: round(val_distribution(edge_number / edges_total), 4)
                for edge_number, edge in enumerate(graph.edges(keys=True))
            },
            "edge_betweenness_linear",
        )

        part = BetweennessPartitioner(
            name="BetweennessPartitioner_test",
            city_name="BetweennessPartitioner_test",
            graph=graph,
        )
        monkeypatch.setattr(part, "calculate_metrics_before", lambda **kwargs: None)
        part.write_attribute(percentile, scaling="linear")
        # check that at least one edge is outside and at least one inside the sparse
        # graph
        count_0, count_1 = 0, 0
        for val in get_edge_attributes(part.graph, part.attribute_label).values():
            if val == 0:
                count_0 += 1
            elif val == 1:
                count_1 += 1
        assert 0 < count_0 < part.graph.number_of_edges()
        assert 0 < count_1 < part.graph.number_of_edges()
        print(f"count_0: {count_0}, count_1: {count_1}")
