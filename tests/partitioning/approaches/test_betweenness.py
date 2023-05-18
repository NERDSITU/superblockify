"""Tests for the partitioner module."""
import pytest

from superblockify.partitioning import BetweennessPartitioner


class TestBetweennessPartitioner:
    """Tests for the BetweennessPartitioner"""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("percentile", [100.0 - 1e-8, 90.0, 80.0, 10.0, 1e-8])
    @pytest.mark.parametrize("scaling", ["normal", "length", "linear"])
    def test_run(self, test_one_city_copy, percentile, scaling):
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
        )
        part.__class__.attribute_label = "betweenness_percentile"

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
