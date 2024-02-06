"""Tests for the steiner tree partitioner."""

import pytest

from superblockify.partitioning import MinimumPartitioner


class TestMinimumPartitioner:  # pylint: disable=too-few-public-methods
    """Tests for the MinimumPartitioner"""

    @pytest.mark.parametrize("weight", ["length", "travel_time", None])
    @pytest.mark.parametrize("fraction", [0.2, 0.6])
    @pytest.mark.parametrize("low_betweenness_mode", [None, "normal"])
    @pytest.mark.parametrize("num_subtrees", [1, 2])
    @pytest.mark.parametrize("remove_oneway_edges", [True, False])
    def test_run(
        self,
        test_one_city_copy,
        weight,
        fraction,
        low_betweenness_mode,
        num_subtrees,
        remove_oneway_edges,
    ):
        """Test the run method of the BetweennessPartitioner."""
        city_name, graph = test_one_city_copy
        part = MinimumPartitioner(
            name=city_name + "_test",
            city_name=city_name,
            graph=graph,
        )
        part.run(
            calculate_metrics=False,
            weight=weight,
            fraction=fraction,
            low_betweenness_mode=low_betweenness_mode,
            num_subtrees=num_subtrees,
            remove_oneway_edges=remove_oneway_edges,
        )
        assert part.attribute_label == "steiner_tree"
