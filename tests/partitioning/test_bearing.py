"""Tests for the partitioner module."""
from itertools import product

import numpy as np
import pytest

from superblockify.partitioning import BearingPartitioner


class TestBearingPartitioner:
    """Tests for the BearingPartitioner."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("bin_num", [360, 500, 563, 900, 5981, 9000])
    def test_bin_bearings(self, test_city_bearing, bin_num):
        """Test `__bin_bearings` class method by design."""
        city_name, graph = test_city_bearing
        part = BearingPartitioner(graph, name=city_name)
        part._BearingPartitioner__bin_bearings(bin_num)
        assert part._bin_info["num_bins"] == bin_num
        assert len(part._bin_info["bin_edges"]) == bin_num + 1
        assert len(part._bin_info["bin_frequency"]) == bin_num
        assert "peak_ind" not in part._bin_info
        assert "peak_props" not in part._bin_info

    @pytest.mark.parametrize("bin_num", [359, 0, -1, -30])
    def test_bin_num_not_positive(self, test_city_bearing, bin_num):
        """Test `__bin_bearings` class method for invalid `bin_nums`."""
        _, graph = test_city_bearing
        part = BearingPartitioner(graph)
        with pytest.raises(ValueError):
            part._BearingPartitioner__bin_bearings(bin_num)

    def test_find_peaks_missing_binning(self, test_city_bearing):
        """Test `find_peaks` class method without binning."""
        _, graph = test_city_bearing
        part = BearingPartitioner(graph)
        with pytest.raises(AssertionError):
            part._BearingPartitioner__find_peaks()

    # pylint: enable=protected-access

    def test_plot_peakfinding_missing_peakfinding(self, test_city_bearing):
        """Test `plot_peakfinding` class method with missing peakfinding."""
        _, graph = test_city_bearing
        part = BearingPartitioner(graph)
        with pytest.raises(AssertionError):
            part.plot_peakfinding()

    def test_plot_interval_splitting_missing_peakfinding(self, test_city_bearing):
        """Test `plot_interval_splitting` class method without partitioned bounds."""
        _, graph = test_city_bearing
        part = BearingPartitioner(graph)
        with pytest.raises(AssertionError):
            part.plot_interval_splitting()

    @pytest.mark.parametrize(
        "left_bases,right_bases,overlapping_sets",
        [
            ([], [], []),
            ([0], [0], []),
            ([0, 4.0], [0, 5.0], []),
            ([-20, 0], [0, 20], []),
            ([-20.0, 0.0], [0.0, 20.0], []),
            ([-20.0, 0.0], [0.1, 20.0], [{0, 1}]),
            ([-20.0, 0.0, -300], [0.1, 20.0, 300], [{0, 1, 2}]),
            ([-20.0, 0.0, -300, 2000], [0.1, 20.0, 300, 3000], [{0, 1, 2}]),
            (
                [-20.0, 0.0, -300, 2000, -1e-20],
                [0.1, 20.0, 300, 3000, -1e-21],
                [{0, 1, 2, 4}],
            ),
            ([0, 2, 5, 7], [5, 3, 10, 11], [{0, 1}, {2, 3}]),
            ([5, 7, 0, 2], [10, 11, 5, 3], [{0, 1}, {2, 3}]),
            ([0, 5, 2, 7], [5, 10, 3, 11], [{0, 2}, {1, 3}]),
            ([0, 5, 2, 7, 15], [5, 10, 3, 11, 20], [{0, 2}, {1, 3}]),
            ([0, 5, 2, 7, -4], [5, 10, 3, 11, 1], [{0, 2, 4}, {1, 3}]),
            ([0, 0, 1, 3, 4, 9], [1, 2, 3, 4, 10, 10], [{0, 1, 2}, {4, 5}]),
            ([0.0, 0, 1, 3, 4.0, 9], [1, 2.0, 3, 4.0, 10.0, 10], [{0, 1, 2}, {4, 5}]),
            ([0, 0, 2, 3, 0], [1, 2, 4, 4, 4], [{0, 1, 2, 3, 4}]),
            ([0, 0, 3, 4, 0], [1, 2, 5, 5, 5], [{0, 1, 2, 3, 4}]),
            ([3, 4, 0, 0, 0], [5, 5, 1, 2, 5], [{0, 1, 2, 3, 4}]),
            ([0, 0, 3, 4, 0, 6, 7], [1, 2, 5, 5, 5, 40, 12], [{0, 1, 2, 3, 4}, {5, 6}]),
            ([0, 0, 3, 4, 0, 6, 7], [1, 2, 5, 5, 7, 40, 12], [{0, 1, 2, 3, 4, 5, 6}]),
            ([0, 0, 3, 4, 2, 6, 7], [1, 2, 5, 5, 7, 40, 12], [{0, 1}, {2, 3, 4, 5, 6}]),
            (
                [85, 85, 85, 85, 112, 112, 124, 126, 136, 85],
                [91, 101, 104, 110, 118, 146, 146, 146, 146, 260],
                [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}],
            ),
        ],
    )
    def test_group_overlapping_intervals(
        self, left_bases, right_bases, overlapping_sets
    ):
        """Test `group_overlapping_intervals` static class method by design."""
        left_bases, right_bases = np.array(left_bases), np.array(right_bases)
        assert (
            BearingPartitioner.group_overlapping_intervals(left_bases, right_bases)
            == overlapping_sets
        )

    @pytest.mark.parametrize(
        "left_bases,right_bases",
        [
            (1, 2),
            (np.array([1]), 2),
            (1, np.array([2])),
            (None, np.array([2])),
            ("str", np.array([2])),
            (0.0, np.array([2])),
            ({2}, np.array([2])),
        ],
    )
    def test_group_overlapping_intervals_type_mismatch(self, left_bases, right_bases):
        """Test `group_overlapping_intervals` static class method for type mismatch."""
        with pytest.raises(TypeError):
            BearingPartitioner.group_overlapping_intervals(left_bases, right_bases)

    @pytest.mark.parametrize(
        "left_bases,right_bases",
        [
            ([0, 0, 1, 3, 4], [1, 2, 3, 4, 10, 10]),
            ([2], []),
            ([], [2]),
            ([-60, 3], []),
            (
                [-60, 3],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
        ],
    )
    def test_group_overlapping_intervals_length_mismatch(self, left_bases, right_bases):
        """Test `group_overlapping_intervals` static class method
        for length mismatch.
        """
        left_bases, right_bases = np.array(left_bases), np.array(right_bases)
        with pytest.raises(ValueError):
            BearingPartitioner.group_overlapping_intervals(left_bases, right_bases)

    @pytest.mark.parametrize("left_right_bases_ndim", list(product(range(6), repeat=2)))
    def test_group_overlapping_intervals_ndim_mismatch(self, left_right_bases_ndim):
        """Test `group_overlapping_intervals` static class method for ndim mismatch."""
        if left_right_bases_ndim != (1, 1):
            left_bases = np.empty((1,) * left_right_bases_ndim[0])
            right_bases = np.empty((1,) * left_right_bases_ndim[1])
            with pytest.raises(TypeError):
                BearingPartitioner.group_overlapping_intervals(left_bases, right_bases)

    @pytest.mark.parametrize(
        "test_input,test_output",
        [
            ([], []),
            ([{0}], [{0}]),
            ([{0}, {1}], [{0}, {1}]),
            ([{0}, {1}, {2}], [{0}, {1}, {2}]),
            ([{0}, {0}, {2}], [{0}, {2}]),
            ([{0}, {0}, {0}], [{0}]),
            ([{0, 1, 2}], [{0, 1, 2}]),
            ([{0, 1, 2}, {3}], [{0, 1, 2}, {3}]),
            ([{0, 1, 2}, {3, 4}], [{0, 1, 2}, {3, 4}]),
            ([{1, 2}, {2, 3}, {4, 5}], [{1, 2, 3}, {4, 5}]),
            ([{1, 2}, {2, 3}, {4, 5}, {6, 7}], [{1, 2, 3}, {4, 5}, {6, 7}]),
            ([{1, 2}, {2, 3}, {4, 5}, {1, 2, 3}], [{1, 2, 3}, {4, 5}]),
        ],
    )
    def test_merge_sets(self, test_input, test_output):
        """Test `merge_sets` static class method by design."""
        assert BearingPartitioner.merge_sets(test_input) == test_output

    @pytest.mark.parametrize(
        "test_input",
        [
            1,  # not a list
            None,  # not a list
            "str",  # not a list
            0.0,  # not a list
            ({2}),  # not a list
            ([1, 2]),  # not a list of sets
            ([{1, 2}, 2]),  # not a list of sets
            ([{1, 2}, {2, 3}, {4, 5}, {6, 7}, 8]),  # not a list of sets
        ],
    )
    def test_merge_sets_type_mismatch(self, test_input):
        """Test `merge_sets` static class method for type mismatch."""
        with pytest.raises(TypeError):
            BearingPartitioner.merge_sets(test_input)

    # pylint: disable=protected-access
    @pytest.mark.parametrize(
        "boundaries_input,center_values_input,boundaries_output,center_values_output",
        [
            ([0.0], [], [0.0], []),
            ([0.0, 1.0], [None], [0.0, 1.0], [None]),
            ([0.0, 1.0], [0.0], [0.0, 1.0], [0.0]),
            ([0.0, 1.0], [0.5], [0.0, 1.0], [0.5]),
            ([0.0, 1.0], [1.0], [0.0, 1.0], [1.0]),
            ([-3, 1.0], [0.0], [-3, 1.0], [0.0]),
            ([0.0, 1.0, 2.0], [0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 1.0]),
            ([0, 1, 2, 3], [0.5, 1.5, 2.5], [0, 1, 2, 3], [0.5, 1.5, 2.5]),
            ([0, 1, 0, 1], [0.2, None, 0.8], [0, 1], [np.mean([0.2, 0.8])]),
            ([0, 1, 0, 1], [0.8, None, 0.2], [0, 1], [np.mean([0.8, 0.2])]),
            ([0, 1, 0, 1], [0.0, None, 0.8], [0, 1], [np.mean([0.0, 0.8])]),
            ([0, 1, 0, 1], [0.5, None, 0.4], [0, 1], [np.mean([0.5, 0.4])]),
            (
                [-1, 0, 1, 0, 1],
                [-0.5, 0.5, None, 0.4],
                [-1, 0, 1],
                [-0.5, np.mean([0.5, 0.4])],
            ),
            (
                [0, 1, 0, 1, 0, 1],
                [0.2, None, 0.8, None, 0.7],
                [0, 1],
                [np.mean([0.2, 0.8, 0.7])],
            ),
            (
                [0, 2, 0, 2, 0, 2, 0, 2],
                [0.2, None, 0.8, None, 0.7, None, 1.8],
                [0, 2],
                [np.mean([0.2, 0.8, 0.7, 1.8])],
            ),
            (
                [0, 1, 0, 1, 0, 1, 3, 4],
                [0.2, None, 0.8, None, 0.7, 2.0, 3.5],
                [0, 1, 3, 4],
                [np.mean([0.2, 0.8, 0.7]), 2.0, 3.5],
            ),
            (
                [0, 1, 0, 1, 0, 1, 3, 4],
                [0.2, None, 0.8, None, 0.7, None, 3.5],
                [0, 1, 3, 4],
                [np.mean([0.2, 0.8, 0.7]), None, 3.5],
            ),
            (
                [0, 1, 0, 1, 0, 1, 3, 4, 3, 4],
                [0.2, None, 0.8, None, 0.7, 2.0, 3.5, None, 3.6],
                [0, 1, 3, 4],
                [np.mean([0.2, 0.8, 0.7]), 2.0, np.mean([3.5, 3.6])],
            ),
            (
                [0, 1, 0, 1, 0, 1, 3, 4, 3, 4],
                [0.2, None, 0.8, None, 0.7, None, 3.5, None, 3.6],
                [0, 1, 3, 4],
                [np.mean([0.2, 0.8, 0.7]), None, np.mean([3.5, 3.6])],
            ),
            (
                [-3, 0, 1, 0, 1, 0, 1, 3, 4, 3, 4],
                [-1, 0.2, None, 0.8, None, 0.7, None, 3.5, None, 3.6],
                [-3, 0, 1, 3, 4],
                [-1, np.mean([0.2, 0.8, 0.7]), None, np.mean([3.5, 3.6])],
            ),
        ],
    )
    def test_find_and_merge_intervals(
        self,
        boundaries_input,
        center_values_input,
        boundaries_output,
        center_values_output,
    ):
        """Test `find_and_merge_intervals` static class method by design."""
        assert BearingPartitioner._BearingPartitioner__find_and_merge_intervals(
            boundaries_input, center_values_input
        ) == (boundaries_output, center_values_output)

    # pylint: disable=protected-access

    @pytest.mark.parametrize(
        "boundaries_input,center_values_input",
        [
            (["a"], []),  # not a list of floats
            (1, [0.0]),  # not a list of floats
            ([0.0], 1),  # not a list of floats
            (None, [0.0]),  # not a list of floats
            ([0, 1, 0, 1], [0.2, None, None]),  # center values cannot be None
        ],
    )
    def test_find_and_merge_intervals_type_mismatch(
        self, boundaries_input, center_values_input
    ):
        """Test `find_and_merge_intervals` static class method for type mismatch."""
        with pytest.raises(TypeError):
            BearingPartitioner._BearingPartitioner__find_and_merge_intervals(
                boundaries_input, center_values_input
            )

    @pytest.mark.parametrize(
        "boundaries_input,center_values_input",
        [
            ([], []),  # length mismatch
            ([0.0], [0.0]),  # length mismatch
            ([0.0, 1.0], [0.0, 1.0]),  # length mismatch
            ([0.0, 1.0, 2.0], [0.0]),  # length mismatch
            ([0.0, 1.0, 2.0, 4.0], [0.0, 1.0]),  # length mismatch
            ([0.0], [0.0, 1.0]),  # length mismatch
        ],
    )
    def test_find_and_merge_intervals_length_mismatch(
        self, boundaries_input, center_values_input
    ):
        """Test `find_and_merge_intervals` static class method for length mismatch."""
        with pytest.raises(AssertionError):
            BearingPartitioner._BearingPartitioner__find_and_merge_intervals(
                boundaries_input, center_values_input
            )

    # pylint: enable=protected-access
