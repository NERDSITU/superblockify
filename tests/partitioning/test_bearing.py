"""Tests for the partitioner module."""
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
