"""Tests for the partitioner module."""
import pytest

from superblockify.partitioning import BearingPartitioner


class TestBearingPartitioner:
    """Tests for the BearingPartitioner."""

    # pylint: disable=protected-access
    @pytest.mark.parametrize("bin_num", [360, 500, 563, 900, 5981, 9000])
    def test_bin_bearings(self, test_city_bearing, bin_num):
        """Test `__bin_bearings` class method by design."""
        _, graph = test_city_bearing
        part = BearingPartitioner(graph)
        part._BearingPartitioner__bin_bearings(bin_num)
        assert len(part._bin_info["bin_edges"]) == bin_num + 1
        assert len(part._bin_info["bin_frequency"]) == bin_num
        assert part._bin_info["peak_ind"] is not None
        assert part._bin_info["peak_props"] is not None

    @pytest.mark.parametrize("bin_num", [359, 0, -1, -30])
    def test_bin_num_not_positive(self, test_city_bearing, bin_num):
        """Test `__bin_bearings` class method for invalid `bin_nums`."""
        _, graph = test_city_bearing
        part = BearingPartitioner(graph)
        with pytest.raises(ValueError):
            part._BearingPartitioner__bin_bearings(bin_num)
    # pylint: enable=protected-access