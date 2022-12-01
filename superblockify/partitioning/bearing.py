"""Approach relating using edge bearings."""
import networkx as nx
import numpy as np
from scipy.signal import find_peaks

from superblockify.partitioning import BasePartitioner
from superblockify import attribute


class BearingPartitioner(BasePartitioner):
    """Bearing partitioner.

    Partitions based on the edge bearings.
    """

    def __init__(self, graph):
        """Construct a BearingPartitioner"""
        super().__init__(graph)

        self._bin_info = None

    def run(self):
        """Group by prominent bearing directions"""

        # Binning
        self.__bin_bearings(9000)

        self.attribute_label = "bearing_group"



        # # Somehow determining the partition of edges
        # # - edges also may not be included in any partition and miss the label
        # values = list(range(3))
        # attribute.new_edge_attribute_by_function(
        #     self.graph, lambda bear: choice(values), "osmid", self.attribute_label
        # )
        #
        # # A List of the existing partitions, the 'value' attribute should be equal to
        # # the edge attributes under the instances `attribute_label`, which belong to
        # # this partition
        # self.partition = [{"name": str(num), "value": num} for num in values]

    def __bin_bearings(self, num_bins: int):
        """Construct histogram of `self.graph` bearings.

        Parameters
        ----------
        num_bins : int, >= 360
            Number of bins to split the bearings into / radial resolution.

        Raises
        ------
        ValueError
            If `num_bins` is < 360.
        """

        if num_bins < 360:
            raise ValueError(f"The number of bins needs to be greater than 360, "
                             f"but is {num_bins}.")

        self._bin_info = {}

        # Write attribute where bearings are baked down modulo 90 degrees.
        attribute.new_edge_attribute_by_function(
            self.graph, lambda bear: bear % 90, "bearing", "bearing_90"
        )

        bins = np.arange((num_bins * 2) + 1) * 90 / (num_bins * 2)
        count, bin_edges = np.histogram(
            list(nx.get_edge_attributes(self.graph, 'bearing_90').values()), bins=bins)
        # move last bin to front, so eg 0.01° and 359.99° will be binned together
        count = np.roll(count, 1)
        bin_counts = count[::2] + count[1::2]
        # because we merged the bins, their edges are now only every other one
        self._bin_info["bin_edges"] = bin_edges[range(0, len(bin_edges), 2)]
        # Intensity
        self._bin_info["bin_frequency"] = bin_counts / bin_counts.sum()
        # Find peaks
        self._bin_info["peak_ind"], self._bin_info["peak_props"] = find_peaks(
            self._bin_info["bin_frequency"],
            distance=int((0.4) * num_bins / 90),
            # Required minimal
            # horizontal distance (>= 1) in samples between
            # neighbouring peaks.
            prominence=0.0003,
        )
