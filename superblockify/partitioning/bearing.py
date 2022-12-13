"""Approach relating using edge bearings."""
from bisect import bisect_right

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from scipy.signal import find_peaks

from superblockify import attribute
from .partitioner import BasePartitioner


class BearingPartitioner(BasePartitioner):
    """Bearing partitioner.

    Partitions based on the edge bearings.
    """

    def __init__(self, graph):
        """Construct a BearingPartitioner"""
        super().__init__(graph)

        self._bin_info = None
        self._inter_vals = {}

    def run(self, show_analysis_plots=False, num_bins: int = None, **kwargs):
        """Group by prominent bearing directions.

        Procedure to determine the graphs partitions based on the edges bearings.
            1. Bin bearings.
            2. Find peaks.
            3. Determine boundaries/intervals corresponding to a partition.
            4. Write partition attribute edges.

        Parameters
        ----------
        show_analysis_plots : bool, optional
            If True show visualization graphs of the approach.
            Peakfinding,
        num_bins : int >= 360, optional
            Number of bins to split the bearings into / radial resolution.

        Raises
        ------
        ArithmeticError
            If no peaks are being found.
        """

        # Determine number of bins if not passed explicitly
        if not isinstance(num_bins, int):
            num_edges = len(self.graph.edges.data(data=self.attribute_label))
            num_bins = num_edges ** (1 / 2) if num_edges ** (1 / 2) > 360 else 360

        # Binning
        self.__bin_bearings(num_bins)

        # Find peaks
        self.__find_peaks()

        if len(self._bin_info["peak_ind"]) < 1:
            raise ArithmeticError("No peaks were found.")

        if show_analysis_plots:
            self.plot_peakfinding()

        # Make boundaries
        self.__make_boundaries()

        if show_analysis_plots:
            self.plot_interval_splitting()

        # Write grouping attribute to graph
        self.attribute_label = "bearing_group"
        group_bearing = nx.get_edge_attributes(self.graph, "bearing_90")
        for node, bearing in group_bearing.items():
            if np.isnan(bearing):
                group_bearing[node] = np.nan
            else:
                i = bisect_right(self._inter_vals["boundaries"], bearing)
                group_bearing[node] = self._inter_vals["center_values"][i - 1]
        nx.set_edge_attributes(self.graph, group_bearing, self.attribute_label)

        # Write partiton dict
        self.partition = [
            {
                "name": str(self._inter_vals["boundaries"][i : i + 2]),
                "value": center_val,
            }
            for (i, center_val) in enumerate(self._inter_vals["center_values"])
            if center_val is not None
        ]

    def __bin_bearings(self, num_bins: int):
        """Construct histogram of `self.graph` bearings.

        Writes to bin_edges and bin_frequencies of `_bin_info`.

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
            raise ValueError(
                f"The number of bins needs to be greater than 360, "
                f"but is {num_bins}."
            )

        self._bin_info = {"num_bins": num_bins}

        # Write attribute where bearings are baked down modulo 90 degrees.
        attribute.new_edge_attribute_by_function(
            self.graph, lambda bear: bear % 90, "bearing", "bearing_90"
        )

        bins = (
            np.arange((self._bin_info["num_bins"] * 2) + 1)
            * 90
            / (self._bin_info["num_bins"] * 2)
        )
        count, bin_edges = np.histogram(
            list(nx.get_edge_attributes(self.graph, "bearing_90").values()), bins=bins
        )
        # move last bin to front, so eg 0.01° and 359.99° will be binned together
        count = np.roll(count, 1)
        bin_counts = count[::2] + count[1::2]
        # because we merged the bins, their edges are now only every other one
        self._bin_info["bin_edges"] = bin_edges[range(0, len(bin_edges), 2)]
        # Intensity
        self._bin_info["bin_frequency"] = bin_counts / bin_counts.sum()

    def __find_peaks(self):
        """Find peaks in the histogram of bearings.

        Writes to peak_ind and peak_values of `_bin_info`.
        TODO: Find way to automate prominence threshold. Cite reference.

        Raises
        ------
        AssertionError
            If bearing histogram is empty.
        """

        if self._bin_info is None or not all(
                name in self._bin_info
                for name in ["num_bins", "bin_edges", "bin_frequency"]
        ):
            raise AssertionError(f"{self.__class__.__name__} has not been binned yet, "
                                 f"run `__bin_bearings` before finding peaks.")

        self._bin_info["peak_ind"], self._bin_info["peak_props"] = find_peaks(
            self._bin_info["bin_frequency"],
            distance=int(0.4 * self._bin_info["num_bins"] / 90),
            # Required minimal
            # horizontal distance (>= 1) in samples between
            # neighbouring peaks.
            prominence=0.005,
        )

    def __make_boundaries(self):
        """Determine partition boundaries

        Determine boundaries based on the binned data's peaks.
        """
        # Make partitioning boundaries out of peak bases

        left_bases1 = self._bin_info["peak_props"]["left_bases"]
        right_bases1 = self._bin_info["peak_props"]["right_bases"]
        overlap_groups = self.group_overlapping_intervals(left_bases1, right_bases1)

        self._inter_vals["base_vals"] = [
            (
                self._bin_info["bin_edges"][left_bases1[i]],
                self._bin_info["bin_edges"][right_bases1[i]],
            )
            for i in range(len(left_bases1))
        ]

        # split overlapping groups by the unique bases they share
        for group in overlap_groups:
            # unique borders per group
            borders = np.sort(
                np.unique([(left_bases1[g], right_bases1[g]) for g in group])
            )
            for i, group in enumerate(group):
                self._inter_vals["base_vals"][group] = (
                    self._bin_info["bin_edges"][borders[i]],
                    self._bin_info["bin_edges"][borders[i + 1]],
                )

        self._inter_vals["base_vals"] = {
            ab: [
                self._bin_info["bin_edges"][peak_i]
                for peak_i in self._bin_info["peak_ind"]
                if ab[0] < self._bin_info["bin_edges"][peak_i] < ab[1]
            ][0]
            for ab in self._inter_vals["base_vals"]
        }

        # Make boundary and value array
        self._inter_vals["boundaries"] = [0]
        self._inter_vals["center_values"] = [None]
        for interval, value in self._inter_vals["base_vals"].items():
            if self._inter_vals["boundaries"][-1] == interval[0]:
                self._inter_vals["boundaries"].append(interval[1])
                self._inter_vals["center_values"] = self._inter_vals["center_values"][
                    :-1
                ] + [value, None]
            else:
                self._inter_vals["boundaries"].extend(interval)
                self._inter_vals["center_values"] = self._inter_vals[
                    "center_values"
                ] + [value, None]
        if self._inter_vals["boundaries"][-1] == 90:
            self._inter_vals["center_values"].pop()
        else:
            self._inter_vals["boundaries"].append(90)

    @staticmethod
    def group_overlapping_intervals(left_bases1, right_bases1):
        """Find groups of overlapping intervals"""
        mask = (left_bases1 < right_bases1[:, None]) & (
            right_bases1 > left_bases1[:, None]
        )
        # scales badly with n^2; optimizable
        overlaps = np.triu(mask, k=1).nonzero()
        overlap_groups = []
        for group_1, group_2 in tuple(zip(*overlaps)):
            if len(overlap_groups) == 0:
                overlap_groups.append({group_1, group_2})
            else:
                added = False
                for i, sublist in enumerate(overlap_groups):
                    if group_1 in sublist:
                        overlap_groups[i].add(group_2)
                        added = True
                        break
                    if group_2 in sublist:
                        overlap_groups[i].add(group_1)
                        added = True
                        break
                if not added:
                    overlap_groups.append({group_1, group_2})
        # # add missing groups that are not overlapping anything - not neccessary
        # overlap_groups += [{i} for i in range(len(left_bases1)) if i not in
        #                    [group for ol_group in overlap_groups
        #                     for group in ol_group]]
        return overlap_groups

    def plot_peakfinding(self):
        """Show the histogram and found peaks.

        Execute `run` before or plot during running (`run(show_plots=True)`).

        Raises
        ------
        AssertionError
            If peakfinding has not been done yet.

        Returns
        -------
        fig, axe : tuple
            matplotlib figure, axis

        """

        if self._bin_info is None:
            raise AssertionError(
                f"{self.__class__.__name__} has not been binned yet, "
                f"run `__bin_bearings` before plotting graph."
            )

        fig, axe = plt.subplots(figsize=(12, 8))

        self._inter_vals["max_height"] = max(
            self._bin_info["peak_props"]["prominences"]
        )
        self._inter_vals["colors"] = cm.hsv(
            self._bin_info["peak_ind"] / self._bin_info["num_bins"]
        )
        self._inter_vals["boxes"] = [
            (
                self._bin_info["bin_edges"][
                    self._bin_info["peak_props"]["left_bases"][i]
                ],
                self._bin_info["bin_edges"][
                    self._bin_info["peak_props"]["right_bases"][i]
                ]
                - self._bin_info["bin_edges"][
                    self._bin_info["peak_props"]["left_bases"][i]
                ],
            )
            for i in range(len(self._bin_info["peak_ind"]))
        ]
        axe.broken_barh(
            self._inter_vals["boxes"],
            (0, self._inter_vals["max_height"] * 0.45),
            alpha=0.2,
            facecolors=self._inter_vals["colors"],
            edgecolors=self._inter_vals["colors"],
        )

        for i in range(len(self._bin_info["peak_ind"])):
            axe.broken_barh(
                [self._inter_vals["boxes"][i]],
                (
                    self._inter_vals["max_height"]
                    * 0.5
                    * (1 + i / len(self._bin_info["peak_ind"])),
                    self._inter_vals["max_height"]
                    / (2 * len(self._bin_info["peak_ind"])),
                ),
                alpha=0.6,
                facecolors=self._inter_vals["colors"][i],
                edgecolors=self._inter_vals["colors"][i],
            )

        plt.bar(
            self._bin_info["bin_edges"][:-1],
            self._bin_info["bin_frequency"],
            width=90 / self._bin_info["num_bins"],
            # edgecolor='k'
        )
        plt.xticks([0, 15, 30, 45, 60, 75, 90])
        plt.xticks(np.linspace(0, 90, 91), minor=True)
        plt.xlabel(r"Direction ($\degree$)")
        plt.ylabel("Density")

        plt.scatter(
            [self._bin_info["bin_edges"][i] for i in self._bin_info["peak_ind"]],
            [self._bin_info["bin_frequency"][i] for i in self._bin_info["peak_ind"]],
            marker="x",
        )

        midpoints_idx = np.array(
            (self._bin_info["peak_ind"][1:] + self._bin_info["peak_ind"][:-1]) / 2,
            dtype=int,
        )

        for i in midpoints_idx:
            plt.axvline(self._bin_info["bin_edges"][i], color="black", alpha=0.25)

        plt.show()
        return fig, axe

    def plot_interval_splitting(self):
        """Plot the split up of peak bases into intervals

        Show how the peaks with their overlapping left and right bases are being
        split up into non overlapping intervals.

        Raises
        ------
        AssertionError
            If boundaries have not been partitioned yet.

        Returns
        -------
        fig, axe : tuple
            matplotlib figure, axis

        """

        if not all(
            name in self._inter_vals
            for name in ["base_vals", "boundaries", "center_values"]
        ):
            raise AssertionError(
                f"{self.__class__.__name__}'s boundaries have not been partitioned "
                f"yet, run `__make_boundaries` before plotting graph."
            )

        fig, axe = plt.subplots(figsize=(12, 8))
        for i in range(len(self._bin_info["peak_ind"])):
            axe.broken_barh(
                [self._inter_vals["boxes"][i]],
                (
                    self._inter_vals["max_height"]
                    * 0.5
                    * (i / len(self._bin_info["peak_ind"])),
                    self._inter_vals["max_height"]
                    / (2 * len(self._bin_info["peak_ind"])),
                ),
                alpha=0.6,
                facecolors=self._inter_vals["colors"][i],
                edgecolors=self._inter_vals["colors"][i],
            )
            axe.broken_barh(
                [self._inter_vals["boxes"][i]],
                (
                    0,
                    self._inter_vals["max_height"]
                    * 0.5
                    * (i / len(self._bin_info["peak_ind"])),
                ),
                alpha=0.1,
                facecolors=self._inter_vals["colors"][i],
                edgecolors=self._inter_vals["colors"][i],
            )
        for i, interval in enumerate(self._inter_vals["base_vals"]):
            col = cm.hsv(sum(interval) / 180)
            axe.broken_barh(
                [(interval[0], interval[1] - interval[0])],
                (
                    -self._inter_vals["max_height"]
                    / (2 * len(self._bin_info["peak_ind"])),
                    self._inter_vals["max_height"]
                    / (2 * len(self._bin_info["peak_ind"])),
                ),
                alpha=0.6,
                facecolors=col,
                edgecolors=col,
            )
            axe.broken_barh(
                [(interval[0], interval[1] - interval[0])],
                (
                    -self._inter_vals["max_height"]
                    * (2 + i)
                    / (2 * len(self._bin_info["peak_ind"])),
                    self._inter_vals["max_height"]
                    / (2 * len(self._bin_info["peak_ind"])),
                ),
                alpha=0.6,
                facecolors=col,
                edgecolors=col,
            )

        plt.show()
        return fig, axe
