"""Approach relating using edge bearings."""

from bisect import bisect_right
from typing import List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from networkx import weakly_connected_components
from scipy.signal import find_peaks
from typing_extensions import deprecated

from ..base import BasePartitioner
from ... import attribute
from ...config import logger, Config
from ...plot import save_plot


@deprecated("BearingPartitioner does not necessarily produce a valid partitioning.")
class BearingPartitioner(BasePartitioner):  # pragma: no cover
    """Bearing partitioner.

    Partitions based on the edge bearings.
    """  # pylint: disable=too-many-instance-attributes

    def __init__(self, *args, **kwargs):
        """Construct a BearingPartitioner"""
        super().__init__(*args, **kwargs)

        self._bin_info = None
        self._inter_vals = {}
        self.attr_value_minmax = (0, 90)

        self.residential_graph = self.graph.copy()
        self.residential_graph.remove_edges_from(
            [
                (u, v)
                for u, v, d in self.residential_graph.edges(data=True)
                if d["highway"] != "residential" or "residential" not in d["highway"]
            ]
        )

    def partition_graph(
        self, make_plots=False, min_length=500, min_edge_count=5, **kwargs
    ):
        """Group by prominent bearing directions.

        Procedure to determine the graphs partitions based on the edges bearings.
            1. Bin bearings. Only on residential edges.
            2. Find peaks.
            3. Determine boundaries/intervals corresponding to a partition.
            4. (optional) Plot found peaks and interval splits.
            5. Write partition attribute edges.

        The number of bins is fixed to 360°, as rules for numbers of bins [1]_ ask
        for much lower numbers of bins, for common number of edges in a street network
        (approx. 300 to 60.000 edges), which would produce a too small resolution
        (Sturges' formula: 10 to 17 bins; Square-root choice: 18 to 245 bins).

        Parameters
        ----------
        make_plots : bool, optional
            If True show visualization graphs of the approach.
        min_length : float, optional
            Minimum component length in meters to be considered for partitioning.
        min_edge_count : int, optional
            Minimum component edge count to be considered for partitioning.

        Raises
        ------
        ArithmeticError
            If no peaks are being found.

        References
        ----------
        .. [1] Wikipedia contributors, "Histogram," Wikipedia, The Free Encyclopedia,
           https://en.wikipedia.org/w/index.php?title=Histogram&oldid=1113935482
           (accessed December 14, 2022).

        """

        # Binning
        self.__bin_bearings(num_bins=360)

        # Find peaks
        self.__find_peaks()

        if len(self._bin_info["peak_ind"]) < 1:
            raise ArithmeticError("No peaks were found.")

        if make_plots:
            fig, _ = self.plot_peakfinding()
            save_plot(
                self.results_dir, fig, f"{self.name}_peakfinding.{Config.PLOT_SUFFIX}"
            )

        # Make boundaries
        self.__make_boundaries()

        if make_plots:
            fig, _ = self.plot_interval_splitting()
            save_plot(
                self.results_dir,
                fig,
                f"{self.name}_interval_splitting.{Config.PLOT_SUFFIX}",
            )

        # Write grouping attribute to graph
        self.attribute_label = "bearing_group"
        self.attribute_dtype = str
        group_bearing = nx.get_edge_attributes(self.residential_graph, "bearing_90")
        logger.debug("Writing attribute 'bearing_group' to graph.")
        for edge, bearing in group_bearing.items():
            if np.isnan(bearing):  # pragma: no cover
                group_bearing[edge] = np.nan
            else:
                i = bisect_right(self._inter_vals["boundaries"], bearing)
                group_bearing[edge] = self._inter_vals["center_values"][i - 1]
        nx.set_edge_attributes(self.graph, group_bearing, self.attribute_label)

        # Write partition dict
        self.partitions = [
            {
                "name": str(self._inter_vals["boundaries"][i : i + 2]),
                "value": center_val,
            }
            for (i, center_val) in enumerate(self._inter_vals["center_values"])
            if center_val is not None
        ]

        # Make subgraphs for each partition
        # Overwrite `self.attribute_label` so residential edges are not added to
        # subgraphs.
        for node1, node2, key in self.graph.edges(keys=True):
            # check if edge is in self.residential_graph
            if (node1, node2) not in self.residential_graph.edges:
                self.graph.edges[node1, node2, key][self.attribute_label] = None

        self.make_subgraphs_from_attribute(
            split_disconnected=True,
            min_edge_count=min_edge_count,
            min_length=min_length,
        )

        # Make sparsified graph
        self.set_sparsified_from_components()
        # Produces graph that may be not connected, so we only take the LCC.
        self.sparsified = self.graph.subgraph(
            max(weakly_connected_components(self.sparsified), key=len)
        )

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
                f"The number of bins needs to be greater or equal than 360, "
                f"but is {num_bins}."
            )

        self._bin_info = {"num_bins": num_bins}

        # Write attribute where bearings are baked down modulo 90 degrees.
        attribute.new_edge_attribute_by_function(
            self.residential_graph, lambda bear: bear % 90, "bearing", "bearing_90"
        )

        bins = (
            np.arange((self._bin_info["num_bins"] * 2) + 1)
            * 90
            / (self._bin_info["num_bins"] * 2)
        )
        count, bin_edges = np.histogram(
            list(nx.get_edge_attributes(self.residential_graph, "bearing_90").values()),
            bins=bins,
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

        `scipy.signal.find_peaks` is used to find the peaks and configured as follows:
        - height: minimum height of a peak: mean of the histogram
        - prominence: minimum prominence of a peak: one standard deviation
        - distance: minimum distance between two peaks: >= 0.4°

        Raises
        ------
        AssertionError
            If bearing histogram is empty.
        """

        if self._bin_info is None or not all(
            name in self._bin_info
            for name in ["num_bins", "bin_edges", "bin_frequency"]
        ):
            raise AssertionError(
                f"{self.__class__.__name__} has not been binned yet, "
                f"run `__bin_bearings` before finding peaks."
            )

        # Add general info about histogram: mean, median, std, min, max
        self._bin_info["mean"] = np.mean(self._bin_info["bin_frequency"])
        self._bin_info["medi"] = np.median(self._bin_info["bin_frequency"])
        self._bin_info["std"] = np.std(self._bin_info["bin_frequency"])
        self._bin_info["min"] = np.min(self._bin_info["bin_frequency"])
        self._bin_info["max"] = np.max(self._bin_info["bin_frequency"])

        self._bin_info["peak_ind"], self._bin_info["peak_props"] = find_peaks(
            self._bin_info["bin_frequency"],
            height=self._bin_info["mean"],
            # Required minimal height of peaks.
            prominence=self._bin_info["std"],
            # Required minimal prominence of peaks.
            distance=int(0.4 * self._bin_info["num_bins"] / 90),
            # Required minimal horizontal distance (>= 0.4°) between neighbouring peaks.
        )

    def __make_boundaries(self):
        """Determine partition boundaries.

        Determine boundaries based on the binned data's peaks.

        Warnings
        --------
        If left and right bases have identical intervals.
        """
        # Find bases left and right of peak indices
        # Union of all bases
        bases = set(self._bin_info["peak_props"]["left_bases"]) | set(
            self._bin_info["peak_props"]["right_bases"]
        )
        # For peak indices, find the closest base to the left and right.
        # That is for each self._bin_info["peak_ind"] the index with the biggest
        # value, but smaller than the peak index, and the index with the smallest
        # value, but bigger than the peak index.
        logger.debug("Base indices: %s", bases)
        logger.debug("Peak indices: %s", self._bin_info["peak_ind"])
        left_right_bases = [
            (max(b for b in bases if b < p), min(b for b in bases if b > p))
            for p in self._bin_info["peak_ind"]
        ]
        logger.debug("Left and right bases: %s", left_right_bases)
        # Check if there are identical base pairs in left_right_bases.
        if len(left_right_bases) != len(set(left_right_bases)):
            logger.warning(
                "There are identical base pairs in left_right_bases. "
                "This means that there are two peaks with identical intervals."
            )

        left_right_bases_values = [
            (self._bin_info["bin_edges"][l], self._bin_info["bin_edges"][r])
            for (l, r) in left_right_bases
        ]

        # Make boundary and value lists
        # So that for all values in [0, 90[ a center value is defined.
        # For all intervals without a peak, the center value is `None`.
        peak_values = [
            self._bin_info["bin_edges"][p] for p in self._bin_info["peak_ind"]
        ]
        self._inter_vals["boundaries"] = [0]
        self._inter_vals["center_values"] = [None]
        for (l_val, r_val), value in zip(left_right_bases_values, peak_values):
            if self._inter_vals["boundaries"][-1] == l_val:
                self._inter_vals["boundaries"].append(r_val)
                self._inter_vals["center_values"] = self._inter_vals["center_values"][
                    :-1
                ] + [value, None]
            else:
                self._inter_vals["boundaries"].extend((l_val, r_val))
                self._inter_vals["center_values"] = self._inter_vals[
                    "center_values"
                ] + [value, None]
        if self._inter_vals["boundaries"][-1] == 90:
            self._inter_vals["center_values"].pop()
        else:
            self._inter_vals["boundaries"].append(90)

        # Find and merge repeated/overlaying intervals.
        (
            self._inter_vals["boundaries"],
            self._inter_vals["center_values"],
        ) = self.__find_and_merge_intervals(
            self._inter_vals["boundaries"], self._inter_vals["center_values"]
        )

        # Log the boundaries and center values
        logger.debug("Boundaries: %s", self._inter_vals["boundaries"])
        logger.debug("Center values: %s", self._inter_vals["center_values"])

        # For plotting
        self._inter_vals["base_vals"] = dict(
            zip(left_right_bases_values, self._bin_info["peak_ind"])
        )

    @staticmethod
    def __find_and_merge_intervals(boundaries, center_values):
        """Find and merge duplicate intervals.

        Find repeated intervals where boundaries violate strict monotone increasing
        order and have the same start and end value. In the center values for each
        repeated interval there is the center value and a `None` value. The `None`
        values are removed and the repeated intervals are merged, by taking the
        arithmetic mean of their center values.

        Parameters
        ----------
        boundaries : list of float or int
            List of boundaries of intervals.
        center_values : list of float or int or None
            List of center values of intervals. If there is no center value, `None`.

        Returns
        -------
        list of float or int
            List of boundaries of intervals.
        list of float or int or None
            List of center values of intervals. If there is no center value, `None`.

        Raises
        ------
        AssertionError
            If the length of `boundaries` is not one more than the length of
            `center_values`.
        TypeError
            If `boundaries` is not a numerical list.
        TypeError
            If `center_values` is not a list filled with numerical values or `None`.
            For repeated intervals the center values cannot be `None`.
        """

        # Type check
        if not isinstance(boundaries, list):
            raise TypeError("Boundaries must be a list.")
        if not isinstance(center_values, list):
            raise TypeError("Center values must be a list.")
        if not all(isinstance(b, (int, float)) for b in boundaries):
            raise TypeError(
                "The values in boundaries must be of type int (excluding bool) or "
                "float."
            )  # As bool is a subclass of int, exclude bool explicitly.
        if not all(
            isinstance(c, (int, float, type(None))) and not isinstance(c, bool)
            for c in center_values
        ):
            raise TypeError(
                "The values in center_values must be of type int, float, and None."
            )

        # Length check
        if len(boundaries) != len(center_values) + 1:
            raise AssertionError(
                "The length of `boundaries` is not one more than the length of "
                "`center_values`."
            )

        # Find indices of repeated intervals and their center values
        # If two consecutive boundary pairs are equal, the interval is repeated.
        indices = {
            i: {
                "boundaries": [boundaries[i], boundaries[i + 1]],
                "center_values": center_values[i],
            }
            for i in range(len(boundaries) - 1)
            if boundaries[i : i + 2] == boundaries[i + 2 : i + 4]
            or boundaries[i : i + 2] == boundaries[i - 2 : i]
        }

        # Merge the repeated intervals by unioning their boundaries and taking the
        # arithmetic mean of their center values.
        unique_boundaries = np.unique(
            [indices[i]["boundaries"] for i in indices], axis=0
        ).tolist()
        repeating_intervals = [
            {
                "interval": interval,
                "indices": [i for i in indices if indices[i]["boundaries"] == interval],
            }
            for interval in unique_boundaries
        ]
        # Add the center values to the repeating intervals
        for interval in repeating_intervals:
            interval["center_values"] = [center_values[i] for i in interval["indices"]]

        # Throw an error if there are intervals with where numerical and None values
        # are mixed.
        for interval in repeating_intervals:
            if any(c is None for c in interval["center_values"]) and any(
                c is not None for c in interval["center_values"]
            ):
                raise TypeError(
                    "The center values of intervals cannot be mixed with None values."
                )

        # Drop repeating intervals with center values all None
        repeating_intervals = [
            interval
            for interval in repeating_intervals
            if any(c is not None for c in interval["center_values"])
        ]

        # Sort repeating intervals by indices, descending
        repeating_intervals = sorted(
            repeating_intervals, key=lambda x: x["indices"][0], reverse=True
        )

        # From high to low, replace the boundaries and center values of the repeating
        # intervals with the arithmetic mean of their center values. Also remove the
        # None values.
        for interval in repeating_intervals:
            boundaries = (
                boundaries[: interval["indices"][0]]
                + interval["interval"]
                + boundaries[interval["indices"][-1] + 2 :]
            )
            center_values = (
                center_values[: interval["indices"][0]]
                + [np.mean(interval["center_values"])]
                + center_values[interval["indices"][-1] + 1 :]
            )

        return boundaries, center_values

    @staticmethod
    def group_overlapping_intervals(left_bases, right_bases):
        """Find groups of overlapping intervals.

        *Unused at the moment.*

        Parameters
        ----------
        left_bases : numpy.array
            List of left bases of intervals.
        right_bases : numpy.array
            List of right bases of intervals.

        Returns
        -------
        list
            List of sets of overlapping intervals.

        Raises
        ------
        TypeError
            If `left_bases` or `right_bases1` are not numpy arrays of 1d shape.
        ValueError
            If `left_bases` and `right_bases` are not of the same length.

        Examples
        --------
        >>> left_bases = [0, 0, 1, 3, 4, 9]
        >>> right_bases1 = [1, 2, 3, 4, 10, 10]
        >>> group_overlapping_intervals(left_bases, right_bases)
        [{0, 1, 2}, {4, 5}]

        """

        if not all(isinstance(arr, np.ndarray) for arr in [left_bases, right_bases]):
            raise TypeError(
                f"Input lists must be 1d numpy arrays, "
                f"got types {type(left_bases)} and {type(right_bases)}."
            )
        if not all(arr.ndim == 1 for arr in [left_bases, right_bases]):
            raise TypeError(
                f"Input lists must be of 1d shape, "
                f"got shapes {left_bases.shape} and {right_bases.shape}."
            )
        if len(left_bases) != len(right_bases):
            raise ValueError(
                f"Input lists must be of the same length, "
                f"length of left_bases is {len(left_bases)} and "
                f"length of right_bases is {len(right_bases)}."
            )

        mask = (left_bases < right_bases[:, None]) & (right_bases > left_bases[:, None])
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

        # Merge overlap groups that were split by unique bases
        overlap_groups = BearingPartitioner.merge_sets(overlap_groups)

        # # add missing groups that are not overlapping anything - not neccessary
        # overlap_groups += [{i} for i in range(len(left_bases)) if i not in
        #                    [group for ol_group in overlap_groups
        #                     for group in ol_group]]
        return overlap_groups

    @staticmethod
    def merge_sets(sets: List[Set]) -> List[Set]:
        """Merge sets that share at least one element.

        Parameters
        ----------
        sets : list of sets
            List of sets to merge.

        Returns
        -------
        list of sets
            List of merged sets.

        Raises
        ------
        TypeError
            If `sets` is not a list of sets.

        Examples
        --------
        >>> sets = [{1, 2}, {2, 3}, {4, 5}]
        >>> merge_sets(sets)
        [{1, 2, 3}, {4, 5}]

        >>> sets = [{'a', 'b'}, {'m', 3.4}, {'b', 'c'}]
        >>> merge_sets(sets)
        [{'a', 'b', 'c'}, {'m', 3.4}]

        """

        if not isinstance(sets, list):
            raise TypeError(f"Input must be a list, got {type(sets)}.")
        if not all(isinstance(s, set) for s in sets):
            raise TypeError(f"Input must be a list of sets, got {type(sets)}.")

        for i, group in enumerate(sets):
            for j, group2 in enumerate(sets):
                if i != j and group & group2:
                    sets[i] = group | group2
                    sets.pop(j)
        return sets

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

        if self._bin_info is None or not all(
            name in self._bin_info
            for name in [
                "num_bins",
                "bin_edges",
                "bin_frequency",
                "peak_ind",
                "peak_props",
                "mean",
                "medi",
                "std",
                "min",
                "max",
            ]
        ):
            raise AssertionError(
                f"{self.__class__.__name__} has not been binned yet, "
                f"run `__bin_bearings` and `__find_peaks` before plotting graph."
            )

        # Setup figure and axis
        fig, axe = plt.subplots(figsize=(12, 8))

        # Max peak height
        self._inter_vals["max_height"] = max(
            self._bin_info["peak_props"]["prominences"]
        )
        # Color peaks with colormap
        self._inter_vals["colors"] = cm.hsv(
            self._bin_info["peak_ind"] / self._bin_info["num_bins"]
        )
        # Construct list of peak boxes from left and right bases
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
        # Draw list of overlapping boxes behind lower half histogram
        axe.broken_barh(
            self._inter_vals["boxes"],
            (0, self._inter_vals["max_height"] * 0.45),
            alpha=0.2,
            facecolors=self._inter_vals["colors"],
            edgecolors=self._inter_vals["colors"],
        )
        # Draw list of non-overlapping boxes behind upper half histogram
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

        # Draw histogram
        plt.bar(
            self._bin_info["bin_edges"][:-1],
            self._bin_info["bin_frequency"],
            width=90 / self._bin_info["num_bins"],
            # edgecolor='k'
            alpha=0.8,
        )

        # Draw horizontal lines for min, max, mean, median and std
        l_min = axe.axhline(
            self._bin_info["min"], color="mediumblue", linestyle="--", linewidth=1
        )
        l_max = axe.axhline(
            self._bin_info["max"], color="crimson", linestyle="--", linewidth=1
        )
        l_mean = axe.axhline(
            self._bin_info["mean"], color="k", linestyle="--", linewidth=1.5
        )
        l_median = axe.axhline(
            self._bin_info["medi"], color="k", linestyle="--", linewidth=1
        )
        l_std_up = axe.axhline(
            self._bin_info["mean"] + self._bin_info["std"],
            color="orange",
            linestyle="--",
            linewidth=1,
        )
        axe.axhline(
            self._bin_info["mean"] - self._bin_info["std"],
            color="orange",
            linestyle="--",
            linewidth=1,
        )

        # Mark peaks with x
        plt.scatter(
            [self._bin_info["bin_edges"][i] for i in self._bin_info["peak_ind"]],
            [self._bin_info["bin_frequency"][i] for i in self._bin_info["peak_ind"]],
            marker="x",
        )

        # Calculate midpoints between peaks
        midpoints_idx = np.array(
            (self._bin_info["peak_ind"][1:] + self._bin_info["peak_ind"][:-1]) / 2,
            dtype=int,
        )
        # Draw midpoints
        for i in midpoints_idx:
            plt.axvline(self._bin_info["bin_edges"][i], color="black", alpha=0.25)

        # Show custom legend with min, max, mean, median and std in scientific notation
        axe.legend(
            [l_min, l_max, l_mean, l_std_up, l_median],
            [
                f"Min: {self._bin_info['min']:.2e}",
                f"Max: {self._bin_info['max']:.2e}",
                f"Mean: {self._bin_info['mean']:.2e}",
                f"$\\pm$Std: {self._bin_info['std']:.2e}",
                f"Median: {self._bin_info['medi']:.2e}",
            ],
            loc="upper left",
        )

        plt.xticks([0, 15, 30, 45, 60, 75, 90])
        plt.xticks(np.linspace(0, 90, 91), minor=True)
        plt.xlabel(r"Direction ($\degree$)")
        plt.ylabel("Density")
        plt.title(f"Bearing histogram of {self.name}")

        return fig, axe

    def plot_interval_splitting(self):
        """Plot the split up of peak bases into intervals.

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

        return fig, axe
