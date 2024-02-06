"""Approach based on high betweenness centrality of nodes and edges."""

from numpy import array

from .attribute import AttributePartitioner
from ...attribute import new_edge_attribute_by_function
from ...config import logger


class BetweennessPartitioner(AttributePartitioner):
    """Partitioner using betweenness centrality of nodes and edges.

    Set sparsified graph from edges or nodes with high betweenness centrality.
    """

    def write_attribute(
        self, percentile=85.0, scaling="normal", max_range=None, **kwargs
    ):
        """Determine edges with high betweenness centrality for sparsified graph.

        Edges with high betweenness centrality are used to construct the sparsified
        graph.

        The high percentile is determined through ranking all edges by their
        betweenness centrality and taking the top percentile. The percentile is
        determined by the `percentile` parameter.

        Parameters
        ----------
        percentile : float, optional
            The percentile to use for determining the high betweenness centrality
            edges, by default 90.0
        scaling : str, optional
            The type of betweenness to use, can be `normal`, `length`, or `linear`,
            by default `normal`
        max_range : int, optional
            The range to use for calculating the betweenness centrality, by default
            None, which uses the whole graph. Its unit is meters.
        **kwargs
            Additional keyword arguments. `calculate_metrics_before` takes the
            `make_plots` parameter.


        Raises
        ------
        ValueError
            If `scaling` is not `normal`, `length`, or `linear`.
        ValueError
            If `percentile` is not between, 0.0 and 100.0.
        """
        self.attribute_label = "betweenness_percentile"
        self.attribute_dtype = int

        logger.debug("Writing edge betweenness attribute to graph.")
        if not isinstance(percentile, (float, int)) or not 0.0 < percentile < 100.0:
            raise ValueError(
                f"Percentile must be between, 0.0 and 100.0, but is {percentile}."
            )
        if scaling not in ["normal", "length", "linear"]:
            raise ValueError(
                f"Scaling must be 'normal', 'length', or 'linear', but is {scaling}."
            )

        self.calculate_metrics_before(
            make_plots=kwargs.get("make_plots", False), betweenness_range=max_range
        )

        # determine a threshold for betweenness from ranking
        attr_list = array(
            [
                val
                for _, _, val in self.graph.edges(
                    data=f"edge_betweenness_{scaling}"
                    + ("_range_limited" if max_range else "")
                )
            ]
        )

        attr_list.sort()
        threshold = attr_list[int(len(attr_list) * percentile / 100.0)]

        # Threshold is not allowed to be below the minimal and above the maximal value
        if threshold <= attr_list[0]:
            ixd_2nd_smallest = 1
            while attr_list[ixd_2nd_smallest] == attr_list[0]:
                ixd_2nd_smallest += 1
            # set threshold to second-smallest value
            threshold = attr_list[ixd_2nd_smallest]
            # at least one node/edge is outside
        elif threshold >= attr_list[-1]:  # improbable case due to betw. distribution
            ixd_2nd_largest = -2
            while attr_list[ixd_2nd_largest] == attr_list[-1]:
                ixd_2nd_largest -= 1
            # set threshold to second-largest value
            threshold = attr_list[ixd_2nd_largest]
            # at least one node/edge is inside

        # write boolean attribute to graph
        new_edge_attribute_by_function(
            self.graph,
            lambda x: 0 if x < threshold else 1,
            source_attribute=f"edge_betweenness_{scaling}",
            destination_attribute=self.attribute_label,
        )
