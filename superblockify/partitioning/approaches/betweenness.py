"""Approach based on high betweenness centrality of nodes and edges."""
from numpy import array

from .attribute import AttributePartitioner
from ...attribute import new_edge_attribute_by_function
from ...config import logger


class BetweennessPartitioner(AttributePartitioner, attribute="betweenness_percentile"):
    """Partitioner using betweenness centrality of nodes and edges.

    Set sparsified graph from edges or nodes with high betweenness centrality.
    """

    def write_attribute(self, mode="edge", percentile=85.0, scaling="normal", **kwargs):
        """Determine edges with high betweenness centrality for sparsified graph.

        Approach works in different modes:
        - `edge`: edges with high betweenness centrality are used to construct the
          sparsified graph.
        - `node`: edges for which both connecting nodes are in the high betweenness
          centrality percentile are used to construct the sparsified graph.

        The high percentile is determined through ranking all nodes/edges by their
        betweenness centrality and taking the top percentile. The percentile is
        determined by the `percentile` parameter.

        Parameters
        ----------
        mode : str, optional
            Whether to use edges or nodes for the sparsified graph, by default `edge`
        percentile : float, optional
            The percentile to use for determining the high betweenness centrality
            nodes/edges, by default 90.0
        scaling : str, optional
            The type of betweenness to use, can be `normal`, `length`, or `linear`,
            by default `normal`

        Raises
        ------
        ValueError
            If `mode` is not `edge` or `node`.
        ValueError
            If `scaling` is not `normal`, `length`, or `linear`.
        ValueError
            If `percentile` is not between, including, 0.0 and 100.0.
        """
        logger.debug("Writing %s betweenness attribute to graph.", mode)
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError(
                "Percentile must be between, including, 0.0 and 100.0, "
                f"but is {percentile}."
            )
        if scaling not in ["normal", "length", "linear"]:
            raise ValueError(
                f"Scaling must be 'normal', 'length', or 'linear', but is {scaling}."
            )

        self.calculate_metrics_before(make_plots=kwargs.get("make_plots", False))

        # determine threshold for betweenness from ranking
        if mode == "edge":
            attr_list = array(
                [
                    val
                    for _, _, val in self.graph.edges(
                        data=f"{mode}_betweenness_{scaling}"
                    )
                ]
            )
        elif mode == "node":
            attr_list = array(
                [
                    val
                    for _, val in self.graph.nodes(data=f"{mode}_betweenness_{scaling}")
                ]
            )
        else:
            raise ValueError(f"Mode must be 'edge' or 'node', but is {mode}.")

        attr_list.sort()
        threshold = attr_list[int(len(attr_list) * percentile / 100.0)]
        print(attr_list)

        # write boolean attribute to graph
        new_edge_attribute_by_function(
            self.graph,
            lambda x: 0 if x < threshold else 1,
            source_attribute=f"{mode}_betweenness_{scaling}",
            destination_attribute=BetweennessPartitioner.attribute_label,
        )
