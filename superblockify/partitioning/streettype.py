"""Approach only based on street type."""
import logging

from networkx import (
    strongly_connected_components,
)

from .partitioner import BasePartitioner
from ..attribute import (
    new_edge_attribute_by_function,
)

logger = logging.getLogger("superblockify")


class ResidentialPartitioner(BasePartitioner):
    """Partitioner that only uses street type to partition the graph.

    This partitioner groups edges by their street type. Nodes that only connect to
    residential edges are then grouped into subgraphs. The resulting subgraphs are
    then partitioned into components based on their size and length.

    Notes
    -----
    The effectiveness of this partitioner is highly dependent on the quality of the
    OSM data. If the data is not complete, this partitioner will not be able to
    partition the graph into meaningful subgraphs.
    """

    def partition_graph(self, make_plots=False, **kwargs):
        """Group by street type and remove small components.

        Construct subgraphs for nodes that only contain residential edges around them.

        Parameters
        ----------
        make_plots : bool, optional
            Whether to show and save plots of the partitioning analysis, by default
            False
        """

        self.attribute_label = "residential"

        # Write to 'residential' attribute 1 if edge['highway'] is or contains
        # 'residential', None otherwise
        new_edge_attribute_by_function(
            self.graph,
            lambda highway: 1 if "residential" in highway else None,
            source_attribute="highway",
            destination_attribute=self.attribute_label,
        )

        # Find all edges that are not residential and make a subgraph of them
        non_residential_edges = [
            (u, v, k)
            for u, v, k, d in self.graph.edges(keys=True, data=True)
            if d[self.attribute_label] is None
        ]

        logger.debug(
            "Found %d edges that are not residential, find LCC of them.",
            len(non_residential_edges),
        )

        # Find the largest connected component of the non-residential edges
        self.sparsified = self.graph.edge_subgraph(non_residential_edges)
        # Nodes in of the largest strongly connected component
        self.sparsified = max(strongly_connected_components(self.sparsified), key=len)
        # Construct subgraph of nodes in the largest weakly connected component
        self.sparsified = self.graph.subgraph(self.sparsified)

        self.set_components_from_sparsified()
