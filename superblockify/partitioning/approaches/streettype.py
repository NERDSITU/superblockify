"""Approach only based on street type."""
from networkx import weakly_connected_components

from ..base import BasePartitioner
from ...attribute import (
    new_edge_attribute_by_function,
    get_edge_subgraph_with_attribute_value,
)
from ...config import logger


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
            # check if 'residential' or 'living_street' == highway or in highway
            lambda h: 1 if h in ["residential", "living_street"] else 0,
            source_attribute="highway",
            destination_attribute=self.attribute_label,
        )

        self.sparsified = get_edge_subgraph_with_attribute_value(
            self.graph, self.attribute_label, 0
        )
        logger.debug(
            "Found %d edges that are not residential, find LCC of them.",
            len(self.sparsified.edges),
        )

        # Find the largest connected component of the non-residential edges
        # Nodes in of the largest weakly connected component
        self.sparsified = max(weakly_connected_components(self.sparsified), key=len)
        # Construct subgraph of nodes in the largest weakly connected component
        self.sparsified = self.graph.subgraph(self.sparsified)
        # Graph was spanned with nodes, disregarding edge types
        self.sparsified = get_edge_subgraph_with_attribute_value(
            self.sparsified, self.attribute_label, 0
        )

        self.set_components_from_sparsified()
