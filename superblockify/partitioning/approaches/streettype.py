"""Approach only based on street type."""

from .attribute import AttributePartitioner
from ...attribute import new_edge_attribute_by_function
from ...config import logger


class ResidentialPartitioner(AttributePartitioner):
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

    def write_attribute(self, **kwargs):
        """Group by street type.

        Write 0 to :attr:`attribute_label` if the edge is or contains a residential
        street, 1 otherwise.
        """
        self.attribute_label = "residential"
        self.attribute_dtype = int
        logger.debug("Writing residential attribute to graph.")
        new_edge_attribute_by_function(
            self.graph,
            # check if 'residential' or 'living_street' == highway or in highway
            lambda h: 0 if h in ["residential", "living_street"] else 1,
            source_attribute="highway",
            destination_attribute=self.attribute_label,
        )
