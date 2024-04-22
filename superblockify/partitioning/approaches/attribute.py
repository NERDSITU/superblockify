"""Approaches based on node and edge attributes of the graph.

For example using the betweenness centrality of nodes and edges to partition the
graph.
"""

from abc import abstractmethod, ABC

from networkx import weakly_connected_components

from ..base import BasePartitioner
from ...attribute import get_edge_subgraph_with_attribute_value
from ...config import logger


class AttributePartitioner(BasePartitioner, ABC):
    """Parent class for all partitioners that use node and/or edge attributes.

    A child class needs to write boolean edge attribute (`True`/`1` or `False`/`0`)
    to the :attr:`attribute_label` of the graph :attr:`self.graph`.
    All edges with the `True` belong to the sparsified graph, as well as all touching
    nodes. The rest of the graph falls apart into Superblocks.
    """

    @abstractmethod
    def write_attribute(self, **kwargs):
        """Write boolean edge attribute :attr:`attribute_label` to the graph.
        Abstract method, needs to be implemented by child class.
        There need to be both edges with the attribute value `True` and `False`.
        The case of all edges being `True` or `False` is equivalent to having no
        restrictions.
        """

    def partition_graph(self, make_plots=False, **kwargs):
        """Group by boolean attribute and remove small components.

        Construct sparsified graph from boolean attribute and Superblock subgraphs
        for the components that fall apart.

        Parameters
        ----------
        make_plots : bool, optional
            Whether to show and save plots of the partitioning analysis, by default
            False
        """
        self.write_attribute(make_plots=make_plots, **kwargs)

        self.sparsified = get_edge_subgraph_with_attribute_value(
            self.graph, self.attribute_label, 1
        )
        logger.debug(
            "Found %d edges with attribute %s == 1, find LCC of them.",
            len(self.sparsified.edges),
            self.attribute_label,
        )
        # Find the largest connected component in the sparsified graph
        # Nodes in of the largest weakly connected component
        self.sparsified = max(weakly_connected_components(self.sparsified), key=len)
        # Construct subgraph of nodes in the largest weakly connected component
        self.sparsified = self.graph.subgraph(self.sparsified)
        # Graph was spanned with nodes, disregarding edge types
        self.sparsified = get_edge_subgraph_with_attribute_value(
            self.sparsified, self.attribute_label, 1
        )
        # Set the components of the graph - Superblocks that fall apart
        self.set_components_from_sparsified()
