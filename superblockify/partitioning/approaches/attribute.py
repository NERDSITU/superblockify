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
    nodes. The rest of the graph falls apart into LTNs.
    """

    @classmethod
    def __init_subclass__(cls, attribute=None, **kwargs):
        """Initialize the partitioner.

        Parameters
        ----------
        attribute : str
            The attribute name to use for partitioning.
        **kwargs
            Keyword arguments passed to the parent class
            :class:`superblockify.partitioning.base.BasePartitioner`.

        Raises
        ------
        ValueError
            If `attribute` is not a string.

        """
        if not isinstance(attribute, str):
            raise ValueError(
                f"Attribute must be a string, but is of type {type(attribute)}."
            )
        super().__init_subclass__(**kwargs)
        cls.attribute_label = attribute

    @abstractmethod
    def write_attribute(self, **kwargs):
        """Write boolean edge attribute :attr:`attribute_label` to the graph.
        Abstract method, needs to be implemented by child class."""

    def partition_graph(self, make_plots=False, **kwargs):
        """Group by boolean attribute and remove small components.

        Construct sparsified graph from boolean attribute and LTN subgraphs for the
        components that fall apart.

        Parameters
        ----------
        make_plots : bool, optional
            Whether to show and save plots of the partitioning analysis, by default
            False
        """
        self.write_attribute(make_plots=make_plots, **kwargs)

        self.sparsified = get_edge_subgraph_with_attribute_value(
            self.graph, self.__class__.attribute_label, 1
        )
        logger.debug(
            "Found %d edges with attribute %s == 1, find LCC of them.",
            len(self.sparsified.edges),
            self.__class__.attribute_label,
        )
        # Find the largest connected component in the sparsified graph
        # Nodes in of the largest weakly connected component
        self.sparsified = max(weakly_connected_components(self.sparsified), key=len)
        # Construct subgraph of nodes in the largest weakly connected component
        self.sparsified = self.graph.subgraph(self.sparsified)
        # Graph was spanned with nodes, disregarding edge types
        self.sparsified = get_edge_subgraph_with_attribute_value(
            self.sparsified, self.__class__.attribute_label, 1
        )
        # Set the components of the graph - LTNs that fall apart
        self.set_components_from_sparsified()
