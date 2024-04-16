"""Dummy partitioner."""

from networkx import weakly_connected_components
from numpy import mean, min as npmin, max as npmax
from typing_extensions import deprecated

from ..base import BasePartitioner


@deprecated("DummyPartitioner does not necessarily produce a valid partitioning.")
class DummyPartitioner(BasePartitioner):
    """Dummy partitioner.

    Using the inner fifth in terms of x coordinates of the graph as LCC, and the
    rest as partitions.
    """

    def partition_graph(self, make_plots=False, **kwargs):
        """Run method. Must be overridden.

        Idea: Take sparsified graph as edges connected by nodes with x coordinates
              in the middle fifth of the graph, then the LCC. Partitions are then
              the WCCs of the rest.
        """

        # The label under which the partition attribute is saved in the `self.graph`.
        self.attribute_label = "dummy_attribute"
        self.attribute_dtype = None

        id_x_coords = self.graph.nodes(data="x")

        x_range_mean = (
            npmax(id_x_coords, axis=0)[1] - npmin(id_x_coords, axis=0)[1],
            mean(id_x_coords, axis=0)[1],
        )

        lcc_nodes = max(
            weakly_connected_components(
                self.graph.subgraph(
                    [
                        node
                        for node, x in id_x_coords
                        if x_range_mean[1] - x_range_mean[0] / 5
                        < x
                        < x_range_mean[1] + x_range_mean[0] / 5
                    ]
                )
            ),
            key=len,
        )

        self.sparsified = self.graph.subgraph(lcc_nodes)

        self.set_components_from_sparsified()

        # For tests where the partitioner only uses partitions, not components
        self.components = None
