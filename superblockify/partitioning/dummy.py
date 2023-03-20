"""Dummy partitioner."""
from random import choice

from .partitioner import BasePartitioner
from ..attribute import (
    get_edge_subgraph_with_attribute_value,
    new_edge_attribute_by_function,
)


class DummyPartitioner(BasePartitioner):
    """Dummy partitioner.

    Partitions randomly.
    """

    def partition_graph(self, make_plots=False, **kwargs):
        """Run method. Must be overridden.

        Assign random partitions to edges.
        """

        # The label under which the partition attribute is saved in the `self.graph`.
        self.attribute_label = "dummy_attribute"

        # Somehow determining the partition of edges
        # - edges also may not be included in any partition and miss the label
        values = list(range(3))
        self.attr_value_minmax = (min(values), max(values))
        new_edge_attribute_by_function(
            self.graph, lambda bear: choice(values), "osmid", self.attribute_label
        )

        # A List of the existing partitions, the 'value' attribute should be equal to
        # the edge attributes under the instances `attribute_label`, which belong to
        # this partition
        self.partitions = [
            {
                "name": str(num),
                "value": num,
                "subgraph": get_edge_subgraph_with_attribute_value(
                    self.graph, self.attribute_label, num
                ),
                "num_edges": num,
                "num_nodes": num,
                "length_total": num,
            }
            for num in values
        ]
