"""Approach only based on street type."""
import logging

import matplotlib.pyplot as plt
from networkx import weakly_connected_components
from osmnx.stats import edge_length_total

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

    def __init__(self, *args, **kwargs):
        """Construct a new ResidentialPartitioner."""
        super().__init__(*args, **kwargs)

    def run(
        self, make_plots=False, min_edge_count=5, min_length=400, **kwargs
    ):
        """Group by street type and remove small components.

        Construct subgraphs for nodes that only contain residential edges around them.

        Parameters
        ----------
        make_plots : bool, optional
            Whether to show and save plots of the partitioning analysis, by default
            False
        min_edge_count : int, optional
            Minimum number of edges in a component to be considered, by default 5
        min_length : int, optional
            Minimum length of a component to be considered, by default 400
        """

        # Write to 'residential' attribute 1 if edge['highway'] is or contains
        # 'residential', None otherwise
        new_edge_attribute_by_function(
            self.graph,
            lambda highway: 1 if "residential" in highway else None,
            source_attribute="highway",
            destination_attribute="residential",
        )
        self.attribute_label = "residential"

        # Find all nodes that only connect to residential edges
        residential_nodes = set()
        for node in self.graph.nodes():
            if all(
                data[self.attribute_label] == 1
                for _, _, data in self.graph.edges(node, data=True)
            ):
                residential_nodes.add(node)

        logger.debug(
            "Found %d nodes that connect to a non-residential street. "
            "Constructing subgraphs around them.",
            len(residential_nodes),
        )

        # Weakly connected components
        subgraph_residential = self.graph.subgraph(residential_nodes)

        wc_components = list(weakly_connected_components(subgraph_residential))
        self.attr_value_minmax = (0, len(wc_components))
        self.partitions = []
        for i, component in enumerate(wc_components):
            # Find all edges that are connected to the edges in the component and
            # residential
            edges_in_component = []
            for node in component:
                for edge in self.graph.edges(node, keys=True, data=True):
                    if edge[3][self.attribute_label] == 1:
                        edges_in_component.append((edge[0], edge[1], edge[2]))

            subgraph = self.graph.edge_subgraph(edges_in_component)

            self.partitions.append(
                {
                    "name": f"residential_{i}",
                    "value": i,
                    "subgraph": subgraph,
                    "num_edges": subgraph.number_of_edges(),
                    "num_nodes": subgraph.number_of_nodes(),
                    "length_total": edge_length_total(subgraph),
                }
            )

        if make_plots:
            fig, _ = self.plot_partition_graph()
            self.save_plot(fig, f"{self.name}_partition_graph.pdf")
            plt.show()

        self.components = self.partitions
        for component in self.components:
            component["ignore"] = (
                component["num_edges"] < min_edge_count
                or component["length_total"] < min_length
            )

        logger.debug(
            "Found %d components, %d of which are considered.",
            len(self.components),
            len([c for c in self.components if not c["ignore"]]),
        )

        if make_plots:
            fig, _ = self.plot_subgraph_component_size("length")
            self.save_plot(fig, f"{self.name}_subgraph_component_size.pdf")
            plt.show()

        if make_plots:
            fig, _ = self.plot_component_graph()
            self.save_plot(fig, f"{self.name}_component_graph.pdf")
            plt.show()
