"""Approach only based on street type."""
import logging

import matplotlib.pyplot as plt
from networkx import (
    weakly_connected_components,
    strongly_connected_components,
)
from osmnx.stats import edge_length_total

from .partitioner import BasePartitioner
from ..attribute import (
    new_edge_attribute_by_function,
)
from ..plot import save_plot

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

        # Write to 'residential' attribute 1 if edge['highway'] is or contains
        # 'residential', None otherwise
        new_edge_attribute_by_function(
            self.graph,
            lambda highway: 1 if "residential" in highway else None,
            source_attribute="highway",
            destination_attribute="residential",
        )
        self.attribute_label = "residential"

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

        # if make_plots:
        #     # Make plot of the sparsified subgraph
        #     fig, _ =

        # Find difference, edgewise, between the graph and the sparsified subgraph
        rest = self.graph.edge_subgraph(
            [
                (u, v, k)
                for u, v, k, d in self.graph.edges(keys=True, data=True)
                if (u, v, k) not in self.sparsified.edges(keys=True)
            ]
        )
        wc_components = list(weakly_connected_components(rest))

        self.attr_value_minmax = (0, len(wc_components))
        self.partitions = []
        for i, component in enumerate(wc_components):
            # Find edges that are connected to the component nodes, but not sparsified
            subgraph = self.graph.edge_subgraph(
                [
                    (u, v, k)
                    for u, v, k in rest.edges(keys=True)
                    if u in component or v in component
                ]
            )
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
            save_plot(self.results_dir, fig, f"{self.name}_partition_graph.pdf")
            plt.show()

        self.components = self.partitions
        for component in self.components:
            component["ignore"] = False
            #     (
            #     component["num_edges"] < min_edge_count
            #     or component["length_total"] < min_length
            # )

        logger.debug(
            "Found %d components, %d of which are considered.",
            len(self.components),
            len([c for c in self.components if not c["ignore"]]),
        )

        if make_plots:
            fig, _ = self.plot_subgraph_component_size("length")
            save_plot(self.results_dir, fig, f"{self.name}_subgraph_component_size.pdf")
            plt.show()

        if make_plots:
            fig, _ = self.plot_component_graph()
            save_plot(self.results_dir, fig, f"{self.name}_component_graph.pdf")
            plt.show()
