"""A approximation of the minimum Steiner tree."""

from networkx import set_edge_attributes, strongly_connected_components
from networkx.algorithms.approximation.steinertree import steiner_tree
from numpy import min as npmin, max as npmax, sum as npsum
from numpy.random import default_rng

from .attribute import AttributePartitioner
from ...config import logger


class MinimumPartitioner(AttributePartitioner):
    """Partitioner that sets the sparsified network to the steiner tree.
    Can be in terms of distance or travel time.
    """

    __exclude_test_fixture__ = True  # own tests will be run

    def write_attribute(
        self,
        weight="travel_time",
        fraction=0.4,
        seed=None,
        low_betweenness_mode=None,
        num_subtrees=1,
        remove_oneway_edges=False,
        **kwargs,
    ):
        """Set the sparsified graph to the Steiner tree.

        Sample random nodes with fixed seed and find the minimum spanning tree of the
        subgraph induced by these nodes. The partitions are then the related components
        into which the residual graph decomposes.

        Edges that are `oneway` can be excluded with `remove_oneway_edges`.
        The idea is so this approach can produce Superblocks that are reachable from
        every other Superblock, as Steiner trees are calculated on undirected graphs.
        But this highly depends on how the place was mapped. If arterial roads in two
        directions are mapped as two separate ways, this is not the way out.

        Normally, the sampling probability is uniform, but if `low_betweenness_mode` is
        set, the sampling probability is inversely proportional to the betweenness
        centrality of the nodes, in the respective betweenness type. This way, the rest
        graph should fall into more components.

        This approach violates the requirement that every Superblock is reachable from
        every other Superblock.

        Parameters
        ----------
        weight : str, optional
            Edge attribute to use as weight for the minimum spanning tree, by default
            'length', can also be 'travel_time' or None for hop count
        fraction : float, optional
            Fraction of nodes to sample, by default 0.5
        seed : int, optional
            Seed for the random number generator, by default None.
        low_betweenness_mode : str, optional
            Can be 'normal', 'length', 'linear', or None, by default None.
            Read more about the centrality types in the resources of
            :func:`superblockify.metrics.measures.betweenness_centrality`.
        num_subtrees : int, optional
            Number of subtrees to find, by default 1.
            Sampled nodes are divided into `subtrees` subsets, and a steiner tree is
            found for each subset. The sparsified graph is then the union of these
            steiner trees. This way, the sparsified graph can be a forest and

        Notes
        -----
        The runtime of this approach is not optimized and can be improved by using the
        shortest paths calculation of :mod:`superblockify.metrics.distances`.
        """
        self.attribute_label = "steiner_tree"
        self.attribute_dtype = int
        rng = default_rng(seed)
        if remove_oneway_edges:
            # Get graph without one-way edges - if edge (u, v) and (v, u) exist
            graph = self.graph.edge_subgraph(
                (u, v, k)
                for u, v, k in self.graph.edges(keys=True)
                if (v, u) not in self.graph.edges(keys=False)
            )
            graph = max(strongly_connected_components(graph), key=len)
            graph = self.graph.subgraph(graph)
            if graph.number_of_nodes() < 0.5 * self.graph.number_of_nodes():
                logger.warning(
                    "Graph without oneway edges has less than half of the nodes of the "
                    "original graph. This can lead to bad performance."
                )
            logger.debug(
                "Graph without oneway edges has %s nodes.",
                graph.number_of_nodes(),
            )
        else:
            graph = self.graph.subgraph(self.graph.nodes)

        if weight is None:
            set_edge_attributes(graph, values=1, name="hop")

        # 1. Sample nodes
        if low_betweenness_mode:
            logger.info("Sampling nodes with low betweenness centrality.")
            nodes = self.sample_nodes_low_betweenness(
                graph,
                fraction=fraction,
                rng=rng,
                betweenness_type=low_betweenness_mode,
                make_plots=kwargs.get("make_plots", False),
            )
        else:
            logger.info("Sampling nodes uniformly.")
            nodes = self.sample_nodes_uniform(graph, fraction=fraction, rng=rng)
        # 2. Find the steiner tree
        st_trees = []
        for subtree in range(num_subtrees):
            logger.info(
                "Finding the steiner tree. %s/%s",
                subtree + 1,
                num_subtrees,
            )
            # divide nodes in len(num_subtrees) subsets
            st_trees.append(
                steiner_tree(
                    graph.to_undirected(as_view=True),
                    nodes[subtree::num_subtrees],  # every subtree-th node
                    weight=weight or "hop",
                    method="kou",
                )
            )
            logger.debug("Steiner tree has %s nodes.", st_trees[-1].number_of_nodes())
        st_trees = self.graph.edge_subgraph(
            (u, v, k) for st_tree in st_trees for u, v, k in st_tree.edges(keys=True)
        )

        logger.debug("Union of steiner trees has %s nodes.", st_trees.number_of_nodes())
        # 3. Set the attribute
        logger.debug("Setting the attribute.")
        set_edge_attributes(
            self.graph,
            {
                (u, v, k): (
                    1
                    if (u, v) in st_trees.edges(keys=False)
                    or (v, u) in st_trees.edges(keys=False)
                    else 0
                )
                for u, v, k in self.graph.edges(keys=True)
            },
            name=self.attribute_label,
        )

    def sample_nodes_uniform(self, graph, fraction, rng):
        """Sample nodes uniformly."""
        return list(
            rng.choice(
                graph.nodes,
                size=min(
                    int(fraction * self.graph.number_of_nodes()),
                    graph.number_of_nodes(),
                ),
                replace=False,
            )
        )

    def sample_nodes_low_betweenness(
        self, graph, fraction, rng, betweenness_type="normal", make_plots=False
    ):
        """Sample nodes with low betweenness centrality.

        Parameters
        ----------
        graph : networkx.Graph
            Graph to sample nodes from.
        fraction : float
            Fraction of nodes to sample.
        rng : numpy.random.Generator
            Random number generator.
        betweenness_type : str, optional
            Can be 'normal', 'length', or 'linear', by default 'normal'.
        make_plots : bool, optional
            Whether to make distance matrix plots, by default False.

        Returns
        -------
        list of int
            List of sampled nodes.
        """
        # 1. Calculate the betweenness centrality
        self.calculate_metrics_before(make_plots=make_plots)
        # 2. Sample nodes with low betweenness centrality
        # each node has the attribute `node_betweenness_{betweenness_type}`
        # sample by probability proportional to the inverse of the betweenness
        # centrality
        min_max = [
            npmin(
                [
                    graph.nodes[node][f"node_betweenness_{betweenness_type}"]
                    for node in graph.nodes
                ]
            ),
            npmax(
                [
                    graph.nodes[node][f"node_betweenness_{betweenness_type}"]
                    for node in graph.nodes
                ]
            ),
        ]
        nodes, probability = zip(
            *[
                (
                    node,  # rescale [min, max] to [1.1, 0.1]
                    1.1
                    - (
                        graph.nodes[node][f"node_betweenness_" f"{betweenness_type}"]
                        - min_max[0]
                    )
                    / (min_max[1] - min_max[0]),
                )
                for node in graph.nodes
            ]
        )
        probability = probability / npsum(probability)
        return list(
            rng.choice(
                nodes,
                size=min(
                    int(fraction * self.graph.number_of_nodes()),
                    graph.number_of_nodes(),
                ),
                p=probability,
                replace=False,
            )
        )
