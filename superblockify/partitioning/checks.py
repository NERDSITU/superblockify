"""Checks for the partitioning module."""

import logging

from networkx import is_weakly_connected
from osmnx import plot_graph

from ..plot import plot_by_attribute

logger = logging.getLogger("superblockify")


def is_valid_partitioning(partitioning):
    """Check if a partitioning is valid.

    The components of a partitioning are the subgraphs, a special subgraph is the
    sparsified graph.

    A partitioning is valid if the following conditions are met:
        1. The sparsified graph is connected
        2. Each subgraph is connected
        3. Each node is contained in exactly one subgraph and not the sparsified graph
        4. Each edge is contained in exactly one subgraph and not the sparsified graph
        5. No node or edge of `graph` is not contained in any subgraph or the
           sparsified graph
        6. Each subgraph is connected to the sparsified graph


    Parameters
    ----------
    partitioning : partitioning.partitioner.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether the partitioning is valid

    """

    # 1. Check if the sparsified graph is connected
    logger.debug(
        "Checking if the sparsified graph of %s is connected.", partitioning.name
    )
    if not is_weakly_connected(partitioning.sparsified):
        logger.error("The sparsified graph of %s is not connected.", partitioning.name)
        return False

    # 2. Check if each subgraph is connected
    logger.debug("Checking if each subgraph of %s is connected.", partitioning.name)
    if not components_are_connected(partitioning):
        return False

    # 3. - 5. For every node and edge in the graph, check if it is contained in exactly
    # one subgraph and not the sparsified graph
    logger.debug(
        "Checking if each node and edge of %s is contained in exactly one subgraph "
        "and not the sparsified graph.",
        partitioning.name,
    )
    if not nodes_and_edges_are_contained_in_exactly_one_subgraph(partitioning):
        return False

    # 6. Check if each subgraph is connected to the sparsified graph
    logger.debug(
        "Checking if each subgraph of %s is connected to the sparsified graph.",
        partitioning.name,
    )
    if not components_are_connect_sparsified(partitioning):
        return False

    logger.info("The partitioning %s is valid.", partitioning.name)

    return True


def components_are_connected(partitioning):
    """Check if each component is connected to the sparsified graph.

    Parameters
    ----------
    partitioning : partitioning.partitioner.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether each component is connected to the sparsified graph.
    """
    found = True

    for component in partitioning.components:
        if not is_weakly_connected(component["subgraph"]):
            logger.error(
                "The subgraph %s of %s is not connected.",
                component["name"],
                partitioning.name,
            )
            # Plot graph with highlighted subgraph
            # Write to attribute 'highlight' 1 if node is in subgraph, 0 otherwise
            for edge in component["subgraph"].edges:
                partitioning.graph.edges[edge]["highlight"] = 1
            plot_by_attribute(
                partitioning.graph,
                "highlight",
                attr_types="numerical",
                cmap="hsv",
                minmax_val=(0, 1),
            )
            # Reset edge attribute 'highlight'
            for edge in component["subgraph"].edges:
                partitioning.graph.edges[edge]["highlight"] = None

            found = False

    return found


def nodes_and_edges_are_contained_in_exactly_one_subgraph(partitioning):
    """Check if each node and edge is contained in exactly one subgraph.

    Edges can also be contained in the sparsified graph.

    Parameters
    ----------
    partitioning : partitioning.partitioner.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether each node and edge is contained in exactly one subgraph.
    """

    duplicate_nodes = set()

    for node in partitioning.graph.nodes:
        num_contained = 0
        for part in partitioning.get_partition_nodes():
            if node in part["nodes"]:
                num_contained += 1
        if node in partitioning.sparsified.nodes:
            num_contained += 1
        if num_contained != 1:
            duplicate_nodes.add(node)
            logger.error(
                "The node %s of %s is contained in %d subgraphs. It should be "
                "contained in exactly one subgraph or the sparsified graph.",
                node,
                partitioning.name,
                num_contained,
            )

    # Plot graph with marked duplicate nodes
    if len(duplicate_nodes) > 0:
        # Write to attribute 'duplicate' 1 if node is in duplicate_nodes, 0 otherwise
        for node in partitioning.graph.nodes:
            partitioning.graph.nodes[node]["duplicate_node"] = node in duplicate_nodes
        plot_graph(
            partitioning.graph,
            node_color=[
                "red" if partitioning.graph.nodes[node]["duplicate_node"] else "none"
                for node in partitioning.graph.nodes
            ],
            bgcolor="none",
        )
        return False

    for edge in partitioning.graph.edges:
        num_contained = 0
        for component in partitioning.components:
            if edge in component["subgraph"].edges:
                num_contained += 1
        if edge in partitioning.sparsified.edges:
            num_contained += 1
        if num_contained != 1:
            logger.error(
                "The edge %s of %s is contained in %d subgraphs. It should be "
                "contained in exactly one subgraph or the sparsified graph.",
                edge,
                partitioning.name,
                num_contained,
            )
            return False

    return True


def components_are_connect_sparsified(partitioning):
    """Check if each subgraph is connected to the sparsified graph.

    Parameters
    ----------
    partitioning : partitioning.partitioner.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether each subgraph is connected to the sparsified graph
    """

    for component in partitioning.components:
        # subgraph and sparsified graph are connected if there is at least one node
        # that is contained in both
        if not any(
            node in component["subgraph"].nodes
            and node in partitioning.sparsified.nodes
            for node in partitioning.graph.nodes
        ):
            logger.error(
                "The subgraph %s of %s is not connected to the sparsified graph.",
                component["name"],
                partitioning.name,
            )
            return False

    return True
