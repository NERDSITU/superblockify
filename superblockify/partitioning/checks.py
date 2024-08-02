"""Checks for the partitioning module."""

from itertools import chain

from networkx import is_weakly_connected
from numpy import argwhere, fill_diagonal
from osmnx import plot_graph

from ..config import logger
from ..plot import plot_by_attribute
from ..utils import has_pairwise_overlap


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
        7. Representative nodes are contained in their respective subgraph


    Parameters
    ----------
    partitioning : partitioning.base.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether the partitioning is valid

    """
    is_valid = True

    # 1. Check if the sparsified graph is connected
    logger.debug(
        "Checking if the sparsified graph of %s is connected.", partitioning.name
    )
    if not is_weakly_connected(partitioning.sparsified):
        logger.warning(
            "The sparsified graph of %s is not connected.", partitioning.name
        )
        is_valid = False

    # 2. Check if each subgraph is connected
    logger.debug("Checking if each subgraph of %s is connected.", partitioning.name)
    if not components_are_connected(partitioning):
        is_valid = False

    # 3. - 5. For every node and edge in the graph, check if it is contained in exactly
    # one subgraph and not the sparsified graph
    logger.debug(
        "Checking if each node and edge of %s is contained in exactly one subgraph "
        "and not the sparsified graph.",
        partitioning.name,
    )
    if not nodes_and_edges_are_contained_in_exactly_one_subgraph(partitioning):
        is_valid = False

    # 6. Check if each subgraph is connected to the sparsified graph
    logger.debug(
        "Checking if each subgraph of %s is connected to the sparsified graph.",
        partitioning.name,
    )
    if not components_are_connect_sparsified(partitioning):
        is_valid = False

    # 7. Check if representative nodes are contained in their subgraph
    logger.debug(
        "Checking if representative nodes of %s are contained in their subgraph.",
        partitioning.name,
    )
    if not representative_nodes_are_contained_in_subgraph(partitioning):
        is_valid = False

    if is_valid:
        logger.info("The partitioning %s is valid.", partitioning.name)
    else:
        logger.error(
            "The partitioning %s is not valid. See warnings for more information.",
            partitioning.name,
        )
    return is_valid


def components_are_connected(partitioning):
    """Check if each component is connected to the sparsified graph.

    Parameters
    ----------
    partitioning : partitioning.base.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether each component is connected to the sparsified graph.
    """
    found = True

    for component in partitioning.get_ltns():
        if not is_weakly_connected(component["subgraph"]):
            logger.warning(
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
                edge_attr="highlight",
                edge_attr_types="numerical",
                edge_cmap="hsv",
                edge_minmax_val=(0, 1),
            )
            # Reset edge attribute 'highlight'
            for edge in component["subgraph"].edges:
                partitioning.graph.edges[edge]["highlight"] = None

            found = False

    return found


def nodes_and_edges_are_contained_in_exactly_one_subgraph(partitioning):
    """Check if each node and edge is contained in exactly one subgraph.

    Edges can also be contained in the sparsified graph.
    Use :func:`superblockify.utils.has_pairwise_overlap` to check subgraphs overlap.

    Parameters
    ----------
    partitioning : partitioning.base.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether each node and edge is contained in exactly one subgraph.
    """
    is_valid = True

    partition_nodes_subgraphs = partitioning.get_partition_nodes()

    # Check overlap of nodes
    nodes_overlap_matrix = has_pairwise_overlap(
        [list(part["nodes"]) for part in partition_nodes_subgraphs]
    )
    fill_diagonal(nodes_overlap_matrix, False)
    if nodes_overlap_matrix.any():
        logger.warning(
            "The nodes of %s overlap in the following subgraphs: %s",
            partitioning.name,
            [
                f"{partition_nodes_subgraphs[i]['name']} and "
                f"{partition_nodes_subgraphs[j]['name']}"
                for i, j in argwhere(nodes_overlap_matrix)
                # only lower triangle, because matrix is symmetric
                if i < j
            ],
        )
        is_valid = False

    # Nodes that are in no subgraph, neither the sparsified graph
    missing_nodes = (
        set(partitioning.graph.nodes)
        - set(
            # flattened set of all nodes in all subgraphs
            chain.from_iterable(part["nodes"] for part in partition_nodes_subgraphs)
        )
        - set(partitioning.sparsified.nodes)
    )
    if len(missing_nodes) > 0:
        logger.warning(
            "%s has %d nodes that are not contained in any subgraph or the "
            "sparsified graph: %s",
            partitioning.name,
            len(missing_nodes),
            missing_nodes,
        )
        # Write to attribute 'missing_nodes' 1 if node is in missing_nodes, 0 otherwise
        for node in partitioning.graph.nodes:
            partitioning.graph.nodes[node]["missing_nodes"] = node in missing_nodes
        plot_graph(
            partitioning.graph,
            node_color=[
                "red" if partitioning.graph.nodes[node]["missing_nodes"] else "none"
                for node in partitioning.graph.nodes
            ],
            bgcolor="none",
        )
        is_valid = False

    # Check overlap of edges, including sparsified graph
    edges_overlap_matrix = has_pairwise_overlap(
        [list(part["subgraph"].edges) for part in partition_nodes_subgraphs]
        + [list(partitioning.sparsified.edges)]
    )
    fill_diagonal(edges_overlap_matrix, False)
    if edges_overlap_matrix.any():
        namelist = [part["name"] for part in partition_nodes_subgraphs] + ["sparse"]

        logger.warning(
            "The edges of %s overlap in the following subgraphs: %s",
            partitioning.name,
            [
                f"{namelist[i]} and {namelist[j]}"
                for i, j in argwhere(edges_overlap_matrix)
                # only lower triangle, because matrix is symmetric
                if i < j
            ],
        )
        is_valid = False

    # Edges that are in no subgraph, neither the sparsified graph
    missing_edges = (
        set(partitioning.graph.edges)
        - set(
            # flattened set of all edges in all subgraphs
            chain.from_iterable(
                part["subgraph"].edges for part in partition_nodes_subgraphs
            )
        )
        - set(partitioning.sparsified.edges)
    )
    if len(missing_edges) > 0:
        logger.warning(
            "%s has %d edges that are not contained in any subgraph or the "
            "sparsified graph: %s",
            partitioning.name,
            len(missing_edges),
            missing_edges,
        )
        is_valid = False

    return is_valid


def components_are_connect_sparsified(partitioning):
    """Check if each subgraph is connected to the sparsified graph.

    Parameters
    ----------
    partitioning : partitioning.base.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether each subgraph is connected to the sparsified graph
    """

    for component in partitioning.get_ltns():
        # subgraph and sparsified graph are connected if there is at least one node
        # that is contained in both
        if not any(
            node in component["subgraph"].nodes
            and node in partitioning.sparsified.nodes
            for node in partitioning.graph.nodes
        ):
            logger.warning(
                "The subgraph %s of %s is not connected to the sparsified graph.",
                component["name"],
                partitioning.name,
            )
            return False

    return True


def representative_nodes_are_contained_in_subgraph(partitioning):
    """Check if representative nodes are contained in their subgraph.

    Parameters
    ----------
    partitioning : partitioning.base.BasePartitioner
        Partitioning to check.

    Returns
    -------
    bool
        Whether representative nodes are contained in their subgraph.
    """
    partition_nodes_subgraphs = partitioning.get_partition_nodes()

    for component in partition_nodes_subgraphs:
        if not component["rep_node"] in component["subgraph"].nodes:
            logger.warning(
                "The representative node %s of %s is not contained in the "
                "subgraph %s.",
                component["rep_node"],
                partitioning.name,
                component["name"],
            )
            return False
    return True
