"""Utility functions for Partitioners."""
import logging
from os import remove
from os.path import exists, join

from networkx import set_edge_attributes
from osmnx import graph_to_gdfs

logger = logging.getLogger("superblockify")


def save_to_gpkg(partitioner, save_path=None):
    """Save the partitioner's graph and LTNs to a geodatapackage.

    The name of the components (/partitions) are saved into a "classification" edge
    attribute. The sparse graph is saved with the value "SPARSE" into the
    "classification" edge attribute.

    Parameters
    ----------
    partitioner : superblockify.partitioning.BasePartitioner
        The partitioner to save.
    save_path : str, optional
        The path to save the geodatapackage to. If None, it will be saved to the
        partitioners folder at (part.results_dir, part.name + ".gpkg")

    Raises
    ------
    ValueError
        If the partitioner has no components or partitions attribute.
    ValueError
        If the partitioner has no sparsified subgraph.

    Notes
    -----
    The geopackage will contain the following layers:
    - nodes
        - representative_node_name
        - missing_nodes: if node is not in any component/partition or sparsified graph
        - y, x, lat, lon, geometry
    - edges
        - classification
        - osmid, highway, length, geometry
        - (bearing: the bearing of the edge, bearing_90: mod(bearing, 90))
        - (residential: 1 if edge['highway'] is or contains 'residential',
          None otherwise)
    """

    if partitioner.sparsified is None:
        raise ValueError("Partitioner has no sparsified subgraph.")
    if not isinstance(partitioner.sparsified, type(partitioner.graph)):
        raise ValueError(
            f"Partitioner's sparsified subgraph is of type "
            f"{type(partitioner.sparsified)}, but should be of type "
            f"{type(partitioner.graph)}. This is not supported."
        )

    # if partitioner.components and partitioner.partitions are None
    if not partitioner.components and not partitioner.partitions:
        raise ValueError("Partitioner has no components nor partitions attribute.")

    filepath = (
        save_path
        if save_path is not None
        else join(partitioner.results_dir, partitioner.name + ".gpkg")
    )
    if partitioner.components:
        parts = partitioner.components
        logger.info(
            "Using components attribute to save LTNs to geodatapackage %s", filepath
        )
    elif partitioner.partitions:
        parts = partitioner.partitions
        logger.info(
            "Using partitions attribute to save LTNs to geodatapackage %s", filepath
        )

    if not isinstance(parts, list):
        raise ValueError(
            f"Partitioner's components/partitions attribute is of type {type(parts)}, "
            "but should be a list of dicts where each dict has a 'subgraph' and "
            "'name' key."
        )
    # if parts are not None and type is not list of dicts with "subgraph" and "name"
    if not all(
        isinstance(comp, dict) and "subgraph" in comp and "name" in comp
        for comp in parts
    ):
        raise ValueError(
            f"Partitioner's components/partitions attribute is of type {type(parts)}, "
            "but should be a list of dicts where each dict has a 'subgraph' and "
            "'name' key."
        )

    # Bake the LTNs into the graph
    for _, part in enumerate(parts):
        # As part["subgraph"] is connected to partitioner.graph, we can just
        # change the edge attribute in the whole subgraph, applying the LTN
        set_edge_attributes(part["subgraph"], part["name"], "classification")
    # Sparsified edges are saved with the value "SPARSE" into the "classification"
    # edge attribute
    set_edge_attributes(partitioner.sparsified, "SPARSE", "classification")

    nodes, edges = graph_to_gdfs(partitioner.graph, nodes=True, fill_edge_geometry=True)
    # For attributes that are lists, we need to convert them to strings
    for col in edges.columns:
        if edges[col].dtype == "object":
            logger.debug("Converting column %s of type %s to str.", col, type(col))
            edges[col] = edges[col].astype(str)

    # list attributes in nodes and edges
    logger.info("Node attributes: %s", nodes.columns)
    logger.info("Edge attributes: %s", edges.columns)

    # Remove certain attributes from nodes and edges
    # nodes = nodes.drop(columns=[])
    edges = edges.drop(
        columns=[attr for attr in edges.columns if attr in ["component_name", "length"]]
    )

    # Save nodes and edges to seperate layers/geodataframes of same geodatapackage
    # if file already exists, remove it
    if exists(filepath):
        remove(filepath)
    nodes.to_file(filepath, layer="nodes", index=False, mode="w")
    logger.info("Saved %d nodes to %s", len(nodes), filepath)
    edges.to_file(filepath, layer="edges", index=False, mode="w")
    logger.info("Saved %d edges to %s", len(edges), filepath)


def show_highway_stats(graph):
    """Show the number of edges for each highway type in the graph.

    Also show the type proportions of the highway attributes.

    Parameters
    ----------
    graph : networkx.classes.multidigraph.MultiDiGraph
        Graph, where the edges have a "highway" attribute.

    Notes
    -----
    The highway is usually a string, but can also be a list of strings.
    If the proportion of edges with a highway attribute of type 'str' is
    below 98%, a warning is logged.
    """
    edges = graph_to_gdfs(graph, nodes=False, fill_edge_geometry=False)
    # highway counts
    highway_counts = edges.highway.value_counts().to_frame("count")
    highway_counts["proportion"] = highway_counts["count"] / len(edges)
    logger.info(
        "Highway counts (type, count, proportion): \n%s", highway_counts.to_string()
    )
    # dtypes of the highway attributes
    dtype_counts = edges.highway.apply(type).value_counts().to_frame("count")
    dtype_counts["proportion"] = dtype_counts["count"] / len(edges)
    logger.debug(
        "Dtype counts (type, count, proportion): \n%s", dtype_counts.to_string()
    )
    # Warning if 'str' is underrepresented
    if dtype_counts.loc[dtype_counts.index == str, "proportion"].values < 0.98:
        logger.warning(
            "The dtype of the 'highway' attribute is not 'str' in %d%% of the edges.",
            (1 - dtype_counts.loc[dtype_counts.index == str, "proportion"]) * 100,
        )
