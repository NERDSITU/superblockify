"""Utility functions for Partitioners."""
import logging
from os import path

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
    save_path : str,
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
        else path.join(partitioner.results_dir, partitioner.name + ".gpkg")
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
    else:
        raise ValueError(
            "Partitioner has neither components nor partitions attribute, "
            "this should not happen."
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

    nodes, edges = graph_to_gdfs(partitioner.graph, nodes=True, fill_edge_geometry=True)
    # For attributes that are lists, we need to convert them to strings
    for col in edges.columns:
        if edges[col].dtype == "object":
            logger.debug("Converting column %s of type %s to str.", col, type(col))
            edges[col] = edges[col].astype(str)
    # Sparsified edges are saved with the value "SPARSE" into the "classification"
    # edge attribute
    set_edge_attributes(partitioner.sparsified, "SPARSE", "classification")

    # Remove certain attributes from nodes and edges
    # nodes = nodes.drop(columns=[])
    edges = edges.drop(columns=["component_name", "length"])

    # list attributes in nodes and edges
    logger.info("Nodes attributes: %s", nodes.columns)
    logger.info("Edges attributes: %s", edges.columns)

    # Save nodes and edges to seperate layers/geodataframes of same geodatapackage
    nodes.to_file(filepath, layer="nodes", index=False, mode="w")  # overwrite
    edges.to_file(filepath, layer="edges", index=False, mode="a")  # append
