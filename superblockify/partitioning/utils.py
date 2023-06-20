"""Utility functions for Partitioners."""
from os import remove
from os.path import exists, join
from uuid import uuid4

from networkx import set_edge_attributes, strongly_connected_components
from osmnx import graph_to_gdfs, get_undirected
from shapely import Point
from shapely.ops import substring

from ..config import logger


def save_to_gpkg(partitioner, save_path=None):
    """Save the partitioner's graph and LTNs to a geodatapackage.

    The name of the components (/partitions) is saved into a "classification" edge
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
        If the partitioner has no `components` or `partitions` attribute.
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

    # Save nodes and edges to separate layers/geodataframes of the same geodatapackage
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


def remove_dead_ends_directed(graph):
    """Remove all dead ends from the directed graph.

    Comes down to removing all nodes that are not in the largest strongly connected
    component.

    Parameters
    ----------
    graph : networkx.classes.multidigraph.MultiDiGraph
        Graph to remove dead ends from.

    Raises
    ------
    ValueError
        If the graph is not directed.

    Notes
    -----
    The graph is modified in place.
    """
    if not graph.is_directed():
        raise ValueError("Graph must be directed.")
    num_nodes = len(graph.nodes)
    # Get the largest strongly connected component
    scc = max(strongly_connected_components(graph), key=len)
    # Remove all nodes that are not in the largest strongly connected component
    for node in list(graph.nodes):
        if node not in scc:
            graph.remove_node(node)
    if len(graph.nodes) < num_nodes:
        logger.debug("Removed %d dead ends.", num_nodes - len(graph.nodes))


def split_up_isolated_edges_directed(graph, sparsified):
    """Split up all edges in the directed graph than are isolated disregarding
    the sparsified subgraph.

    Isolated edges are edges that are not connected to any other edge in the graph.
    Also, two parallel edges are considered isolated together
    if their geometries equal.

    These nodes are inserted at a middle point of the edge(s) geometry,
    the geometry split up.
    Attributes `population` and `area` split, but the further attributes are kept
    the same.

    Parameters
    ----------
    graph : networkx.classes.multidigraph.MultiDiGraph
        Graph to split up edges in.
    sparsified : networkx.classes.multidigraph.MultiDiGraph
        View of graph, edges taken away from graph to expose isolated edges.

    Raises
    ------
    ValueError
        If the graphs are not directed.
    NotImplementedError
        Parallel edges of a degree higher than 2 are not supported.


    Notes
    -----
    The graph is modified in place.
    The function generates the new node ids `get_new_node_id`.
    """  # pylint: disable=too-many-locals

    if not graph.is_directed():
        raise ValueError("Graph must be directed.")
    if not sparsified.is_directed():
        raise ValueError("Sparsified graph must be directed.")

    # Find difference, edgewise, between the graph and the sparsified subgraph
    rest = graph.edge_subgraph(
        [
            (u, v, k)
            for u, v, k in graph.edges(keys=True, data=False)
            if (u, v, k) not in sparsified.edges(keys=True)
        ]
    )
    rest_un = get_undirected(rest)

    # for u, v, k, d in rest.edges(keys=True, data=True):
    for u_isol, v_isol in [
        (u, v)
        for u, v in rest_un.edges()
        if rest_un.degree(u) == 1 and rest_un.degree(v) == 1
    ]:
        # For one-way-street, only one edge
        if rest.degree(u_isol) == 1 and rest.degree(v_isol) == 1:
            edges = [
                (
                    *(  # single edge with an unknown direction and key
                        list(rest.edges(v_isol, keys=True, data=True))
                        + list(rest.edges(u_isol, keys=True, data=True))
                    )[0],
                    u_isol,
                )
            ]
        # For two-way-street, two edges
        elif rest.degree(u_isol) == 2 and rest.degree(v_isol) == 2:
            edges = [(*list(rest.edges(v_isol, keys=True, data=True))[0], u_isol)]
            edges += [(*list(rest.edges(u_isol, keys=True, data=True))[0], v_isol)]
        else:
            raise NotImplementedError(
                f"Parallel edges of degree {rest.degree(u_isol)} and "
                f"{rest.degree(v_isol)} are not supported yet, "
                f"but the elif case should be general enough."
            )  # If you want to add this, please add a test case

        # Split up at the middle point
        if "geometry" not in edges[0][3]:
            geom = None
            # interpolate between the nodes
            middle = Point(
                (rest.nodes[u_isol]["x"] + rest.nodes[v_isol]["x"]) / 2,
                (rest.nodes[u_isol]["y"] + rest.nodes[v_isol]["y"]) / 2,
            )
        else:
            geom = edges[0][3].get("geometry", None)
            middle = geom.interpolate(geom.length / 2)

        # Add a node at the middle point
        middle_id = get_new_node_id(graph)
        graph.add_node(middle_id, x=middle.x, y=middle.y, split=True)

        # Add the edges
        for u_parallel, v_parallel, k, data_p, start_node in edges:
            data_p["length"] = data_p["length"] / 2
            # `population` and `area` need to be split up, `cell_id` assigned separately
            # This is what the start_node is for,
            # if the inserted node is the start_node, the cell_id is kept,
            # otherwise it is inverted (usually cell_ids are positive)
            if "population" in data_p:
                data_p["population"] = data_p["population"] / 2
                data_p["area"] = data_p["area"] / 2
            cell_id = data_p.pop("cell_id", None)
            if geom is None:
                graph.add_edge(
                    u_parallel,
                    middle_id,
                    k,
                    **data_p,
                    cell_id=cell_id
                    if cell_id is None
                    else cell_id
                    if u_parallel == start_node
                    else -cell_id,
                )
                graph.add_edge(
                    middle_id,
                    v_parallel,
                    k,
                    **data_p,
                    cell_id=cell_id
                    if cell_id is None
                    else cell_id
                    if v_parallel == start_node
                    else -cell_id,
                )

            else:
                data_p.pop("geometry")
                graph.add_edge(
                    u_parallel,
                    middle_id,
                    k,
                    **data_p,
                    geometry=substring(geom, 0, geom.project(middle)),
                    cell_id=cell_id
                    if cell_id is None
                    else cell_id
                    if u_parallel == start_node
                    else -cell_id,
                )
                graph.add_edge(
                    middle_id,
                    v_parallel,
                    k,
                    **data_p,
                    geometry=substring(geom, geom.project(middle), geom.length),
                    cell_id=cell_id
                    if cell_id is None
                    else cell_id
                    if v_parallel == start_node
                    else -cell_id,
                )
            # Remove the original edge
            graph.remove_edge(u_parallel, v_parallel, k)


def get_new_node_id(graph):
    """Get a new node id that is not yet in the graph.

    Parameters
    ----------
    graph : networkx.classes.multidigraph.MultiDiGraph
        Graph to get the new node id for.

    Returns
    -------
    int
        New node id.

    Notes
    -----
    The node id is generated by the function `uuid4().int`.

    We only accept ids > int64.max because the node ids should be differentiable
    from the osm node ids.
    """
    node_id = uuid4().int
    while node_id in graph.nodes or node_id < 2**63:
        node_id = uuid4().int
    return node_id
