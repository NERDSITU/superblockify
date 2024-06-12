"""Utility functions for Partitioners."""

from os import remove
from os.path import exists, join
from uuid import uuid4

from geopandas import GeoDataFrame
from networkx import set_edge_attributes, strongly_connected_components
from numpy import generic
from osmnx import graph_to_gdfs
from osmnx.convert import to_undirected
from pandas import DataFrame
from shapely import Point, Geometry
from shapely.ops import substring

from .representative import find_representative_node_id
from ..config import logger
from ..graph_stats import get_population_area, basic_graph_stats


def save_to_gpkg(
    partitioner,
    save_path=None,
    ltn_boundary=False,
):
    """Save the partitioner's graph and Superblocks to a geodatapackage.

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
    ltn_boundary : bool, optional
        If True, the boundary of the Superblocks will be saved as a polygon into the
        `cell` attribute of the Superblocks layer. For this, the tessellation needs
        to be calculated.

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
        - `representative_node_name`
        - `missing_nodes`: if node is not in any component/partition or sparsified graph
        - y, x, lat, lon, geometry
    - edges
        - `classification`
        - `osmid`, `highway`, `length`, `geometry`
        - (bearing: the bearing of the edge, bearing_90: mod(bearing, 90))
        - (residential: 1 if edge['highway'] is or contains 'residential',
          None otherwise)
    - ltns
        - `classification`
        - `cell`: if `ltn_boundary` is True
        - else `representative_node_point`
        - `population_density` (in people/m^2)
        - `area` (in m^2)
        - `population` (in people)
        - see :func:`osmnx.stats.basic_stats` for more
    - graph_meta
        - `boundary` (by OSM relation)
        - `boundary_crs`
        - `area` (in m^2)
        - `street_orientation_order`
        - `circuity_avg`
        - see :func:`osmnx.stats.basic_stats` for more
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
    parts = None
    if partitioner.components:
        parts = partitioner.components
        logger.info(
            "Using components attribute to save Superblocks to geodatapackage %s",
            filepath,
        )
    elif partitioner.partitions:
        parts = partitioner.partitions
        logger.info(
            "Using partitions attribute to save Superblocks to geodatapackage %s",
            filepath,
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

    # Bake the Superblocks into the graph
    for _, part in enumerate(parts):
        # As part["subgraph"] is connected to partitioner.graph, we can just
        # change the edge attribute in the whole subgraph, applying the Superblock
        set_edge_attributes(part["subgraph"], part["name"], "classification")
    # Sparsified edges are saved with the value "SPARSE" into the "classification"
    # edge attribute
    set_edge_attributes(partitioner.sparsified, "SPARSE", "classification")

    nodes, edges = graph_to_gdfs(partitioner.graph, nodes=True, fill_edge_geometry=True)

    # Components/Partitions
    ltns = _get_ltns(partitioner, nodes, ltn_boundary)

    # Graph meta
    graph_meta = _get_graph_meta(partitioner)

    # list attributes in nodes and edges
    logger.info("Node attributes: %s", nodes.columns)
    logger.info("Edge attributes: %s", edges.columns)
    logger.info("Superblock attributes: %s", ltns.columns)
    logger.info("Graph meta attributes: %s", graph_meta.columns)

    # Remove certain attributes
    # nodes = nodes.drop(columns=[])
    edges = edges.drop(
        columns=[attr for attr in edges.columns if attr in ["component_name"]]
    )

    # For attributes that are lists or other objects, we need to convert them to strings
    for gdfs in [edges, ltns, graph_meta]:
        for col in gdfs.columns:
            if gdfs[col].dtype == "object":
                logger.debug("Converting column %s of type %s to str.", col, type(col))
                gdfs[col] = gdfs[col].astype(str)

    # Save nodes and edges to separate layers/geodataframes of the same geodatapackage
    # if file already exists, remove it
    if exists(filepath):
        remove(filepath)
    nodes.to_file(filepath, layer="nodes", index=False, mode="w")
    logger.info("Saved %d nodes to %s", len(nodes), filepath)
    edges.to_file(filepath, layer="edges", index=False, mode="w")
    logger.info("Saved %d edges to %s", len(edges), filepath)
    ltns.to_file(filepath, layer="ltns", index=False, mode="w")
    logger.info("Saved %d Superblocks to %s", len(ltns), filepath)
    graph_meta.to_file(filepath, layer="graph_meta", index=False, mode="w")
    logger.info("Saved graph meta to %s", filepath)


def _get_ltns(partitioner, nodes, ltn_boundary):
    """Prepare ltn geodataframe for the geopackage export.

    If `ltn_boundary` is True, the ltns are the cells of the tessellation.
    If `ltn_boundary` is False, the ltns are the representative nodes of the
    components/partitions.

    Parameters
    ----------
    partitioner : superblockify.partitioning.BasePartitioner
        The partitioner to get the ltns from.
    nodes : geopandas.GeoDataFrame
        The nodes of the graph.
    ltn_boundary : bool
        Whether to use the cells of the tessellation as ltns or the
        representative nodes of the components/partitions.

    Returns
    -------
    ltns : geopandas.GeoDataFrame
        The ltns as a geodataframe.
    """

    if ltn_boundary:
        partitioner.add_component_tessellation()
    else:
        # add node geometry for representative node from nodes layer
        for _, part in enumerate(partitioner.get_ltns()):
            part["representative_node_point"] = nodes.loc[
                part["representative_node_id"], "geometry"
            ]
    ltns = GeoDataFrame.from_dict(
        partitioner.get_ltns(),
        geometry="cell" if ltn_boundary else "representative_node_point",
        orient="columns",
        crs=partitioner.graph.graph["crs"],
    )
    ltns.rename(columns={"name": "classification"}, inplace=True)
    ltns.drop(
        columns=[attr for attr in ltns.columns if attr in ["subgraph"]], inplace=True
    )
    return ltns


def _get_graph_meta(partitioner):
    """Prepare metadata for the graph for the geopackage export.

    Parameters
    ----------
    partitioner : superblockify.partitioning.BasePartitioner
        The partitioner to get the graph metadata from.

    Returns
    -------
    graph_meta : geopandas.GeoDataFrame
        One line gdf with the graph boundary and all graph attributes.
    """
    return GeoDataFrame.from_dict(
        partitioner.graph.graph,
        orient="columns",
        geometry="boundary",
        crs=partitioner.graph.graph["crs"],
    )


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


def show_graph_stats(graph):
    """Show selected graph statistics.

    Parameters
    ----------
    graph : networkx.classes.multidigraph.MultiDiGraph
        Graph to show statistics for.

    Notes
    -----
    Graph must have basic graph stats added by means of
    :func:`superblockify.graph_stats.basic_graph_stats`.
    """
    logger.info(
        "Graph stats: \n%s",
        DataFrame.from_dict(
            {
                "Number of nodes": graph.graph["n"],
                "Number of edges": graph.graph["m"],
                "Average degree": graph.graph["k_avg"],
                "Circuity average": graph.graph["circuity_avg"],
                "Street orientation order": graph.graph["street_orientation_order"],
                "Date created": graph.graph["created_date"],
                "Projection": graph.graph["crs"],
                "Area by OSM boundary (m²)": graph.graph["area"],
            },
            orient="index",
        ).to_string(),
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
    num_nodes = graph.number_of_nodes()
    # Get the largest strongly connected component
    scc = max(strongly_connected_components(graph), key=len)
    # Remove all nodes that are not in the largest strongly connected component
    for node in list(graph.nodes):
        if node not in scc:
            graph.remove_node(node)
    if graph.number_of_nodes() < num_nodes:
        logger.debug("Removed %d dead ends.", num_nodes - len(graph.nodes))


def split_up_isolated_edges_directed(graph, sparsified):
    """Split up all edges in the directed graph that are isolated disregarding
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
    rest_un = to_undirected(rest)

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
            edges = [
                (*list(rest.edges(v_isol, keys=True, data=True))[deg_v_isol], u_isol)
                for deg_v_isol in range(rest.out_degree(v_isol))
            ]
            edges += [
                (*list(rest.edges(u_isol, keys=True, data=True))[deg_u_isol], v_isol)
                for deg_u_isol in range(rest.out_degree(u_isol))
            ]

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
            data_p["length"] = None if "length" not in data_p else data_p["length"] / 2
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
                    cell_id=(
                        cell_id
                        if cell_id is None
                        else cell_id if u_parallel == start_node else -cell_id
                    ),
                )
                graph.add_edge(
                    middle_id,
                    v_parallel,
                    k,
                    **data_p,
                    cell_id=(
                        cell_id
                        if cell_id is None
                        else cell_id if v_parallel == start_node else -cell_id
                    ),
                )

            else:
                data_p.pop("geometry", None)
                graph.add_edge(
                    u_parallel,
                    middle_id,
                    k,
                    **data_p,
                    geometry=substring(geom, 0, geom.project(middle)),
                    cell_id=(
                        cell_id
                        if cell_id is None
                        else cell_id if u_parallel == start_node else -cell_id
                    ),
                )
                graph.add_edge(
                    middle_id,
                    v_parallel,
                    k,
                    **data_p,
                    geometry=substring(geom, geom.project(middle), geom.length),
                    cell_id=(
                        cell_id
                        if cell_id is None
                        else cell_id if v_parallel == start_node else -cell_id
                    ),
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


def get_key_figures(partitioner):
    """Get key figures of the partitioner.

    Contains the name, city_name, graph_stats, metrics, component stats,
    attribute_label, and attribute_dtype.

    Parameters
    ----------
    partitioner : BasePartitioner
        Partitioner to get the key figures for.

    Returns
    -------
    dict
        Key figures of the partitioner. See code for structure.
    """
    return _make_yaml_compatible(
        {
            "name": partitioner.name,
            "city_name": partitioner.city_name,
            "attribute_label": partitioner.attribute_label,
            "attribute_dtype": partitioner.attribute_dtype,
            "graph_stats": {
                key: value
                for key, value in partitioner.graph.graph.items()
                if not isinstance(value, Geometry)  # discard OSM graph boundary
            },
            "components": [
                # everything of the components/partitions, except the subgraph
                {key: value for key, value in comp.items() if key != "subgraph"}
                for comp in partitioner.get_ltns()
            ],
            "metric": {
                "unit": partitioner.metric.unit,
                "coverage": partitioner.metric.coverage,
                "directness": partitioner.metric.directness,
                "global_efficiency": partitioner.metric.global_efficiency,
                "high_bc_clustering": partitioner.metric.high_bc_clustering,
                "high_bc_anisotropy": partitioner.metric.high_bc_anisotropy,
            },
        }
    )


def _make_yaml_compatible(input_val):
    """Make the dict compatible with the yaml format.

    Represent only with python types, recursively.

    Parameters
    ----------
    dict : dict
        Dictionary to make compatible.

    Returns
    -------
    dict
        Copy of the dict with only python types.
    """
    # Recursively call the function if the value is a dict
    if isinstance(input_val, dict):
        new_dict = {}
        for key, value in input_val.items():
            new_dict[key] = _make_yaml_compatible(value)
        return new_dict
    # Recursively call the function if the value is a list or tuple
    if isinstance(input_val, (list, tuple)):
        return [_make_yaml_compatible(value) for value in input_val]
    # If int, float, str, bool, re-cast it to the same type
    if isinstance(input_val, bool):
        output_val = bool(input_val)
    elif isinstance(input_val, int):
        output_val = int(input_val)
    elif isinstance(input_val, float):
        output_val = float(input_val)
    elif isinstance(input_val, str):
        output_val = str(input_val)
    # Numpy types
    # find out if it is a numpy scalar
    elif isinstance(input_val, generic):
        output_val = input_val.item()
    else:
        output_val = str(input_val)
    return output_val


def reduce_graph(graph, max_nodes):
    """Reduce the graph to have at most `max_nodes` nodes.

    Done by taking the ego graph with not more than `max_nodes` nodes, starting from
    a representative central node. As in
    :func:`superblockify.partitioning.representative.set_representative_nodes`.
    """
    if max_nodes is None or graph.number_of_nodes() <= max_nodes:
        logger.debug(
            "No reduction necessary (%s <= %s)", graph.number_of_nodes(), max_nodes
        )
        return graph
    logger.debug(
        "Reducing graph from %s to %s nodes", graph.number_of_nodes(), max_nodes
    )
    # Find representative node
    rep_node_id = find_representative_node_id(graph)
    # Get ego graph - breadth first search (BFS)
    included = dict(_bfs_egograph(graph, rep_node_id, max_nodes))
    reduced = graph.subgraph(included).copy()
    # Update basic graph stats
    (
        reduced.graph["reduced_population"],
        reduced.graph["reduced_area"],
    ) = get_population_area(reduced)
    try:
        reduced.graph["reduced_population_density"] = (
            reduced.graph["reduced_population"] / reduced.graph["reduced_area"]
        )
    except ZeroDivisionError:
        reduced.graph["reduced_population_density"] = 0

    # Add basic_stats to the graph - prepend "reduced_" to the keys
    if reduced.number_of_edges() > 0:
        reduced.graph.update(
            {
                "reduced_" + key: value
                for key, value in basic_graph_stats(
                    reduced, area=reduced.graph["reduced_area"]
                ).items()
            }
        )
    logger.debug("Reduced graph has %s nodes", reduced.number_of_nodes())
    return reduced


def _bfs_egograph(graph, rep_node_id, max_nodes):
    """Breadth-first search (BFS) to get the ego graph of the representative node.

    Parameters
    ----------
    graph : networkx.classes.multidigraph.MultiDiGraph
        Graph to get the ego graph for.
    rep_node_id : int
        Node id of the seed node.
    max_nodes : int
        Maximum number of nodes in the ego graph.

    Yields
    ------
    (node_id, level) : tuple
        Node id and level of the node in the BFS.

    Notes
    -----
    Modified from
    :func:`networkx.algorithms.shortest_paths.unweighted._single_shortest_path_length`.
    """
    seen = set([rep_node_id])
    nextlevel = [rep_node_id]
    level = 0
    n_total = len(graph.adj)
    # Yield the seed node
    yield (rep_node_id, level)
    # Yield the nodes in the BFS, as long as the sum of the next level and the seen
    # nodes are less than the maximum number of nodes
    while (
        nextlevel
        and len(seen)
        + len([n_next for leaf in nextlevel for n_next in graph.adj[leaf]])
        <= max_nodes
    ):
        level += 1
        thislevel = nextlevel
        nextlevel = []
        for leaf in thislevel:
            for n_next in graph.adj[leaf]:
                if n_next not in seen:
                    seen.add(n_next)
                    nextlevel.append(n_next)
                    # Yield the node id and level
                    yield (n_next, level)
            if len(seen) == n_total:
                return  # pragma: no cover
