"""Methods for finding representative nodes in components and partitions."""

from geopandas.array import GeometryArray
from osmnx import graph_to_gdfs


def set_representative_nodes(components):
    """Find representative nodes in components.

    Works on list of components or partitions dicts, and sets the
    'representative_node_id' attribute, derived from the 'subgraph' attribute.

    It determines the convex hull of the component and returns the node next to the
    representative point of the convex hull.
    If a component consists out of one edge connected to the sparsified graph, the
    representative node is the one that has the lower degree.

    Parameters
    ----------
    components : list of dict
        The components to find representative nodes for, needs to have a 'subgraph'
        attribute.

    Notes
    -----
    The method works in-place. It sets the 'representative_node_id' attribute of the
    components.

    """
    for component in components:
        if component["m"] == 1 and component["n"] == 2:
            component["representative_node_id"] = min(
                component["subgraph"].nodes, key=component["subgraph"].degree
            )
            continue

        component["representative_node_id"] = find_representative_node_id(
            component["subgraph"]
        )

        component["subgraph"].nodes[component["representative_node_id"]][
            "representative_node_name"
        ] = component["name"]


def find_representative_node_id(graph):
    """Find representative node in a graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to find a representative node for.

    Returns
    -------
    representative_node_id : int
        The id of the representative node.
    """
    # Get the nodes as a geodataframe
    nodes = graph_to_gdfs(G=graph, nodes=True, edges=False, fill_edge_geometry=False)

    # 1. create polygon that contains all points
    hull_nodes: GeometryArray = nodes.union_all().convex_hull  # Polygon geometry

    # 2. find representative point of the polygon: Point geometry
    hull_nodes_reppoint: GeometryArray = hull_nodes.representative_point()

    # note that .centroid() is not guaranteed to be *within* the geometry,
    # while .representative point() is

    # 3. find network node that is closest to the representative point

    return nodes.geometry.apply(
        lambda node_geometry: node_geometry.distance(hull_nodes_reppoint)
    ).idxmin()
