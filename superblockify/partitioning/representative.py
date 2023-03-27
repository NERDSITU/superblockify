"""Methods for finding representative nodes in components and partitions."""
from geopandas.array import GeometryArray
from osmnx import graph_to_gdfs


def set_representative_nodes(components):
    """Find representative nodes in components.

    Works on list of components or partitions dicts, and sets the
    'representative_node_id' attribute, derived from the 'subgraph' attribute.

    It determines the convex hull of the component and returns the node next to the
    representative point of the convex hull.

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

    def _distance_to_rep_point(node_geometry):
        """Return the distance from the node to the representative point."""
        return node_geometry.distance(hull_nodes_reppoint)

    for component in components:
        # Get the nodes as a geodataframe
        nodes = graph_to_gdfs(
            G=component["subgraph"], nodes=True, edges=False, fill_edge_geometry=False
        )

        # 1. create polygon that contains all points
        hull_nodes: GeometryArray = nodes.unary_union.convex_hull  # Polygon geometry

        # 2. find representative point of the polygon: Point geometry
        hull_nodes_reppoint: GeometryArray = hull_nodes.representative_point()

        # note that .centroid() is not guaranteed to be *within* the geometry,
        # while .representative point() is

        # 3. find network node that is closest to the representative point

        component["representative_node_id"] = nodes.geometry.apply(
            _distance_to_rep_point
        ).idxmin()
