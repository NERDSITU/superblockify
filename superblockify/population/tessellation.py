"""Graph Tessellation for the population submodule of the superblockify package."""

from datetime import timedelta, datetime

from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from numpy import vstack, linspace
from osmnx import graph_to_gdfs, plot_graph
from osmnx.projection import is_projected
from pandas import Series
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module
from shapely import Polygon, get_coordinates, line_interpolate_point, polygons
from shapely.lib import GEOSException

from ..config import logger


def add_edge_cells(graph, **tess_kwargs):
    """Add edge tessellation cells to edge attributes in the graph.

    Tessellates the graph into plane using a Voronoi cell approach.
    Function writes to edge attribute `cells` of the graph in-place.
    Furthermore, `cell_id` is added to the edge attributes, for easier summary of
    statistics later.

    The approach was developed inspired by the :class:`momepy.Tessellation` class
    and tessellates with :class:`scipy.spatial.Voronoi`.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to tessellate.
    **tess_kwargs
        Keyword arguments for the
        :func:`superblockify.population.tessellation.get_edge_cells` function.

    Raises
    ------
    ValueError
        If the graph is not in a projected coordinate system.
    ValueError
        If the limit and the edge points are disjoint.

    Notes
    -----
    The graph must be in a projected coordinate system.
    """
    edge_cells = get_edge_cells(graph, **tess_kwargs)
    # Add cells to graph (in-place)
    logger.debug("Adding cells to graph.")
    for edge_keys, polygon in edge_cells.geometry.items():
        for edge_key in edge_keys:
            graph.edges[edge_key]["cell"] = polygon
            graph.edges[edge_key]["cell_id"] = edge_cells.index.get_loc(edge_keys)


def get_edge_cells(graph, limit=None, segment=25, show_plot=False):
    """Get edge tessellation cells for the graph.

    Tessellates the graph into plane using a Voronoi cell approach.

    The approach was developed inspired by the :class:`momepy.Tessellation` class
    and tessellates with :class:`scipy.spatial.Voronoi`.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to tessellate.
    limit : shapely.geometry.Polygon or None
        The limit of the tessellation. Must be in the same CRS as the graph.
        If None, it will be calculated as the exterior of the 100m buffered unary
        union of the graph's edges.
    segment : float
        The maximum distance for the point interpolation. Default is 25.
    show_plot : bool
        If True, a plot of the tessellation will be shown. Default is False.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with the tuple of edge keys as index and the tessellation
        cells as geometry.

    Raises
    ------
    ValueError
        If the graph is not in a projected coordinate system.
    ValueError
        If the limit and the edge points are disjoint.

    Notes
    -----
    The graph must be in a projected coordinate system.
    """
    # Check if the graph is projected
    if not is_projected(graph.graph["crs"]):
        raise ValueError(
            "The graph must be in a projected coordinate system. "
            "Use `osmnx.project_graph` to project the graph."
        )
    logger.info(
        "Calculating edge cells for graph with %d edges.", graph.number_of_edges()
    )
    start_time = datetime.now()
    edges = get_edge_polygons(graph)
    logger.debug("Prepared %d edge polygons. Next, interpolating points.", len(edges))
    # Make limit polygon
    if limit is None:
        limit = Polygon(edges.union_all().buffer(100).exterior)
    else:
        # Check if limit and edges are disjoint
        if limit.disjoint(edges.union_all()):
            raise ValueError(
                "The limit and the edge points are disjoint. "
                "Please provide a limit that intersects the edges."
            )

    # Convert edges to points and their multiindex
    edge_points, edge_indices = edges_to_points(edges, segment=segment)

    # Convert limit hull to points
    hull_points = line_interpolate_point(
        limit.boundary,
        linspace(
            0.1,
            limit.length - 0.1,
            num=int((limit.length - 0.1) // segment),
        ),
    )
    # Add to point array
    edge_points = vstack([edge_points, get_coordinates(hull_points)])
    edge_indices += [-1] * len(hull_points)

    # Create Voronoi tessellation
    logger.debug("Prepared %d points. Creating Voronoi tessellation.", len(edge_points))
    edge_voronoi_diagram = Voronoi(edge_points)

    # Reconstruct edge cells from Voronoi point tessellation
    logger.debug("Reconstructing edge cells.")
    edge_cells = reconstruct_edge_cells(
        edge_voronoi_diagram, edge_indices, graph.graph["crs"]
    )

    # Plot tessellation
    if show_plot:
        fig, axe = plt.subplots(figsize=(8, 8))
        edge_cells.to_crs(graph.graph["crs"]).plot(
            ax=axe, cmap="tab20", edgecolor="white", alpha=0.5
        )
        plot_graph(graph, ax=axe, node_size=0, edge_color="black", edge_linewidth=0.5)
        axe.set_axis_off()
        axe.set_title("Edge tessellation")
        fig.tight_layout()

    logger.info(
        "Tessellated %d edge cells in %s.",
        len(edge_cells),
        timedelta(seconds=(datetime.now() - start_time).seconds),
    )
    return edge_cells


def get_edge_polygons(graph):
    """Prepare edge polygons for tessellation.

    This returns a GeoDataFrame where edges with same start and end node are
    merged, if their geometry is equal or a reversed version of the other.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to tessellate.

    Returns
    -------
    edges : geopandas.GeoDataFrame
        The edges with their polygons.
    """
    edges = graph_to_gdfs(
        graph, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True
    )

    # Merge two columns, when the geometry of one is equal or the reverse of the other
    # 1. Group edge ig (u, v) == (v, u) or (u, v) == (u, v)
    # Match by node pair (u, v) where u < v
    # get flat index of the first match
    edges["edge_id"] = edges.index.to_flat_index()
    # sort the edge_id tuples
    edges["node_pair"] = edges["edge_id"].apply(lambda x: tuple(sorted(x[:2])))
    # 2. Aggregate if the geometry is equal or the reverse of the other
    # Merge columns if geometry_1 == geometry_2 or geometry_1 == geometry_2.reverse()
    # reverse geometry if x_start >= x_end (analogous to sorted node_pair)
    edges["geometry"] = edges["geometry"].apply(
        lambda x: x if x.coords[0] < x.coords[-1] else x.reverse()
    )
    # 3. Group by node_pair and geometry, aggregate edge_id
    edges = (
        edges.groupby(["node_pair", "geometry"])
        .agg(
            {
                "edge_id": tuple,  # tuple of edge indices
            }  # tuple, because it is immutable and thus hashable for indexing
        )
        .reset_index()
    )
    edges.set_index("edge_id", inplace=True)
    edges.drop(columns=["node_pair"], inplace=True)
    return GeoDataFrame(edges, geometry="geometry", crs=graph.graph["crs"])


def edges_to_points(edges, segment=25):
    """Convert edges to points.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        The edges to convert to points.
    segment : float
        The maximum distance for the point interpolation. Default is 25.

    Returns
    -------
    points : geopandas.GeoDataFrame
        The points.
    indices : list of int
        The indices of the points in the edges.

    Notes
    -----
    The points are interpolated along the edges with a maximum distance of `segment`.
    """
    edge_points = []
    edge_indices = []
    # iterate over street_id, geometry
    for idx, geometry in edges["geometry"].items():
        if geometry.length < 2 * segment:
            # for edges that would result in no or one point, take the middle
            pts = [line_interpolate_point(geometry, 0.5, normalized=True)]
        else:
            # interpolate points along the line
            pts = line_interpolate_point(
                geometry,
                linspace(
                    0.1,
                    geometry.length - 0.1,
                    num=int((geometry.length - 0.1) // segment),
                ),
            )  # offset to keep nodes out
        edge_points.append(get_coordinates(pts))
        # get multi_indices of first row
        edge_indices += [idx] * len(pts)

    points = vstack(edge_points)

    return points, edge_indices


def reconstruct_edge_cells(voronoi_diagram, indices, crs):
    """Reconstruct edge cells from a Voronoi diagram.

    Regions with the hull index `-1` are discarded.

    Parameters
    ----------
    voronoi_diagram : scipy.spatial.Voronoi
        The Voronoi diagram to reconstruct.
    indices : list
        The indices of the points in the Voronoi diagram.
    crs : value
        The CRS used for the GeoDataFrame. Must be the same as the graph.
        Anything compatible with
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`.

    Returns
    -------
    cells : geopandas.GeoDataFrame
        The Voronoi cells by their indices.
    """
    # Construct cell polygons for each of the dense points
    point_vertices = Series(voronoi_diagram.regions).take(voronoi_diagram.point_region)
    point_polygons = []
    for region in point_vertices:
        if -1 not in region:
            point_polygons.append(polygons(voronoi_diagram.vertices[region]))
        else:
            point_polygons.append(None)

    # Create GeoDataFrame with cells and edge multiindex
    poly_gdf = GeoDataFrame(
        geometry=point_polygons,
        crs=crs,
        index=indices,
        columns=["geometry"],
    )
    # Drop cells that are outside the boundary
    poly_gdf = poly_gdf.loc[poly_gdf.index != -1]
    # Dissolve cells by edge multiindex
    try:
        return poly_gdf.dissolve(by=poly_gdf.index)
    except GEOSException as err:  # pragma: no cover
        raise ValueError(
            "The tessellation might contain invalid geometries. "
            "Try increasing the segment length."
        ) from err
