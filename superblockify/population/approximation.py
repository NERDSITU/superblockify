"""Population approximation for the superblockify package.

See 20230608-cell_population.ipynb notebook for details.
"""
from geopandas import GeoDataFrame
from numpy import float32, sum as npsum
from rasterio import open as rasopen
from rasterio.features import shapes
from shapely import STRtree
from shapely.geometry import shape

from .ghsl import get_ghsl, resample_load_window
from .tessellation import get_edge_cells
from ..config import logger


def add_edge_population(graph, overwrite=False, **tess_kwargs):
    """Add edge population to edge attributes in the graph.

    Calculates the population and area of the edges. First tessellates the edges
    and then determines the population with GHSL data. Function writes to edge
    attributes `population` and `area` of the graph in-place.
    Furthermore, `cell_id` is added to the edge attributes, for easier summary of
    statistics later. The graph attribute `edge_population` is set to True.
    With this information, population densities can be calculated for arbitrary
    subsets of edges.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to tessellate.
    overwrite : bool, optional
        If True, overwrite existing population and area attributes. Only depends on
        the graph attribute `edge_population` and not on the actual attributes.
    **tess_kwargs
        Keyword arguments for the
        :func:`superblockify.population.tessellation.add_edge_cells` function.

    Raises
    ------
    ValueError
        If the graph already has population and area attributes and `overwrite` is
        False.
    ValueError
        If the graph is not in a projected coordinate system.
    ValueError
        If the limit and the edge points are disjoint.

    Notes
    -----
    The graph must be in a projected coordinate system.
    """

    if graph.graph.get("edge_population", False) is True and overwrite is False:
        raise ValueError(
            "The graph already has population and area attributes. "
            "Use `overwrite=True` to overwrite them."
        )

    edge_population = get_edge_population(graph, **tess_kwargs)
    logger.debug("Adding population and area to edges.")
    for edge_keys, population, geometry in edge_population[
        ["population", "geometry"]
    ].itertuples():
        for edge_key in edge_keys:
            # Absolute population for area enclosed in edge cell (people)
            graph.edges[edge_key]["population"] = float32(population)
            # Area of edge cell (m²)
            graph.edges[edge_key]["area"] = float32(geometry.area)
            # Cell ID of edge cell
            graph.edges[edge_key]["cell_id"] = edge_population.index.get_loc(edge_keys)
    # Note in graph attributes that population has been added
    graph.graph["edge_population"] = True


def get_population_area(graph):
    """Calculate the population of a graph or subgraph.

    Calculates the population and area of the graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        Graph or subgraph. Must have edge attributes `population`, `area`
        and `cell_id`.

    Returns
    -------
    population : float
        Population of the subgraph.
    area : float
        Area of the subgraph.

    Raises
    ------
    ValueError
        If the graph does not have the population attributes.
    """
    # Check if the graph has edges
    if graph.number_of_edges() == 0:
        return 0.0, 0.0
    # Check if the graph has population attributes
    if graph.graph.get("edge_population", False) is False:
        raise ValueError(
            "The graph does not have the population attributes. "
            "Use `add_edge_population` to add them."
        )
    # Get population, area and cell_id of edges
    population = []
    area = []
    cell_id = []
    for _, _, data in graph.edges(data=True):
        if data["cell_id"] not in cell_id and isinstance(
            data["population"], (float, float32)
        ):
            population.append(data["population"])
            area.append(data["area"])
            cell_id.append(data["cell_id"])
    return npsum(population), npsum(area)


def get_edge_population(graph, **tess_kwargs):
    """Get edge population for the graph.

    Calculates the population and area of the edge. First tessellates the edges
    and then determines the population with GHSL data.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to tessellate.
    **tess_kwargs
        Keyword arguments for the
        :func:`superblockify.population.tessellation.add_edge_cells` function.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame with the tuple of edge keys as index and the population
        and area of the edge as columns, as well as the tessellation cells as
        geometry.
        The CRS will be in World Mollweide.

    Raises
    ------
    ValueError
        If the graph is not in a projected coordinate system.
    ValueError
        If the limit and the edge points are disjoint.

    Notes
    -----
    The graph must be in a projected coordinate system.
    Output CRS is World Mollweide.
    It uses the STRtree index to speed up the intersection. [1]_

    References
    ----------
    .. [1] Leutenegger, Scott T.; Edgington, Jeffrey M.; Lopez, Mario A.
       (February 1997). "STR: A Simple and Efficient Algorithm for
       R-Tree Packing".
       https://ia600900.us.archive.org/27/items/nasa_techdoc_19970016975/19970016975.pdf
    """

    edge_cells = get_edge_cells(graph, **tess_kwargs)
    # Project to World Mollweide
    edge_cells = edge_cells.to_crs("World Mollweide")
    # Build STRtree index
    logger.debug("Building STRtree index.")
    edge_cells_index = STRtree(edge_cells.geometry)
    bbox_moll = edge_cells.unary_union.buffer(100).bounds
    ghsl_file = get_ghsl(bbox_moll)

    with rasopen(ghsl_file) as src:
        load_window = src.window(*bbox_moll)

    ghsl_polygons = load_ghsl_as_polygons(ghsl_file, window=load_window)

    # Distributing the population over the road cells
    # Initializing population column
    edge_cells["population"] = 0
    # Query all geometries at once edge_cells_sindex.query(ghsl_polygons["geometry"])
    edge_cells_loc_pop = edge_cells.columns.get_loc("population")
    logger.debug("Distributing population over road cells.")
    for pop_cell_idx, road_cell_idx in edge_cells_index.query(
        ghsl_polygons["geometry"]
    ).T:
        # get the intersection of the road cell and the polygon
        intersection = edge_cells.iloc[road_cell_idx]["geometry"].intersection(
            ghsl_polygons.iloc[pop_cell_idx]["geometry"]
        )
        # get the population of the polygon
        population = ghsl_polygons.iloc[pop_cell_idx]["population"]
        # query by row number of edge_cells, as edge_cells has different index
        edge_cells.iat[road_cell_idx, edge_cells_loc_pop] += (
            population
            * intersection.area
            / ghsl_polygons.iloc[pop_cell_idx]["geometry"].area
        )
    return edge_cells


def load_ghsl_as_polygons(file, window=None):
    """Get polygonized GHSL data.

    Polygonizes the GHSL population raster data and returns the population in a
    GeoDataFrame. Area with no population is not included.

    Parameters
    ----------
    file : str
        Path to the raster file. It Can be a tile or the whole raster.
    window : rasterio.windows.Window, optional
        Window of the raster to resample.
        If None, the whole raster will be loaded.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame derived from the GHSL population raster data.
        Includes geometry and population columns.

    Notes
    -----
    When not passing a window, the whole raster will be loaded. Make sure the raster
    is not too big.
    """
    logger.debug("Loading GHSL data for window %s from file %s.", window, file)
    ghsl_unsampled, affine_unsampled = resample_load_window(file=file, window=window)
    # convert to float32
    ghsl_unsampled = ghsl_unsampled.astype(float32)
    # Make shapes
    ghsl_shapes = [
        {"population": pop, "geometry": shp}
        for _, (shp, pop) in enumerate(
            shapes(ghsl_unsampled, transform=affine_unsampled)
        )
        if pop > 0
    ]
    ghsl_polygons = GeoDataFrame(
        geometry=[shape(geom["geometry"]) for geom in ghsl_shapes],
        data=[geom["population"] for geom in ghsl_shapes],
        columns=["population"],
        crs="World Mollweide",
    )
    return ghsl_polygons
