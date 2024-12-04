"""Population approximation for the superblockify package.

See reference notebook for a detailed description of the population approximation.
"""

from functools import partial
from multiprocessing import Pool

from geopandas import GeoDataFrame
from numpy import float32, sum as npsum, zeros
from rasterio import open as rasopen
from rasterio.features import shapes
from shapely import STRtree
from shapely.geometry import shape
from tqdm import tqdm

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
        :func:`superblockify.population.tessellation.get_edge_cells` function.

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


def get_edge_population(graph, batch_size=10000, **tess_kwargs):
    """Get edge population for the graph.

    Calculates the population and area of the edge. First tessellates the edges
    and then determines the population with GHSL data.
    The population distribution process is parallelized with multiprocessing in
    batches of edges.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to tessellate.
    batch_size : int, optional
        Number of edges to process in one batch. By default, 10000.
        It must be greater than 0. If it is greater than the number of edges,
        all edges are processed in one batch.
    **tess_kwargs
        Keyword arguments for the
        :func:`superblockify.population.tessellation.get_edge_cells` function.

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
        If the batch size is not greater than 0.
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
    if not isinstance(batch_size, (float, int)):
        raise ValueError(f"Batch size must be numeric, but is {batch_size}.")
    if batch_size <= 0:
        raise ValueError(f"Batch size must be greater than 0, but is {batch_size}.")

    edge_cells = get_edge_cells(graph, **tess_kwargs)
    # Project to World Mollweide
    edge_cells = edge_cells.to_crs("World Mollweide")
    bbox_moll = edge_cells.union_all().buffer(100).bounds
    ghsl_file = get_ghsl(bbox_moll)

    with rasopen(ghsl_file) as src:
        load_window = src.window(*bbox_moll)

    ghsl_polygons = load_ghsl_as_polygons(ghsl_file, window=load_window)
    # Build STRtree index
    logger.debug("Building STRtree index.")
    ghsl_polygons_index = STRtree(ghsl_polygons.geometry)

    # Add columns for population and area
    edge_cells["population"] = 0.0
    batch_size = int(min(batch_size, len(edge_cells)))
    with Pool() as pool:
        slices = (
            slice(
                n_batch * batch_size, min((n_batch + 1) * batch_size, len(edge_cells))
            )
            for n_batch in range(0, len(edge_cells) // batch_size + 1)
        )

        population_sums = list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        _population_fraction_list_sliced,
                        ghsl_polygons["geometry"].values,
                        ghsl_polygons["population"].values,
                        ghsl_polygons_index,
                        edge_cells["geometry"].values,
                    ),
                    slices,
                ),
                desc="Distributing population over road cells",
                total=len(ghsl_polygons) // batch_size + 1,
                unit="Cells",
                unit_scale=batch_size,
                unit_divisor=batch_size,
            )
        )

        # write the results to the dataframe
        for _, (cell_slice, population) in enumerate(population_sums):
            edge_cells.loc[edge_cells.index[cell_slice], "population"] = population

    return edge_cells


# Marked as `no cover` as it is tested, but as a forked process with `multiprocessing`
def population_fraction(ghsl_polygon, population, road_cell):  # pragma: no cover
    """Function returns fractional population count between road_cell and
    ghsl_polygon.

    Parameters
    ----------
    ghsl_polygon : shapely.geometry.Polygon
        Polygon of GHSL cell.
    population : float
        Population of GHSL cell.
    road_cell : shapely.geometry.Polygon
        Polygon of road cell.

    Returns
    -------
    float
        Fractional population count between road_cell and ghsl_polygon.
    """
    intersection = road_cell.intersection(ghsl_polygon)
    return population * intersection.area / ghsl_polygon.area


def _population_fraction_list(
    ghsl_polygons, ghsl_populations, overlap_index_pairs, road_cell_geometries
):  # pragma: no cover
    """Function returns population count for each road cell in road_cell_geometries

    Parameters
    ----------
    ghsl_polygons : list of shapely.geometry.Polygon
        List of GHSL cells.
    ghsl_populations : list of float
        List of GHSL populations.
    overlap_index_pairs : ndarray with shape (2, n)
        Array of indices of overlapping road cells and GHSL cells.
        The first row contains the indices of the road cells, and the second row
        contains the indices of the GHSL cells.
    road_cell_geometries : list of shapely.geometry.Polygon

    Returns
    -------
    ndarray with shape (n,)
        Array of population counts for each road cell in road_cell_geometries.
    """
    population = zeros(len(road_cell_geometries))
    for road_cell_idx, pop_cell_idx in overlap_index_pairs:
        population[road_cell_idx] += population_fraction(
            ghsl_polygons[pop_cell_idx],
            ghsl_populations[pop_cell_idx],
            road_cell_geometries[road_cell_idx],
        )
    return population


def _population_fraction_list_sliced(
    ghsl_polygons, ghsl_populations, ghsl_polygons_index, road_cell_geometries, slice_n
):  # pragma: no cover
    """Function for the parallelization of _population_fraction_list.

    Works like :func:`_population_fraction_list`, but takes all the road cells
    and only determines the population for the road cells in slice_n.

    Parameters
    ----------
    ghsl_polygons : list of shapely.geometry.Polygon
        List of GHSL cells.
    ghsl_populations : list of float
        List of GHSL populations.
    ghsl_polygons_index : shapely.strtree.STRtree
        STRtree index of ghsl_polygons.
    road_cell_geometries : list of shapely.geometry.Polygon
        List of road cells.
    slice_n : slice
        Slice of road cells to determine the population for.

    Returns
    -------
    slice, ndarray with shape (n,)
        Slice of road cells and array of population counts for each road cell in
        road_cell_geometries[slice_n].
    """
    return slice_n, _population_fraction_list(
        ghsl_polygons,
        ghsl_populations,
        ghsl_polygons_index.query(
            road_cell_geometries[slice_n], predicate="intersects"
        ).T,
        road_cell_geometries[slice_n],
    )


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
