"""GHSL IO functions for the population submodule of the superblockify package."""

from io import BytesIO
from os import path
from pathlib import Path
from zipfile import ZipFile

import requests
from geopandas import GeoDataFrame
from rasterio import open as rasopen
from rasterio.enums import Resampling
from rasterio.windows import Window

from ..config import logger, Config


def resample_load_window(file, resample_factor=1, window=None, res_strategy=None):
    """Load and resample a window of a raster file.

    Parameters
    ----------
    file : str
        Path to the raster file. It Can be a tile or the whole raster.
    resample_factor : float, optional
        Factor to resample the window by. Values > 1 increase the resolution
        of the raster, values < 1 decrease the resolution of the raster by that
        factor in each dimension.
    window : rasterio.windows.Window, geopandas.GeoDataFrame, optional
        Window of the raster to resample, by default None.
        When given a GeoDataFrame, the window is the bounding box of the
        GeoDataFrame.
    res_strategy : rasterio.enums.Resampling, optional
        Resampling strategy, by default Resampling.nearest if resample_factor
        > 1 (up-sampling), Resampling.average if resample_factor < 1 (down-sampling).

    Returns
    -------
    raster_rescaled : numpy.ndarray
        Resampled raster.
    res_affine : rasterio.Affine
        Affine transformation of the resampled raster.
    """
    if res_strategy is None:
        if resample_factor < 1:
            res_strategy = Resampling.average
        else:
            res_strategy = Resampling.nearest

    if not isinstance(window, (Window, GeoDataFrame)) and window is not None:
        raise TypeError(
            f"window must be a rasterio.windows.Window or a GeoDataFrame, "
            f"not {type(window)}. Leave empty to load the whole raster."
        )

    with rasopen(file) as src:
        if window is None:
            window = src.window(*src.bounds)
        elif isinstance(window, GeoDataFrame):
            window = src.window(*window.buffer(100).total_bounds)
        # Resample the window
        res_window = Window(
            window.col_off * resample_factor,
            window.row_off * resample_factor,
            window.width * resample_factor,
            window.height * resample_factor,
        )
        # Read the raster while resampling
        raster_rescaled = src.read(
            1,
            out_shape=(1, int(res_window.height), int(res_window.width)),
            resampling=res_strategy,
            window=window,
            masked=True,
            boundless=True,
            fill_value=0,
        )

        # Affine transformation - respects the resampling
        res_affine = src.window_transform(window) * src.transform.scale(
            1 / resample_factor
        )

    return raster_rescaled, res_affine


def get_ghsl(bbox_moll=None):
    """Get the GHSL population raster path(s) for the given bounding box.

    There are two working modes:
        1. `config.FULL_RASTER` is set.
           This path, to the whole GHSL raster, is returned.
        2. Otherwise: With :attr:`bbox_moll` given, the needed raster tile(s) are
           determined, and their paths are returned. If they are not yet in
           :attr:`config.GHSL_DIR`, they are downloaded from the JRC FTP server.

    Parameters
    ----------
    bbox_moll : list, optional
        Boundary of the place in Mollweide projection. [minx, miny, maxx, maxy]
        Needs to be given if the full raster is not available at
        :attr:`config.FULL_RASTER`.

    Returns
    -------
    str or list
        Path(s) to the GHSL raster tile(s).

    Raises
    ------
    ValueError
        If :attr:`bbox_moll` is not given and :attr:`config.FULL_RASTER` is not set.
    ValueError
        If :attr:`config.FULL_RASTER` is invalid.
    ValueError
        If the bounding box has invalid coordinates.
    """
    if Config.FULL_RASTER:
        if not path.isfile(Config.FULL_RASTER):
            raise ValueError(
                f"The given full GHSL raster path {Config.FULL_RASTER} does not exist."
            )
        logger.info("Using the full GHSL raster at %s.", Config.FULL_RASTER)
        return Config.FULL_RASTER
    if bbox_moll is None:
        raise ValueError(
            "The full GHSL raster is not available, and no bounding box was given. "
            "One of the two is needed to determine the needed raster tile(s)."
        )
    logger.info("Using the GHSL raster tiles for the bounding box %s.", bbox_moll)
    # Check if the bounding box is possibly in World Mollweide projection
    #            -9 000 000
    # -18 041 000          +18 041 000
    #            -9 000 000
    if (
        bbox_moll[0] < -180.41e5
        or bbox_moll[1] < -90e5
        or bbox_moll[2] > 180.41e5
        or bbox_moll[3] > 90e5
    ):
        raise ValueError(
            f"The given bounding box {bbox_moll} is not in the World Mollweide "
            "projection."
        )
    urls = get_ghsl_urls(bbox_moll)
    # Download the raster tiles if they are not yet downloaded
    return download_ghsl(urls)


def get_ghsl_urls(bbox_moll):
    """Get the URLs of the GHSL population raster tiles that contain the
    boundary.

    Parameters
    ----------
    bbox_moll : list
        Boundary of the place in Mollweide projection.
        [minx, miny, maxx, maxy]

    Returns
    -------
    list
        URLs of the GHSL population raster tiles that contain the boundary.

    Raises
    ------
    ValueError
        If the bounding box spans more than two tiles in each dimension.
    ValueError
        If the bounding box spans empty tile.

    Notes
    -----
    Bounding boxes spanning areas larger than two tiles in each dimension are not
    supported. Use the whole raster instead.
    """
    corners = [
        (bbox_moll[0], bbox_moll[1]),  # Lower left
        (bbox_moll[0], bbox_moll[3]),  # Upper left
        (bbox_moll[2], bbox_moll[1]),  # Lower right
        (bbox_moll[2], bbox_moll[3]),  # Upper right
    ]
    tiles = {row_col(lat, long) for long, lat in corners}
    logger.debug("The bounding box spans the tiles %s.", tiles)
    if len(tiles) > 1:
        # that the difference between the min and max row/col is maximally 1
        if max(tiles, key=lambda x: x[0])[0] - min(tiles, key=lambda x: x[0])[0] > 1:
            raise ValueError(
                "The bounding box spans more than two tiles in the row dimension."
            )
        if max(tiles, key=lambda x: x[1])[1] - min(tiles, key=lambda x: x[1])[1] > 1:
            raise ValueError(
                "The bounding box spans more than two tiles in the column dimension."
            )
    # Convert to urls
    tiles = {
        f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
        f"/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/"
        f"GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R{row}_C{col}.zip"
        for row, col in tiles
    }
    return list(tiles)


def row_col(y_moll, x_moll):
    """Resolves the row and column of the GHS-POP raster tile that contains
    the given point.

    Parameters
    ----------
    y_moll : float
        y-coordinate of the point in Mollweide projection.
    x_moll : float
        x-coordinate of the point in Mollweide projection.

    Returns
    -------
    col, row : int, int
        Column and row of the tile.

    Notes
    -----
    The GHS-POP raster tiles are each 100km x 100km on the Mollewide projection.
    Latitude has its origin at the equator, but latitude has an offset of -41km.
    This function was reversely engineered from the GHS-POP raster tile names
    found on the
    `JRC FTP server <https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/>`_
    (see `dataset overview <https://ghsl.jrc.ec.europa.eu/download.php>`_).
    """
    row = int((9000000 - y_moll) / 1e6) + 1
    col = int((18041000 + x_moll) / 1e6) + 1
    return row, min(col, 36)


def download_ghsl(urls, save_dir=Config.GHSL_DIR, timeout=Config.DOWNLOAD_TIMEOUT):
    """Download the GHSL population raster tile.

    Check if the raster tiles are already downloaded, and if not, download and unpack
    them. Create the save directory if it does not exist.

    Parameters
    ----------
    urls : str or list
        URL(s) of the raster tile(s).
    save_dir : str, optional
        Directory to save to and look for the raster tile(s), by default GHSL_DIR.
    timeout : int or float, optional
        Timeout in seconds for the download, by default
        :attr:`config.DOWNLOAD_TIMEOUT`.

    Returns
    -------
    str or list
        Path(s) to the downloaded raster tile(s).

    Raises
    ------
    ValueError
        A given URL does not exist.
    ValueError
        A given URL does not return a zip file.

    Notes
    -----
    The GHSL raster tiles are between 1.4M and 99M in size.
    The sum of all tiles is 4.9G.
    """
    if not path.exists(save_dir):
        logger.debug("Creating GHSL directory at %s...", save_dir)
        Path(save_dir).mkdir(exist_ok=True, parents=True)
    files = []
    for url in urls if isinstance(urls, (list, tuple)) else [urls]:
        # Check if file already exists at the save_dir
        # - same as url but without the path and .tif
        filepath = path.join(save_dir, path.basename(url)[:-4])
        if path.exists(filepath + ".tif"):
            files.append(filepath + ".tif")
            continue
        # Download zip file to save_dir
        logger.debug("Downloading %s to %s...", url, save_dir)
        try:
            response = requests.get(url, timeout=timeout)
        except requests.exceptions.ConnectionError as exc:
            raise ValueError(f"Connection error for {url}.") from exc
        # Check if the URL exists and returns a zip file
        if response.status_code != 200:
            raise ValueError(f"The URL {url} does not exist.")
        if response.headers["Content-Type"] != "application/zip":
            raise ValueError(f"The URL {url} does not return a zip file.")

        # Extract the raster tile with expected name
        logger.debug(
            "Extracting %s to %s...", path.basename(url)[:-4] + ".tif", save_dir
        )
        with ZipFile(BytesIO(response.content)) as zip_file:
            # Only extract path.basename(url)[:-4]+".tif"
            zip_file.extract(path.basename(url)[:-4] + ".tif", save_dir)
            # Trust that the zip file contains that .tif file

        files.append(filepath + ".tif")
    return files[0] if len(files) == 1 else files
