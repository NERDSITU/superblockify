"""Functions for calculating population metrics, using the GHS-POP dataset."""


def get_ghsl_urls(bbox_moll, res=100):
    """Get the URLs of the GHSL population raster tiles that contain the
    boundary.

    Parameters
    ----------
    bbox_moll : list
        Boundary of the place in Mollweide projection.
        [minx, miny, maxx, maxy]
    res : int, optional
        Resolution of the raster in meters (100 or 10), by default 100.
    """
    corners = [
        (bbox_moll[0], bbox_moll[1]),  # Lower left
        (bbox_moll[0], bbox_moll[3]),  # Upper left
        (bbox_moll[2], bbox_moll[1]),  # Lower right
        (bbox_moll[2], bbox_moll[3]),  # Upper right
    ]
    # Empty set of URLs
    urls = set()
    # Check what tile(s) the boundary corners are in
    for corner in corners:
        # Get the row and column of the tile
        row, col = row_col(corner[1], corner[0])
        # Get the URL of the tile
        url = (
            f"https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
            f"GHSL/GHS_BUILT_S_GLOBE_R2023A"
            f"/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_{res}/V1-0/tiles"
            f"/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_{res}_V1_0_R{row}_C{col}.zip"
        )
        # Add the URL to the set of URLs
        urls.add(url)
    return list(urls)


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
