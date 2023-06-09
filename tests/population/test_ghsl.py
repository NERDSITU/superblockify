"""Tests for the ghsl module."""
from os.path import join, isfile
from shutil import rmtree

import pytest
from affine import Affine
from geopandas import GeoDataFrame
from numpy import ndarray

from superblockify.config import TEST_DATA_PATH
from superblockify.population import ghsl


@pytest.mark.parametrize(
    "resample_factor, no_window",
    [
        (1 / 2, False),
        (1, False),
        (2, False),
        (1 / 100, True),
    ],
)
def test_resample_load_window_gdf(resample_factor, no_window, test_one_city_copy):
    """Test loading and resampling a gdf window of a raster file."""
    _, graph = test_one_city_copy
    # Get mollweide bounding box
    boundary_polygon = graph.graph["boundary"]
    crs = graph.graph["boundary_crs"]
    boundary_gdf = GeoDataFrame(geometry=[boundary_polygon], crs=crs).to_crs(
        "World Mollweide"
    )
    ghsl_raster_file = ghsl.get_ghsl(list(boundary_gdf.bounds.values[0]))
    raster_rescaled, res_affine = ghsl.resample_load_window(
        ghsl_raster_file,
        resample_factor=resample_factor,
        window=boundary_gdf if not no_window else None,
    )
    assert isinstance(raster_rescaled, ndarray)
    assert isinstance(res_affine, Affine)


@pytest.mark.parametrize("window", [True, False, 1, 1.0, "test"])
def test_resample_load_window_gdf_wrong_type(window, test_one_city_copy):
    """Test loading and resampling a gdf window with wrong type."""
    _, graph = test_one_city_copy
    # Get mollweide bounding box
    boundary_polygon = graph.graph["boundary"]
    crs = graph.graph["boundary_crs"]
    boundary_gdf = GeoDataFrame(geometry=[boundary_polygon], crs=crs).to_crs(
        "World Mollweide"
    )
    ghsl_raster_file = ghsl.get_ghsl(list(boundary_gdf.bounds.values[0]))
    with pytest.raises(TypeError):
        ghsl.resample_load_window(
            ghsl_raster_file,
            resample_factor=1,
            window=window,
        )


@pytest.mark.parametrize(
    "y_moll, x_moll, row, col",
    [
        (-1, -1, 10, 19),
        (-1, 1, 10, 19),
        (1, 1, 9, 19),
        (1, -1, 9, 19),
        (-1, -40000, 10, 19),
        (-1, -42000, 10, 18),
        (-1, 958000, 10, 19),
        (-1, 959001, 10, 20),
        (-1, -18040999, 10, 1),
        (-1, -17040999, 10, 2),
        (-1, 16958999, 10, 35),
        (-1, 16959001, 10, 36),
        (-1, 18040999, 10, 36),
        (9.9e5, -1e5, 9, 18),
        (9.9e5, 1e5, 9, 19),
        (10.1e5, 1e5, 8, 19),
        (10.1e5, -1e5, 8, 18),
        (89e5, -179e5, 1, 1),
        (89e5, 179e5, 1, 36),
        (-89e5, 179e5, 18, 36),
        (-89e5, -179e5, 18, 1),
        (6064791.288371984, 553663.6837159488, 3, 19),
        (6087463.855133052, 553663.6837159488, 3, 19),
        (6064791.288371984, 576088.653201773, 3, 19),
        (6087463.855133052, 576088.653201773, 3, 19),
        (4.4711754e5, -74.2235137e5, 9, 11),
        (4.4711754e5, -74.0102577e5, 9, 11),
        (4.8331695e5, -74.0102577e5, 9, 11),
        (4.8331695e5, -74.2235137e5, 9, 11),
    ],
)
def test_row_col(y_moll, x_moll, row, col):
    """Test the row_col function."""
    assert ghsl.row_col(y_moll, x_moll) == (row, col)


@pytest.mark.parametrize(
    "bbox_moll, urls",
    [
        (
            [1, 1, 2, 2],
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C19.zip",
            ],
        ),
        (
            [-1, -1, 1, 1],
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C19.zip",
            ],
        ),
        (
            [-1, -1, 1, 1],
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C19.zip",
            ],
        ),
        (
            [-42000, -1, 0, 1],
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C18.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
                "/GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C18.zip",
            ],
        ),
    ],
)
def test_get_ghsl_urls(bbox_moll, urls):
    """Test the get_GHSL_urls function."""
    assert set(ghsl.get_ghsl_urls(bbox_moll)) == set(urls)


@pytest.mark.parametrize(
    "bbox_moll",
    [
        [0, 0, 2.1e6, 1],
        [0, 0, 7.1e6, 1],
        [-7e6, 0, -4e6, 1],
        [0, 0, 1, 2.1e6],
        [0, 0, 1, 7.1e6],
        [0, -7e6, 1, -4e6],
        [-4e6, 3e6, 2e6, 6e6],
    ],
)
def test_get_ghsl_urls_invalid_bbox(bbox_moll):
    """Test the get_GHSL_urls function with invalid bbox."""
    with pytest.raises(ValueError):
        ghsl.get_ghsl_urls(bbox_moll)


BASE_PATH = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A"
    "/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/"
)


@pytest.mark.parametrize(
    "urls",
    [
        BASE_PATH + "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R2_C30.zip",
        [
            BASE_PATH + "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R2_C30.zip",
            BASE_PATH + "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R7_C2.zip",
        ],
    ],
)
def test_download_ghsl(urls, _delete_ghsl_tifs):
    """Test the download_ghsl function."""
    filepaths = ghsl.download_ghsl(urls, save_dir=TEST_DATA_PATH)
    assert (
        isinstance(filepaths, list)
        and isinstance(urls, list)
        or isinstance(filepaths, str)
        and isinstance(urls, str)
    )
    # Check files exist in save_dir
    for filepath in filepaths if isinstance(filepaths, list) else [filepaths]:
        assert isfile(filepath)


BASE_PATH = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"


@pytest.mark.parametrize(
    "url, save_dir",
    [
        (  # non-existent tile
            "GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/"
            "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R18_C27.zip",
            TEST_DATA_PATH,
        ),
        (  # non-zip file
            "copyright.txt",
            TEST_DATA_PATH,
        ),
    ],
)
def test_download_ghsl_invalid_urls(url, save_dir):
    """Test the download_ghsl function with invalid urls."""
    with pytest.raises(ValueError):
        ghsl.download_ghsl(BASE_PATH + url, save_dir=save_dir)


def test_download_ghsl_create_save_dir(_delete_ghsl_tifs):
    """Test the download_ghsl function with non-existent save_dir."""
    url = BASE_PATH + (
        "GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/"
        "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R18_C26.zip"
    )
    ghsl.download_ghsl(url, save_dir=join(TEST_DATA_PATH, "non-existent-dir"))
    # Check files exist in save_dir
    assert isfile(
        join(
            TEST_DATA_PATH,
            "non-existent-dir",
            "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R18_C26.tif",
        )
    )
    # Delete folder
    rmtree(join(TEST_DATA_PATH, "non-existent-dir"))


def test_download_ghsl_connection_error():
    """Test the download_ghsl function for a connection error."""
    with pytest.raises(ValueError):
        ghsl.download_ghsl("https://non-existent-url.com")


def test_download_ghsl_connection_timeout():
    """Test the download_ghsl function for a connection timeout."""
    with pytest.raises(ValueError):
        ghsl.download_ghsl(
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_POP_GLOBE_R2023A/GHS_POP_E2025_GLOBE_R2023A_54009_100/V1-0/tiles/"
            "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R18_C26.zip",
            timeout=0.0001,
        )


@pytest.mark.parametrize(
    "full_raster, bbox_moll, expected",
    [
        (
            join(
                TEST_DATA_PATH,
                "ghsl",
                "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C29.tif",
            ),
            None,
            join(
                TEST_DATA_PATH,
                "ghsl",
                "GHS_POP_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C29.tif",
            ),
        ),
        (
            None,
            [1, 1, 2, 2],
            None,
        ),
        (
            False,
            [1, 1, 2, 2],
            None,
        ),
        (
            None,
            [-1, -1, 1, 1],
            None,
        ),
    ],
)
def test_get_ghsl(full_raster, bbox_moll, expected, monkeypatch, _delete_ghsl_tifs):
    """Test the get_ghsl_urls function.
    Overwrite the config.FULL_RASTER value with FULL_RASTER_val.
    """
    monkeypatch.setattr(ghsl, "FULL_RASTER", full_raster)
    # mock download_ghsl not to download the files
    monkeypatch.setattr(ghsl, "download_ghsl", lambda ghsl_urls: ghsl_urls)
    urls = ghsl.get_ghsl(bbox_moll=bbox_moll)
    if isinstance(expected, str):
        assert urls == expected
    if expected is None:
        assert set(urls) == set(ghsl.get_ghsl_urls(bbox_moll))


@pytest.mark.parametrize(
    "full_raster, bbox_moll",
    [
        (  # faulty full_raster
            "non-existent",
            None,
        ),
        (  # no rasters
            None,
            None,
        ),
        (  # bbox outside World Mollweide
            None,
            [-181e5, -91e5, 181e5, 91e5],
        ),
    ],
)
def test_get_ghsl_invalid(full_raster, bbox_moll, monkeypatch):
    """Test the get_ghsl_urls with invalid inputs.
    Overwrite the config.FULL_RASTER value with FULL_RASTER_val.
    """
    monkeypatch.setattr(ghsl, "FULL_RASTER", full_raster)
    with pytest.raises(ValueError):
        ghsl.get_ghsl(bbox_moll=bbox_moll)
