"""Tests for the population module."""
import pytest

from superblockify.metrics.population import row_col, get_ghsl_urls


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
    assert row_col(y_moll, x_moll) == (row, col)


@pytest.mark.parametrize(
    "bbox_moll, res, urls",
    [
        (
            [1, 1, 2, 2],
            10,
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_10/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_10_V1_0_R9_C19.zip",
            ],
        ),
        (
            [-1, -1, 1, 1],
            10,
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_10/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_10_V1_0_R9_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_10/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_10_V1_0_R10_C19.zip",
            ],
        ),
        (
            [-1, -1, 1, 1],
            100,
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C19.zip",
            ],
        ),
        (
            [-42000, -1, 0, 1],
            100,
            [
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C19.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R9_C18.zip",
                "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
                "GHSL/GHS_BUILT_S_GLOBE_R2023A"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100/V1-0/tiles"
                "/GHS_BUILT_S_E2025_GLOBE_R2023A_54009_100_V1_0_R10_C18.zip",
            ],
        ),
    ],
)
def test_get_ghsl_urls(bbox_moll, res, urls):
    """Test the get_GHSL_urls function."""
    assert set(get_ghsl_urls(bbox_moll, res)) == set(urls)
