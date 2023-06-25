"""Example of saving a partitioner to a geopackage."""

import superblockify as sb
from superblockify.config import logger, PLACES_GENERAL, PLACES_SMALL, PLACES_100_CITIES

if __name__ == "__main__":
    # CITY_NAME, SEARCH_STR = PLACES_GENERAL[2]
    CITY_NAME, SEARCH_STR = PLACES_SMALL[1]
    # CITY_NAME, SEARCH_STR = PLACES_100_CITIES[1]

    logger.info(
        "Running partitioner for %s with search string %s.", CITY_NAME, SEARCH_STR
    )

    part = sb.ResidentialPartitioner(
        name=CITY_NAME + "_main", city_name=CITY_NAME, search_str=SEARCH_STR
    )

    part.run(calculate_metrics=False, make_plots=True)
    # part.save()
    sb.save_to_gpkg(part, save_path=None)  # None means save to default location

    logger.info("Saved %s to geopackage.", part.name)
