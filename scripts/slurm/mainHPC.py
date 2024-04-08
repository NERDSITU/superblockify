"""Main script for development."""

import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

from superblockify import ResidentialPartitioner, save_to_gpkg
from superblockify.config import Config

if __name__ == "__main__":
    CITY_NAME, SEARCH_STR = Config.PLACES_GENERAL[1]
    # CITY_NAME, SEARCH_STR = Config.PLACES_SMALL[0]

    part = ResidentialPartitioner(
        name=CITY_NAME + "_HPC", city_name=CITY_NAME, search_str=SEARCH_STR
    )
    part.run(make_plots=True, num_workers=10, chunk_size=1)
    part.save(save_graph_copy=True)
    save_to_gpkg(part)
