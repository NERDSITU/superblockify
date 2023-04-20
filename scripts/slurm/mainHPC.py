"""Main script for development."""
# Print current working directory
import os
from ast import literal_eval
from configparser import ConfigParser
from os.path import dirname, join

print(f"Current working directory: {os.getcwd()}")

from superblockify import ResidentialPartitioner, save_to_gpkg

config = ConfigParser()
config.read(join(dirname(__file__), "..", "..", "config.ini"))
PLACES_GENERAL = literal_eval(config["tests"]["places_general"])
PLACES_SMALL = literal_eval(config["tests"]["places_small"])

if __name__ == "__main__":
    CITY_NAME, SEARCH_STR = PLACES_GENERAL[1]
    # CITY_NAME, SEARCH_STR = PLACES_SMALL[0]

    part = ResidentialPartitioner(
        name=CITY_NAME + "_HPC", city_name=CITY_NAME, search_str=SEARCH_STR
    )
    part.run(make_plots=True, num_workers=10, chunk_size=1)
    part.save(save_graph_copy=True)
    save_to_gpkg(part)
