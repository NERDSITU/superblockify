"""Script to prepare cities.yml for the analysis.

:func:`get_table` crawls the 100 cities from the html table on Springer website.
:func:`update_nominatim_links` updates the Nominatim links for the cities in cities.yml.
It can also be used to get relation IDs for easy validation of the Nominatim links.
"""

# Self contained script to crawl 100 cities from html table on Springer website.
# Paper: Boeing, G. Urban spatial order: street network orientation, configuration,
# and entropy. Appl Netw Sci 4, 67 (2019). https://doi.org/10.1007/s41109-019-0189-1
import sys

import bs4
import pandas as pd
import requests
from ruamel.yaml import YAML
from tqdm import tqdm

table_url = (
    "https://appliednetsci.springeropen.com/articles/10.1007/s41109-019-0189-1/tables/1"
)


def get_table():
    """Get table from Springer website."""
    # Load html table from url
    table_html = requests.get(table_url, timeout=30).text

    # Get html table from url
    table = bs4.BeautifulSoup(table_html, features="html.parser").find("table")

    # To pandas dataframe
    df = pd.read_html(str(table))[0]

    # Show first 10 rows
    print(f"Found table with {len(df)} rows. Showing first 10 rows:")
    print(df.head(10))

    # Ask if user wants to add it to `cities.yml` - new list under `places`
    print("Do you want to add these cities to `cities.yml`? (y/n)")
    answer = input()
    if answer.lower() != "y":
        print("Aborting.")
        sys.exit(0)

    # Add to cities.yml
    # load cities.yml
    with open("cities.yml", encoding="utf-8") as file:
        yaml = YAML()
        cities = yaml.load(file)

    new_list_name = None
    while new_list_name is None:
        print("How should the new list be called? (e.g. `places_100_cities`)")
        new_list_name = input()
        # check if that name is already in cities['places']
        if new_list_name in cities["place_lists"]:
            print("This name already exists. Do you want to overwrite it? (y/n)")
            if input().lower() != "y":
                new_list_name = None
    print(
        "How should the new list be described? (e.g. `100 cities from Boeing (2019)`)"
    )
    new_list_description = input()

    def region_handle(region):
        if region == "Asia/Oceania":
            return "A&O"
        if region == "Europe":
            return "EU"
        if region == "Latin America":
            return "LatAm"
        if region == "Middle East/Africa":
            return "MEA"
        if region == "US/Canada":
            return "NAM"
        return region

    # add new list
    cities["place_lists"][new_list_name] = {
        "description": new_list_description,
        "cities": {},
    }
    # add cities to new list
    for _, row in df.iterrows():
        city = {
            "query": row["City"],
            "country": None,
            "region": region_handle(row["Region"]),
            "orient_order": row.get("φ", None),
            "circuity_avg": row.get("ς", None),
            "median_segment_length": row.get("ĩ", None),
            "k_avg": row.get("k̅", None),
            "nominatim link": f"https://nominatim.openstreetmap.org/ui/search.html?q="
            f"{row['City']}",
        }
        # Add further columns if they exist
        for column in df.columns:
            if column not in ["City", "Region", "φ", "ς", "ĩ", "σ", "k̅"]:
                city[column] = row[column]
        cities["place_lists"][new_list_name]["cities"][row["City"]] = city

    # save cities.yml
    with open("cities.yml", "w", encoding="utf-8") as file:
        yaml.dump(cities, file)

    # Remind to fill the country codes and check OSM queries
    print(
        f"Added {len(df)} cities to `cities.yml` under `{new_list_name}`. "
        "Please fill in the country codes and check the OSM queries."
    )


def update_nominatim_links():
    """Update Nominatim links for cities in cities.yml.

    Also add link to first found OSM relation.
    This can be used to validate the Nominatim links.
    """
    # load cities.yml
    with open("cities.yml", encoding="utf-8") as file:
        yaml = YAML()
        cities = yaml.load(file)

    print("Also update the OSM relation links? (y/n)")
    update_osm = input().lower() == "y"

    # update nominatim links and add a link to first found OSM relation
    for place_list in cities["place_lists"]:
        for data in tqdm(
            cities["place_lists"][place_list]["cities"].values(),
            desc=f"Updating {place_list}",
            unit="city",
        ):
            # Nominatim link
            data["nominatim link"] = [
                f"https://nominatim.openstreetmap.org/ui/search.html?q={query}".replace(
                    " ", "+"
                )
                for query in (
                    data["query"]
                    if isinstance(data["query"], list)
                    else [data["query"]]
                )
            ]
            # OSM relation link
            if not update_osm:
                continue
            url = "https://nominatim.openstreetmap.org/search"
            data["osm_id"] = []
            data["OSM relation"] = []
            for query in (
                data["query"] if isinstance(data["query"], list) else [data["query"]]
            ):
                response = requests.get(
                    url, params={"q": query, "format": "json"}, timeout=10
                ).json()
                if len(response) > 0:
                    # get first osm_id that is a relation
                    osm_id = None
                    for result in response:
                        if result["osm_type"] == "relation":
                            osm_id = result["osm_id"]
                            break
                    if osm_id is None:
                        print(f"Could not find OSM relation for {query}.")
                        continue
                    data["osm_id"].append(osm_id)
                    data["OSM relation"].append(
                        f"https://www.openstreetmap.org/relation/{osm_id}"
                    )

    # save cities.yml
    with open("cities.yml", "w", encoding="utf-8") as file:
        yaml.dump(cities, file)

    print("Updated Nominatim links in `cities.yml`.")


if __name__ == "__main__":
    # Options
    print(
        "Do you want to (1) crawl the table from the website or "
        "(2) update the Nominatim links? (1/2)"
    )
    answer = input()
    if answer == "1":
        get_table()
    elif answer == "2":
        update_nominatim_links()
    else:
        print("Invalid option. Aborting.")
        sys.exit(0)
