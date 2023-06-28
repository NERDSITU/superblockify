"""Check which cities are done."""
from itertools import product
from os.path import join, dirname, exists
from sys import path

from pandas import DataFrame, set_option

set_option("display.max_rows", None)

path.append(join(dirname(__file__), "..", ".."))

from scripts.analysis.config import combinations, short_name_combination

from superblockify.config import (
    RESULTS_DIR,
    PLACES_100_CITIES,
    PLACES_GERMANY,
)

if __name__ == "__main__":
    # For every city in the list of cities check how many folders with the file done
    # exist
    PLACES_100_CITIES = list(PLACES_100_CITIES.keys()) + list(PLACES_GERMANY.keys())
    combinations = list(combinations)
    # Pandas dataframe with all cities and combinations
    availability = DataFrame(
        columns=["city", "combination", "done"],
        index=range(len(PLACES_100_CITIES) * len(combinations)),
        data=list(product(PLACES_100_CITIES, combinations, [False])),
    )
    # for every column write to `done` if there is a file named `done` in the folder
    # `load_err` if there is a file named `load_err` in the folder
    # and `started` if there is a folder but no file named `done` or `load_err`
    for i, row in availability.iterrows():
        city, combination, _ = row
        availability.loc[i, "combination"] = short_name_combination(combination)
        folder = join(RESULTS_DIR, city + "_" + availability.loc[i, "combination"])
        if exists(join(folder, "done")):
            availability.loc[i, "done"] = True
        elif exists(join(folder, "load_err")):
            availability.loc[i, "done"] = "load_err"
        elif exists(folder):
            availability.loc[i, "done"] = "started"
    print(availability)
    # General percentages
    print(availability["done"].value_counts(normalize=True))
    # Show percentage per combination that has `True` in the `done` column
    # handle load_err and started as 0 - add new column
    print(availability.groupby(["combination", "done"]).count().groupby(level=0))
    # Show percentage of `True` in done column per city - all clu
    print(
        availability.groupby(["city", "done"])
        .count()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
    )
    # Show percentage per city and partitioner (that is the first part of the
    # combination string before the first underscore)
    availability["partitioner"] = availability["combination"].apply(
        lambda x: x.split("_")[0]
    )
    print(
        availability.groupby(["city", "partitioner", "done"])
        .count()
        .groupby(level=[0, 1])
        .apply(lambda x: 100 * x / x.sum())
    )
