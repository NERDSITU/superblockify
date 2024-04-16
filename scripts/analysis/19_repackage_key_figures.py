"""Script to repackage the key figures.

Repackages several .yml files into few .feather file to reduce loading time.
Per configuration creating one file.
The file name of the key figures starts with the "city-name_", followed by the
configuration name, and ends with "_key_figure.yml".

For each configuration we will combine all key figures into one dataframe and
save it as a .feather file. Is then called how the configuration is called
plus "_key_figures.feather".
"""

from glob import glob
from os.path import join

import pandas as pd
from joblib import Parallel, delayed
import yaml
from tqdm import tqdm

KEY_FIGURES_DIR = join("data", "results", "key_figures")


def clean_dicts(d):
    """Converts all keys of a dict to strings recursively."""
    if isinstance(d, list):
        return [clean_dicts(v) for v in d]
    if not isinstance(d, dict):
        return d
    return {
        str(k): clean_dicts(v) if isinstance(v, dict) else v
        for k, v in d.items()
        if k != "representative_node_id"
    }


if __name__ == "__main__":
    # Find all file names
    files = glob(join(KEY_FIGURES_DIR, "*_key_figures.yml"))
    print(f"Loading key figures from {KEY_FIGURES_DIR}, there are {len(files)} files.")
    # find common configuration names:
    # results/key_figures/city-name_CON_FIG_URATION_key_figure.yml
    configs = set(
        # cut everything after last / and before first _ but as string
        "_".join(filename.split("/")[-1].split("_")[1:])
        for filename in files
    )
    print(f"Found {len(configs)} configurations.")

    all_loaded = True

    # Loop over all configurations
    for config in configs:
        print(f"Repackaging key figures for {config}...")
        files_config = [filename for filename in files if config in filename]
        print(f" - Found {len(files_config)} files for {config}")

        def load_or_catch(filename):
            try:
                return yaml.safe_load(open(filename, "r", encoding="utf-8"))
            except:
                print(f"Error loading {filename}")
                return filename

        key_figures = Parallel(n_jobs=-1, verbose=1)(
            delayed(load_or_catch)(filename) for filename in tqdm(files_config)
        )
        if not all(isinstance(kf, dict) for kf in key_figures):
            all_loaded = False
            print("Some files could not be packaged:")
            print([kf for kf in key_figures if isinstance(kf, str)])
        # Remove all files that could not be loaded
        key_figures = [kf for kf in key_figures if not isinstance(kf, str)]

        # Convert to dataframe
        print(" - Converting to dataframe")
        key_figures = pd.DataFrame(key_figures)
        # Apply on every value the function stringify_keys
        key_figures = key_figures.applymap(clean_dicts)

        # Save to disk
        print(" - Saving to disk")
        key_figures.to_feather(
            join(KEY_FIGURES_DIR, config.replace(".yml", ".feather"))
        )

    print("Done!")
    if not all_loaded:
        print("Some files could not be loaded, see above.")
