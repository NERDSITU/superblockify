# Getting Started

## Installation

We recommend using `micromamba` to create a virtual
environment and installing the package in editable mode.
Alternatively, one can use `conda` or `mamba` to create the environment
(they can be used interchangeably).
After cloning the repository, navigate to the root folder and
create the environment with the wished python version and the development dependencies.

```bash
micromamba create -n sb_env -c conda-forge python=3.12 osmnx
micromamba activate sb_env
pip install superblockify
```

This installs the package and its dependencies,
ready for use when activating the environment.

For development, see the [Development Setup](#development-setup) section.

## Usage

The `superblockify` package works out of the box, meaning no further downloads are
necessary. Maps are downloaded from the OpenStreetMap API and population data is
downloaded from the [GHSL-POP 2023](https://ghsl.jrc.ec.europa.eu/ghs_pop2023.php)
dataset. Only tiles needed are being cached in the `data/ghsl` folder.

As example, we will try to partition the French city of Lyon with the
`ResidentialPartitioner` class. Afterward we will save the results to a geopackage
file that can easily be opened and edited in QGIS.

```python
import superblockify as sb

part = sb.ResidentialPartitioner(
    name="Lyon_test", city_name="Lyon", search_str="Lyon, France"
)
part.run(calculate_metrics=False, make_plots=True)
sb.save_to_gpkg(part, save_path=None)
```

This will download the map of Lyon, store it in the `data/graphs` folder for
use later, partition the city and save the results to a geopackage file in the
`data/results/Lyon_test` folder. In the same folder, a `Lyon_test.gpkg` file
will be created that can be opened in QGIS.
If you want to also calculate the metrics, you can set `calculate_metrics=True`.

To learn more about the inner workings and background of the package, please
see the next Reference section. Otherwise, you can also check out the
[API documentation](api/index)

## Development Setup

For development, clone the repository, navigate to the root folder and
create the environment with the wished python version and the development dependencies.

```bash
micromamba create -n sb_env -c conda-forge python=3.12 --file=environment.yml
micromamba activate sb_env
```

Now it is possible to import the package relatively to the root folder.
Optionally, register the package in editable mode with `pip`:

```bash
pip install --no-build-isolation --no-deps -e .
```

## Testing

The tests are specified using the `pytest` signature in the `tests/` folder and
can be run using a test runner of choice.
