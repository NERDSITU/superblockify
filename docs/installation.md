# Installation

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
Now you are ready to use `superblockify` in your projects,
as shown in the following [Usage](#usage) section.

For development, see the [Development Setup](#development-setup) section.

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
