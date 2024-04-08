# `superblockify`

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NERDSITU.github.io/superblockify/)
[![codecov](https://codecov.io/gh/NERDSITU/superblockify/branch/main/graph/badge.svg?token=AS72IFT2Q4)](https://codecov.io/gh/NERDSITU/superblockify)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Docs](https://github.com/NERDSITU/superblockify/actions/workflows/docs.yml/badge.svg)](https://github.com/NERDSITU/superblockify/actions/workflows/docs.yml)
[![Lint](https://github.com/NERDSITU/superblockify/actions/workflows/lint.yml/badge.svg)](https://github.com/NERDSITU/superblockify/actions/workflows/lint.yml)
[![Test](https://github.com/NERDSITU/superblockify/actions/workflows/test.yml/badge.svg)](https://github.com/NERDSITU/superblockify/actions/workflows/test.yml)

Source code for blockifying existing street networks.

---

## Installation

Create a new environment with `conda` or `mamba` and install the package from the
`conda-forge` channel.

```bash
conda create -n sb_env -c conda-forge superblockify
conda activate sb_env
```

This installs the package and its dependencies,
ready for use when activating the environment.
Learn more about `superblockify` by reading
the [documentation](https://NERDSITU.github.io/superblockify/)
or the [minimal working example](scripts/mwe.py).

## Development Setup

For development, we recommend using `micromamba` to create a virtual
environment and installing the package in editable mode.
Alternatively, one can use `conda` or `mamba` to create the environment
(they can be used interchangeably).
After cloning the repository, navigate to the root folder and
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

## Usage

For a quick start there are example scripts in the [`examples/`](scripts/examples/)
folder and a [minimal working example](scripts/mwe.py).

## Logging

The logging is done using the `logging` module. The logging level can be set in the
`setup.cfg` file. The logging level can be set to `DEBUG`, `INFO`, `WARNING`, `ERROR`
or `CRITICAL`. It defaults to `INFO` and a rotating file handler is set up to log
to `results/logs/superblockify.log`. The log file is rotated every megabyte, and the
last three log files are kept.

## Testing

The tests are specified using the `pytest` signature, see [`tests/`](tests/) folder, and
can be run using a test runner of choice.
A pipeline is set up, see [`.github/workflows/test.yml`](.github/workflows/test.yml).
