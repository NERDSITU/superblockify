![superblockify_logo.png](superblockify_logo.png "superblockify")

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://NERDSITU.github.io/superblockify/)
[![PyPI Version](https://badge.fury.io/py/superblockify.svg)](https://pypi.org/project/superblockify/)
[![Python Version](https://img.shields.io/pypi/pyversions/superblockify)](https://pypi.org/project/superblockify/)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI License](https://img.shields.io/pypi/l/superblockify)](https://pypi.org/project/superblockify/)

[![Docs](https://github.com/NERDSITU/superblockify/actions/workflows/docs.yml/badge.svg)](https://github.com/NERDSITU/superblockify/actions/workflows/docs.yml)
[![Lint](https://github.com/NERDSITU/superblockify/actions/workflows/lint.yml/badge.svg)](https://github.com/NERDSITU/superblockify/actions/workflows/lint.yml)
[![Test](https://github.com/NERDSITU/superblockify/actions/workflows/test.yml/badge.svg)](https://github.com/NERDSITU/superblockify/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/NERDSITU/superblockify/branch/main/graph/badge.svg?token=AS72IFT2Q4)](https://codecov.io/gh/NERDSITU/superblockify)

Source code to `superblockify` an urban street network

---

## Installation

We recommend
using [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
to create a virtual environment and installing the package in editable mode.
Alternatively, one can
use [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html)
or [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
to create the environment (they can be used interchangeably).

```bash
micromamba create -n sb_env -c conda-forge python=3.12 osmnx
micromamba activate sb_env
pip install superblockify
```

This installs the package and its dependencies,
ready for use when activating the environment.
Learn more about `superblockify` by reading
the [documentation](https://NERDSITU.github.io/superblockify/)
with the [usage section](https://nerdsitu.github.io/superblockify/usage/)
or
the [minimal working example](https://github.com/NERDSITU/superblockify/blob/main/scripts/examples/mwe.py).

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

## Usage

For a quick start there are example scripts in
the [`examples/`](https://github.com/NERDSITU/superblockify/blob/main/scripts/examples/)
folder and
a [minimal working example](https://github.com/NERDSITU/superblockify/blob/main/scripts/examples/mwe.py).

## Testing

The tests are specified using the `pytest` signature, see [`tests/`](tests/) folder, and
can be run using a test runner of choice.
A pipeline is set up, see [`.github/workflows/test.yml`](.github/workflows/test.yml).
