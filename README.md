![superblockify](superblockify_logo.png "superblockify")

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

`superblockify` is a Python package for partitioning an urban street network into
Superblock-like neighborhoods and for visualizing and analyzing the partition results. A
Superblock is a set of adjacent urban blocks where vehicular through traffic is
prevented or pacified, giving priority to people walking and cycling.

## Installation

### Set up environment
Use [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html) or [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) or [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
to create and activate the virtual environment `sb_env` via the [`environment.yml`](environment.yml) file:

```bash
conda env create --file environment.yml
conda activate sb_env
```

*Alternatively*, or if you run into issues, run:

```bash
conda create -n sb_env -c conda-forge python=3.12 osmnx=1.9.2
conda activate sb_env
```

### Install package
Next, install the package:

```bash
pip install superblockify
```

### Set up Jupyter kernel
If you want to use `superblockify` with its environment `sb_env` in Jupyter, run:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=sb_env
```

This allows you to run Jupyter with the kernel `sb_env` (Kernel > Change Kernel > sb_env)


## Usage

We provide a minimum working example in two formats: 

* [Jupyter notebook (`mwe.ipynb`)](mwe.ipynb)
* [Python script (`mwe.py`)](mwe.py)

There are additional example scripts in
the [`examples/`](scripts/examples/)
folder.

For a guided start after installation, see the [usage section](https://superblockify.city/usage/) in the documentation.

## Documentation

Read the [documentation](https://superblockify.city) to learn more about `superblockify`.


## Testing

The tests are specified using the `pytest` signature, see [`tests/`](tests/) folder, and
can be run using a test runner of choice.
A pipeline is set up, see [`.github/workflows/test.yml`](.github/workflows/test.yml).
