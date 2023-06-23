# Superblockify

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cbueth.github.io/Superblockify/)
[![codecov](https://codecov.io/gh/cbueth/Superblockify/branch/main/graph/badge.svg?token=AS72IFT2Q4)](https://codecov.io/gh/cbueth/Superblockify)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Source code for blockifying existing street networks.

---

## Set up

The environment to run the projects' code can be set up using the
`environment.yaml` by running:

```bash
conda env create --file=environment.yml
```

This initializes a conda environment by the name `OSMnxPyrLab`, which can be
activated using `OSMnxPyrLab`.
If you want to use `mamba` or `micromamba` for faster package resolution, just replace 
`conda` with the respective. For `micromamba`:
    
```bash
micromamba env create --file=environment.yml
```

Alternatively a version-less setup can be done by executing 
(`environmentSetupVersionless.sh` in the working directory)

```bash
conda create -n OSMnxPyrLab -c conda-forge python=3.10 --file requirements.txt
conda activate OSMnxPyrLab
conda env export | grep -v "^prefix: " > environment.yml
```

which does not have explicit versions, but might resolve dependency issues. Using
`git diff environment.yml` the changes can be inspected.
With `mamba` this can be done by running

```bash
mamba create -n OSMnxPyrLab -c conda-forge python=3.10 --file requirements.txt
mamba activate OSMnxPyrLab
mamba env export | grep -v "^prefix: " > environment.yml
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
