# Superblockify

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cbueth.github.io/Superblockify/)
[![Coverage badge](https://github.com/cbueth/Superblockify/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/cbueth/Superblockify/tree/python-coverage-comment-action-data)
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
activated using `OSMnxPyrLab`. Alternatively a versionless setup can be done
by executing (`environmentSetupVersionless.sh`)

```bash
conda create -n OSMnxPyrLab -c conda-forge osmnx pyrosm jupyterlab
conda activate OSMnxPyrLab
conda install -c conda-forge jupyterlab_code_formatter blackd isort pylint pytest coverage sphinx sphinx_rtd_theme
pip install scipy==1.10.1
```

_(as scipy==1.10.1 is not yet available on conda-forge, but fixes needed issue
https://github.com/scipy/scipy/pull/17800, remove when availabe on conda-forge, check
https://anaconda.org/conda-forge/scipy)_

which does not have explicit versions, but might resolve dependency issues.

## Logging

The logging is done using the `logging` module. The logging level can be set in the
`setup.cfg` file. The logging level can be set to `DEBUG`, `INFO`, `WARNING`, `ERROR`
or `CRITICAL`. It defaults to `INFO` and a rotating file handler is set up to log
to `results/logs/superblockify.log`. The log file is rotated every megabyte and the last
three log files are kept.