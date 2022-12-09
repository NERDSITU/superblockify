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
conda install -c conda-forge jupyterlab_code_formatter black isort pylint pytest coverage sphinx sphinx_rtd_theme
```

which does not have explicit versions, but might resolve dependency issues.