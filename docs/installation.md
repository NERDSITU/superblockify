# Installation

### Set up environment

Use [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html)
or [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
or [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
to create the virtual environment `sb_env`:

```bash
conda create -n sb_env -c conda-forge superblockify
conda activate sb_env
```

> **Note:** While `pip` can install `superblockify`, it's not officially supported due
> to potential issues with C dependencies needed for OSMnx. If unsure, use `conda` as
> instructed above to avoid problems.

*Alternatively*, or if you run into
issues, [clone this repository](https://github.com/NERDSITU/superblockify/archive/refs/heads/main.zip)
and create the environment via
the [`environment.yml`](https://github.com/NERDSITU/superblockify/blob/main/environment.yml)
file:

```bash
conda env create --file environment.yml
conda activate sb_env
pip install superblockify
```

### Set up Jupyter kernel

If you want to use `superblockify` with its environment `sb_env` in Jupyter, run:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=sb_env
```

This allows you to run Jupyter with the kernel `sb_env` (Kernel > Change Kernel >
sb_env)

For a guided start after installation, see the following [Usage section](#usage).