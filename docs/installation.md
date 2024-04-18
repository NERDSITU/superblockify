# Installation

### Set up environment
Use [`conda`](https://docs.conda.io/projects/conda/en/latest/index.html) or [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) or [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
to create and activate the virtual environment `sb_env` via the `environment.yml` file:

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

For a guided start after installation, see the following [Usage section](#usage).