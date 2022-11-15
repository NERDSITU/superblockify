# Superblockify

Source code for blockifying existing street networks.

---

## Set up

The environment to run the projects' code can be set up using the
`environment.yaml` by running:

```bash
conda env create --file=environment.yaml
```

This initializes a conda environment by the name `OSMnxPyrLab`, which can be
activated using `OSMnxPyrLab`. Alternatively a versionless setup can be done
by executing (`environmentSetupVersionless.sh`)

```bash
conda config --prepend channels conda-forge
conda create -n OSMnxPyrLab --strict-channel-priority osmnx pyrosm jupyterlab
conda install -c conda-forge jupyterlab_code_formatter black isort
```

which does not have explicit versions, but might resolve dependency issues.