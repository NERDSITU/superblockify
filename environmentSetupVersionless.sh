conda config --prepend channels conda-forge
conda create -n OSMnxPyrLab --strict-channel-priority osmnx pyrosm jupyterlab
conda activate OSMnxPyrLab
conda install -c conda-forge jupyterlab_code_formatter black isort pylint pytest