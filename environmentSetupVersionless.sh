conda create -n OSMnxPyrLab -c conda-forge osmnx pyrosm jupyterlab
conda activate OSMnxPyrLab
conda install -c conda-forge jupyterlab_code_formatter black isort pylint pytest coverage sphinx sphinx_rtd_theme