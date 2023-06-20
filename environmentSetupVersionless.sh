conda create -n OSMnxPyrLab -c conda-forge python=3.10 --file requirements.txt
conda env export > environment.yml
conda activate OSMnxPyrLab