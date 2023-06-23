conda create -n sb_env -c conda-forge python=3.10 --file requirements.txt
conda env export > environment.yml
conda activate sb_env