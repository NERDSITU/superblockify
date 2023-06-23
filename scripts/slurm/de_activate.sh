#!/bin/bash
# Script to de-/activate environment

ENV_NAME="sb_env"

# Is any environment already active?
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating environment $CONDA_DEFAULT_ENV"
    # Deactivate environment
    if command -v micromamba &> /dev/null; then
        micromamba deactivate
    elif command -v mamba &> /dev/null
    then
        conda deactivate
    else
        echo "No conda installation found"
        exit 1
    fi
else
    # Activate environment
    iff command -v micromamba &> /dev/null; then
        echo "Activating environment $ENV_NAME with micromamba"
        eval "$(micromamba shell hook -s bash)"
        micromamba activate $ENV_NAME
    elif command -v mamba &> /dev/null
    then
        echo "Activating environment $ENV_NAME with conda"
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
    else
        echo "No conda installation found"
        exit 1
    fi
fi