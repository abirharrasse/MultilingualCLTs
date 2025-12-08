#!/bin/bash
set -x  # Print each command before execution for debugging

# Load .bashrc to get PATH if necessary
source ~/.bashrc

# Export POETRY_HOME if not set
export POETRY_HOME=$HOME/.local/bin
export HOME=/home/abir19
export PATH="$HOME/.local/bin:$PATH"

echo "Home set to:" $HOME

# Set WANDB_API_KEY to allow wandb to authenticate
export WANDB_API_KEY="097e21df11c8e16d3452a3e5747add10ec3ed5e0"
export HUGGINGFACE_TOKEN="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"

cd /home/abir19/FeatFlow
echo "In directory: $(pwd)"

VENV_PATH=/home/abir19/.cache/pypoetry/virtualenvs/featflow-66hVYwpV-py3.11

# Run the intervention script with passed arguments
$VENV_PATH/bin/python src/featflow/frontend/run_cluster/run_cluster_intervention.py "$@"

echo "run_cluster_intervention.py completed"
