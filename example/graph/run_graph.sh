#!/bin/bash
set -x  # Print each command before execution for debugging

# Load .bashrc to get PATH if necessary
source ~/.bashrc

# Export POETRY_HOME if not set
export POETRY_HOME=$HOME/.local/bin
export HOME=/lustre/home/aharrasse
export PATH="$HOME/.local/bin:$PATH"

echo "Home set to:" $HOME

# Set WANDB_API_KEY to allow wandb to authenticate
export WANDB_API_KEY=""
export HUGGINGFACE_TOKEN=""

cd /home/aharrasse/FeatFlow
echo "In directory: $(pwd)"

VENV_PATH=/home/aharrasse/.cache/pypoetry/virtualenvs/featflow-Dhm6kclg-py3.10
$VENV_PATH/bin/python example/graph/run_graph.py

echo "run_graph.py completed"
