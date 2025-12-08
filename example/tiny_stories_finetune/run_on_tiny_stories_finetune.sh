#!/bin/bash
set -x  # Print each command before execution for debugging

# Load .bashrc to get PATH if necessary
source ~/.bashrc

# Export POETRY_HOME if not set
export POETRY_HOME=$HOME/.local/bin
export HOME=/lustre/home/fdraye
export PATH="$HOME/.local/bin:$PATH"

echo "Home set to:" $HOME

# Set WANDB_API_KEY to allow wandb to authenticate
export WANDB_API_KEY=""
export HUGGINGFACE_TOKEN=""

cd /lustre/home/fdraye/projects/featflow
echo "In directory: $(pwd)"

VENV_PATH=/home/fdraye/.cache/pypoetry/virtualenvs/featflow-66hVYwpV-py3.11
$VENV_PATH/bin/python example/tiny_stories_finetune/run_on_tiny_stories_finetune.py

echo "run_on_tiny_stories_finetune.py completed"
