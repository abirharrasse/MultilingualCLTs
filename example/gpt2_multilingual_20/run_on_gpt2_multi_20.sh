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
export WANDB_API_KEY="097e21df11c8e16d3452a3e5747add10ec3ed5e0"
export HUGGINGFACE_TOKEN="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"

cd /lustre/home/fdraye/projects/featflow
echo "In directory: $(pwd)"

VENV_PATH=/home/fdraye/.cache/pypoetry/virtualenvs/featflow-66hVYwpV-py3.11
$VENV_PATH/bin/python example/gpt2_multilingual_20/run_on_gpt2_multi_20.py $1

echo "run_on_gpt2_multi_20.py completed"
