#!/bin/bash
# filepath: /home/fdraye/projects/featflow/example/autointerp/tiny_stories/run_autointerp.sh
set -x  # Print each command before execution for debugging

# Load .bashrc to get PATH if necessary
source ~/.bashrc

# Export POETRY_HOME if not set
export POETRY_HOME=$HOME/.local/bin
export HOME=/lustre/home/fdraye
export PATH="$HOME/.local/bin:$PATH"
export VLLM_HOST_IP=127.0.0.1
export CUDA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "Home set to:" $HOME

# Set WANDB_API_KEY to allow wandb to authenticate
export WANDB_API_KEY=""
export HUGGINGFACE_TOKEN=""

cd /lustre/home/fdraye/projects/featflow
echo "In directory: $(pwd)"

# Get the combination for this job
JOB_INDEX=$1
PARAMS=$(python3 example/autointerp/tiny_stories/get_combination.py $JOB_INDEX)
echo "Job parameters: $PARAMS"

VENV_PATH=/home/fdraye/.cache/pypoetry/virtualenvs/featflow-66hVYwpV-py3.11
$VENV_PATH/bin/python example/autointerp/tiny_stories/run_autointerp.py $PARAMS

echo "run_autointerp.py completed"
