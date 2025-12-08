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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# export HF_HUB_OFFLINE=1

echo "Home set to:" $HOME

# Add these new exports to help with CUDA issues
export CUDA_LAUNCH_BLOCKING=1  # Makes CUDA errors more visible
export RAY_DEDUP_LOGS=0  # Better error visibility for vLLM
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Try different attention backend

# Set WANDB_API_KEY to allow wandb to authenticate
export WANDB_API_KEY=""
export HUGGINGFACE_TOKEN=""

cd /lustre/home/fdraye/projects/featflow
echo "In directory: $(pwd)"

# Get the combination for this job
JOB_INDEX=$1
PARAMS=$(python3 example/autointerp/gpt2_multilingual/get_combination_multi.py $JOB_INDEX)
echo "Job parameters: $PARAMS"

VENV_PATH=/home/fdraye/.cache/pypoetry/virtualenvs/featflow-66hVYwpV-py3.11
$VENV_PATH/bin/python example/autointerp/gpt2_multilingual/run_autointerp_multi.py $PARAMS

echo "run_autointerp_multi.py completed"
