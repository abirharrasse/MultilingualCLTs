#!/bin/bash
#SBATCH --job-name=intervention
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --account=def-rgrosse
#SBATCH --output=/home/abir19/FeatFlow/src/featflow/frontend/run_cluster/condor_intervention.out
#SBATCH --error=/home/abir19/FeatFlow/src/featflow/frontend/run_cluster/condor_intervention.err

set -x

source ~/.bashrc

export POETRY_HOME=$HOME/.local/bin
export HOME=/home/abir19
export PATH="$HOME/.local/bin:$PATH"
export WANDB_API_KEY="097e21df11c8e16d3452a3e5747add10ec3ed5e0"
export HUGGINGFACE_TOKEN="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"

echo "Home set to:" $HOME

cd /home/abir19/FeatFlow
echo "In directory: $(pwd)"

VENV_PATH=/home/abir19/.cache/pypoetry/virtualenvs/featflow-66hVYwpV-py3.11

$VENV_PATH/bin/python src/featflow/frontend/run_cluster/run_cluster_intervention.py "$@"

echo "run_cluster_intervention.py completed"