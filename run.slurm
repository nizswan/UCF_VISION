#!/bin/bash
#SBATCH --job-name=job
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --partition=highgpu
#SBATCH --qos=adutta
#SBATCH --account=adutta
#SBATCH --time=1:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

module load anaconda
eval "$(conda shell.bash hook)"
conda activate nlp310
hash -r

export WANDB_API_KEY="3699d2250600840d4e2971f3b4109a9c37a69b16"
export WANDB_PROJECT="UCF_COMPUTER_VISION"

cd /home/cr013573/UCF_CV/

# Forward ALL extra args from sbatch directly into run.py
python run.py "$@"

echo "Job finished with exit code $?"
