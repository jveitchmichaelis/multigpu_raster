#!/bin/bash
#SBATCH --job-name=MultiGPURasterInference
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=00:10:00       #Time limit hrs:min:sec
#SBATCH --output=./slurm_logs/%j.out
#SBATCH --error=./slurm_logs/%j.err

# Print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

module purge

# Pre-install your environment and or make sure that you're running within
# an appropriate conda environment with the project dependencies installed.
source .venv/bin/activate

srun mgpr gpus=$SLURM_NTASKS_PER_NODE

# Run example: export NUM_GPUS=2; sbatch --ntasks-per-node=$NUM_GPUS --gpus=$NUM_GPUS job.slurm