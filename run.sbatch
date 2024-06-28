#!/bin/bash
#SBATCH --output=./logs/close_drawer.txt

#SBATCH --job-name=close_drawer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --time=7:00:00

#SBATCH --mail-type=END
#SBATCH --mail-user=vb2386@nyu.edu


# Load necessary modules
module purge
module load anaconda3/2024.02

source ~/.bashrc  # Make sure this points to the correct shell initialization file

source activate /scratch/gb2643/documents/ACT/act


# Run the training script
python train.py
