#!/bin/bash
#SBATCH --job-name=sae_lens_SAE
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=test_sae_lens_%j.out
#SBATCH --error=test_sae_lens_%j.err
#SBATCH --time=4:00:00

python test_sae_lens_SAE.py
