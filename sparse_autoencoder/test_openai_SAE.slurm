#!/bin/bash
#SBATCH --job-name=download_openai_SAE
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=test_hlm_sae_%j.out
#SBATCH --error=test_hlm_sae_%j.err
#SBATCH --time=4:00:00

python test_openai_SAE.py
