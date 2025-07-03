#!/bin/bash
#SBATCH --job-name=print_transformer_hooks
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=print_transformer_hooks_%j.out
#SBATCH --error=print_transformer_hooks_%j.err
#SBATCH --time=4:00:00

python test_transformer_lens_hooks.py
