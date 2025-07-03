#!/bin/bash
#SBATCH --job-name=train_openai_SAE
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=train_openai_SAE_%j.out
#SBATCH --error=train_openai_SAE%j.err
#SBATCH --time=36:00:00

python my_train.py
