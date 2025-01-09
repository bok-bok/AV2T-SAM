#!/bin/bash
#SBATCH --job-name=avs_m3_test

#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

#SBATCH --partition=gpuA40x4
#SBATCH --account=bcza-delta-gpu

#SBATCH -t 01:00:00

#SBATCH --gpus=1

#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset
module load anaconda

conda activate evf_sam

echo "Job started"

python save_all.py 

