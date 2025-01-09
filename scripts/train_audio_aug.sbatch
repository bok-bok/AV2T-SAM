#!/bin/bash
#SBATCH --job-name=avs_m3_train

#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

#SBATCH --partition=gpuA40x4
#SBATCH --account=bcza-delta-gpu

#SBATCH -t 48:00:00  

#SBATCH --gpus=1

#SBATCH -e ./slurm_log/slurm-%j.err
#SBATCH -o ./slurm_log/slurm-%j.out

module reset
module load anaconda

conda activate evf_sam

echo "Job started at $(date)"

# Iterate through audio_aug values
for audio_aug in $(seq 0.1 0.1 0.5)
do
    echo "Running with audio_aug=$audio_aug"
    python train_avs.py --lr 0.0002 --num_workers 8 --model_name mul --projector_type mul --normalize --use_adapter --adapter_type mul --batch_size 1 --augmentation --audio_aug $audio_aug --wandb
done

echo "Job finished at $(date)"
