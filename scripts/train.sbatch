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


# Run the training script

# evf_sam1 no adapter
# M4
python train_avs.py --projector_type mul --evf_version evf_sam --batch_size 1 --wandb

# python train_avs.py --model_name 1_interval --use_adapter --adapter_type mul  --projector_type mul --dataset ms3 --batch_size 1 --wandb

# python train_avs.py --projector_type clap --batch_size 3 --wandb 
# python train_avs.py --projector_type clip --batch_size 3 --wandb 
# python train_avs.py --model_name clap2 --projector_type clap --batch_size 3 --wandb 

# with adapter 
# python train_avs.py --projector_type clip --batch_size 1 --wandb --use_adapter 
# python train_avs.py --projector_type clap --batch_size 1 --wandb --use_adapter 
# python train_avs.py --projector_type mul --batch_size 1 --wandb --use_adapter 



echo "Job finished at $(date)"
