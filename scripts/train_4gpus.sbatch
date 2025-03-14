#!/bin/bash
#SBATCH --job-name=avs_train_4gpus

#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

#SBATCH --partition=gpuA40x4
#SBATCH --account=bcza-delta-gpu

#SBATCH -t 48:00:00  

#SBATCH --gpus=4

#SBATCH -e ./slurm_log/slurm-%j.err
#SBATCH -o ./slurm_log/slurm-%j.out

module reset
module load anaconda

conda activate evf_sam

echo "Job started at $(date)"


# Run the training script
# evf_sam2 
# S4 
# evf_sam2  no adapter 
# python train_avs.py --lr 0.0002  --projector_type mul --evf_version evf_sam2 --dataset s4 --batch_size 16 --wandb

# evf_sam2
# S4 mul adapter 
python train_avs.py --model_name mul2  --evf_version evf_sam2 --projector_type mul --use_adapter --dataset s4 --batch_size 16 --num_workers 4 --wandb

# evf_sam2 S4 mul no adapter 
# python train_avs.py --evf_version evf_sam2 --projector_type mul --dataset s4 --batch_size 16 --num_workers 4 --wandb

# evf_sam2 
# S4 clap adapter 
# python train_avs.py --lr 0.0002   --evf_version evf_sam2 --projector_type clap --use_adapter --dataset s4 --batch_size 16 --num_workers 4 --wandb

# evf_sam2 
# S4 clip adapter 
# python train_avs.py --lr 0.0002   --evf_version evf_sam2 --projector_type clip --use_adapter --dataset s4 --batch_size 16 --num_workers 4 --wandb 

# S4 evf_sam
# python train_avs.py --lr 0.0002  --projector_type mul --evf_version evf_sam --dataset s4 --use_adapter --adapter_type mul --batch_size 6 --wandb


# M4 evf_sam adapter
# python train_avs.py --projector_type mul --evf_version evf_sam --use_adapter --batch_size 1 --wandb




echo "Job finished at $(date)"
