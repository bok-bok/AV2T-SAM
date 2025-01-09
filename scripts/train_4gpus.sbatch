#!/bin/bash
#SBATCH --job-name=avs_m3_train_3gpus

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
# S4 mul adapter
python train_avs.py --lr 0.0002  --projector_type mul --use_adapter --adapter_type mul --dataset s4 --batch_size 16 --wandb

# python train_avs.py --lr 0.0002 --num_workers 0 --model_name clap_adatper_sam --projector_type clap --evf_version evf_sam --batch_size 1 --use_adapter --adapter_type clap --wandb 
# python train_avs.py --lr 0.0002 --num_workers 0 --model_name mul_adatper_fc_train_sam --projector_type mul --evf_version evf_sam --normalize --batch_size 1 --use_adapter --adapter_type mul --wandb 



echo "Job finished at $(date)"
