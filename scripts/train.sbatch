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

# python train_avs.py --lr 0.0002 --model_name av_fuse_av_p --projector_type mul --use_adapter --adapter_type mul --normalize --batch_size 3 --av_fuse --parallel --wandb
# python train_avs.py --lr 0.0002 --model_name av_fuse_av2_b_3 --projector_type mul --normalize --batch_size 3 --av_fuse --wandb

python train_avs.py --lr 0.0002  --projector_type mul --use_adapter --adapter_type mul --normalize --dataset ms3 --batch_size 1 

# python train_avs.py --lr 0.0002 --model_name multitask_fuse2 --projector_type mul --use_adapter --adapter_type mul --normalize --batch_size 1 --av_fuse --wandb
# python train_avs.py --lr 0.0002 --dataset s4 --model_name multitask --projector_type mul --normalize --batch_size 3 --wandb

# python train_avs.py --dataset s4 --model_name av_fuse_av --projector_type mul --normalize --batch_size 3 --av_fuse --wandb
# python train_avs.py --dataset s4 --projector_type mul --normalize --batch_size 3 --wandb

# python train_avs.py --lr 0.0002 --num_workers 0 --model_name clap_fc_train --projector_type clap --batch_size 3 --wandb 
# python train_avs.py --lr 0.0002 --num_workers 0 --model_name mul --projector_type mul --normalize --batch_size 3 --augmentation --audio_aug 0.2 

# python train_avs.py --lr 0.0002 --num_workers 0 --model_name test --projector_type mul --use_adapter --adapter_type mul --normalize --batch_size 1  --wandb
# python train_avs.py --lr 0.0002 --num_workers 0 --model_name clap_adapter_fc_train --projector_type clap --use_adapter --adapter_type clap --batch_size 1  --wandb


# python train_avs.py --lr 0.0002 --num_workers 0 --model_name clap_sam --projector_type clap --evf_version evf_sam --batch_size 1 --wandb 
# python train_avs.py --lr 0.0002 --num_workers 0 --model_name mul_sam --projector_type mul --evf_version evf_sam --normalize --batch_size 1 --wandb 


# python train_avs.py --lr 0.0002 --num_workers 0 --model_name clap_multi --projector_type clap --evf_version evf_sam_multitask --batch_size 1 --wandb 
# python train_avs.py --lr 0.0002 --num_workers 0 --model_name evf_sam_adapter --projector_type clap --evf_version evf_sam --wandb --batch_size 1 --use_adapter
# python train_avs.py --lr 0.0002 --num_workers 0 --model_name evf_sam_multi_adapter --projector_type clap --evf_version evf_sam_multitask --batch_size 1 --wandb --use_adapter


echo "Job finished at $(date)"
