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
# S4 evf_sam_mul adapter 
python test_avs.py --dataset s4 --name mul_adapter_sam1_s4 --evf_version evf_sam --projector_type mul --use_adapter --adapter_type mul --weight_path weights/_mul_adapter_evf_sam_s4_0.0_0.0_best.pth



echo "Job finished at $(date)"
