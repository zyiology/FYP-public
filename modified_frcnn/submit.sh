#!/bin/bash
#SBATCH --time=600
#SBATCH --job-name=cust_train
#SBATCH --gpus=a100mig:1
#SBATCH --partition=long

# xgpg=a100, xgph=a100mig, xgpf=t4, xgpe=titanrtx, xgpd=2xtitanv, xgpc=titanv

nvidia-smi

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate frcnn

#python test.py
#CUDA_LAUNCH_BLOCKING=1 python train_custom_fasterrcnn_attention.py
python -u train_custom_fasterrcnn_attention.py $SLURM_JOB_ID

nvidia-smi