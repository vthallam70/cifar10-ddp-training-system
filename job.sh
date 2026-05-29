#!/bin/bash
#SBATCH -J cifar10_ddp_run
#SBATCH --time=00:30:00
#SBATCH --account=cmda3634_rjh
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -o slurm.cifar10.%j.out
#SBATCH -e slurm.cifar10.%j.err

module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MASTER_ADDR=localhost
export MASTER_PORT=$((20000 + RANDOM % 40000))

IFS=',' read -r -a ngpus <<< "$CUDA_VISIBLE_DEVICES"

srun python3 cifar10_ddp.py \
    --num-nodes 1 \
    --node-id 0 \
    --num-gpus ${#ngpus[@]} \
    --batch-size 64 \
    --lr 0.001 \
    --epochs 12 \
    --target-accuracy 0.85 \
    --patience 2 \
    --model WideResNet