#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -A m3363
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH -o sout/%j.out


module load cgpu
module load pytorch/v1.6.0-gpu  
# module load pytorch
# conda activate v3

echo "hello"
srun python main.py
#srun -n1 python -m torch.distributed.launch --nproc_per_node=4 main.py
