#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --qos=regular
#SBATCH --job-name=lbann_data_extraction_haswell
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=haswell
#SBATCH --account=m3363
#################

conda activate v3
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "--start date" `date` `date +%s`
###srun -n 1 -c 30 python 1_slice_parallel.py --cores 25 -p full_1_ --mode full -d /global/cfs/cdirs/m3363/www/cosmoUniverse_2020_08_4parEgrid --splice 16
srun -n 1 -c 30 python 1_slice_parallel.py --cores 25 -p Om0.3_Sg1.1_H70.0 --smoothing --mode full --splice 8 -d /global/cfs/cdirs/m3363/www/cosmoUniverse_2020_08_4parEgrid/Om0.3_Sg1.1_H70.0/ -i 128
conda deactivate
echo "--end date" `date` `date +%s`
