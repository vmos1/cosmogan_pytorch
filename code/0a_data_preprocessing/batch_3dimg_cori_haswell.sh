#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=regular
#SBATCH --job-name=3d_data_extraction_haswell
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=haswell
#SBATCH --account=m3363
#################

conda activate v3
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "--start date" `date` `date +%s`
# srun -n 1 python 4_slice3d_parallel.py --cores 64 -p Om0.3_Sg1.1_H70.0 --smoothing -d /global/cfs/cdirs/m3363/www/cosmoUniverse_2020_11_4parE_cGAN/Sg1.1 -i 128
# srun -n 1 python 4_slice3d_parallel.py --cores 64 -p Om0.3_Sg0.8_H70.0 --smoothing -d /global/cfs/cdirs/m3363/www/cosmoUniverse_2020_11_4parE_cGAN/Sg0.8 -i 128
# srun -n 1 python 4_slice3d_parallel.py --cores 64 -p Om0.3_Sg0.65_H70.0 --smoothing -d /global/cfs/cdirs/m3363/www/cosmoUniverse_2020_11_4parE_cGAN/Sg0.65 -i 128
# srun -n 1 python 4_slice3d_parallel.py --cores 64 -p Om0.3_Sg0.5_H70.0 --smoothing -d /global/cfs/cdirs/m3363/www/cosmoUniverse_2020_11_4parE_cGAN/Sg0.5 -i 128
srun -n 1 python 4_slice3d_parallel.py --cores 30 -p full_1 --smoothing -d /global/project/projectdirs/m3363/www/cosmoUniverse_2019_08_const -i 512
conda deactivate
echo "--end date" `date` `date +%s`
