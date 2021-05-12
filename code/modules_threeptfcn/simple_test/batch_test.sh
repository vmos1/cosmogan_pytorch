#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --output=slurm-%x-%j.out
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=00:05:00
#SBATCH --job-name=3ptfnc_analysis

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME

# conda activate nbody2
source /global/common/software/m3035/conda-activate.sh 3.7
#bcast-pip https://github.com/bccp/nbodykit/archive/master.zip

code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/code/modules_threeptfcn/simple_test'

srun -n 1 python $code_dir/3pt_fcn_simple.py -n 6 --img_slice 16 -idx 0 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset1_smoothing_const_params_64cube_100k/val.npy -sfx 3d_dset1

echo "--end date" `date` `date +%s`

