#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --output=slurm-%x-%j.out
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=00:30:00
#SBATCH --job-name=3ptfnc_analysis

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME

# conda activate nbody2
source /global/common/software/m3035/conda-activate.sh 3.7

code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/code/modules_threeptfcn'


python $code_dir/compute_3pct_parallelize_single_img.py --nprocs 32 -n 3 --img_slice 16 --start_i 0 --end_i 12 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset1_smoothing_const_params_64cube_100k/val.npy -sfx 3d_dset1 -invtf

echo "--end date" `date` `date +%s`
