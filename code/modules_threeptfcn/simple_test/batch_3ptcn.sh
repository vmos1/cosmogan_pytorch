#!/bin/bash
#################
#SBATCH --nodes=10
#SBATCH --qos=regular
#SBATCH --output=slurm-%x-%j.out
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=0:30:00
#SBATCH --job-name=3ptfnc

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME

I=$1
source /global/common/software/m3035/conda-activate.sh 3.7
code_dir=/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/code/modules_threeptfcn/simple_test
fname=/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset2a_3dcgan_4univs_64cube_simple_splicing/norm_1_sig_0.8_train_val.npy

srun -n 640 python $code_dir/3pt_fcn_simple.py -n 8 --img_slice 64 -idx $I -f $fname -sfx val_3d_64cube-sigma0.8

echo "--end date" `date` `date +%s`
