#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --output=slurm-%x-%j.out
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=02:00:00
#SBATCH --job-name=3ptfnc

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME

I=$1
source /global/common/software/m3035/conda-activate.sh 3.7
code_dir=/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/code/modules_threeptfcn/simple_test
fname=/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/3d/20210506_175558_64cube_bs8_lr0.0006_nodes8_spec0.1_bestrun/images/gen_img_epoch-124_step-19390.npy

srun -n 64 python $code_dir/3pt_fcn_simple.py -n 8 --img_slice 64 -idx $I -f $fname -sfx run124-19390_3d_64cube

echo "--end date" `date` `date +%s`
