#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --output=slurm-%x-%j.out
###SBATCH --image=nugent68/bccp:1.2
#SBATCH --account=m3363
#SBATCH -C haswell
#SBATCH --time=00:30:00
#SBATCH --job-name=3ptfnc_analysis

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME

# conda activate nbody2
source /global/common/software/m3035/conda-activate.sh 3.7

code_dir='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/cosmogan_pytorch/code/modules_threeptfcn'

### 2D validation data
# shifter python $code_dir/compute_3pct_single_file.py --nprocs 128 -n 6 --img_slice 32 --start_i 0 --end_i 500 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_5_4univ_cgan/norm_1_sig_0.8_train_val.npy -sfx 2d_val_0.8 -invtf

# shifter python $code_dir/compute_3pct_single_file.py --nprocs 128 -n 6 --img_slice 128 --start_i 0 --end_i 500 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_5_4univ_cgan/norm_1_sig_0.8_train_val.npy -sfx 2d_val_0.8 -invtf

# shifter python $code_dir/compute_3pct_single_file.py --nprocs 128 -n 6 --img_slice 128 --start_i 0 --end_i 500 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/128sq/20210122_095539_cgan_predict_0.5_m2/images/inference_label-0.5_epoch-15_step-35850.npy -sfx 2d_15-35850_pred-0.5 -invtf

# shifter python $code_dir/compute_3pct_single_file.py --nprocs 128 -n 6 --img_slice 128 --start_i 0 --end_i 500 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/128sq/20210119_174849_cgan_predict_0.65_m2/images/inference_label-0.65_epoch-14_step-34640.npy -sfx 2d_14-34640_pred-0.65 -invtf

# shifter python $code_dir/compute_3pct_single_file.py --nprocs 128 -n 6 --img_slice 128 --start_i 0 --end_i 500 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/128sq/20210122_074619_cgan_predict_1.1_m2/images/inference_label-1.1_epoch-13_step-31770.npy -sfx 2d_13-31770_pred-1.1 -invtf


# ## 3D validation data
# shifter python $code_dir/compute_3pct_single_file.py --nprocs 32 -n 3 --img_slice 32 --start_i 0 --end_i 32 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset3_smoothing_4univ_cgan_varying_sigma_128cube/Om0.3_Sg0.8_H70.0.npy -sfx 3d_val_0.8

## 3D validation data
# shifter python $code_dir/compute_3pct_single_file.py --nprocs 32 -n 3 --img_slice 32 --start_i 0 --end_i 32 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset1_smoothing_const_params_64cube_100k/val.npy -sfx 3d_dset1

python $code_dir/compute_3pct_parallelize_single_img.py --nprocs 32 -n 3 --img_slice 16 --start_i 0 --end_i 12 -f /global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset1_smoothing_const_params_64cube_100k/val.npy -sfx 3d_dset1 -invtf

echo "--end date" `date` `date +%s`
