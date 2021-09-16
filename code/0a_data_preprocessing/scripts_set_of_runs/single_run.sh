
## Script to run extraction on a single folder in parallel
i=$1
prefix=$(basename $i); 
full_path=$(realpath $i); 
echo $prefix; echo $full_path; 
code_loc='/global/u1/v/vpa/project/jpt_notebooks/Cosmology/Cosmo_GAN/repositories/lbann_cosmogan/0a_data_preprocessing/1_slice_parallel.py'
#python $code_loc --cores 1 -p $prefix --smoothing --mode full --splice 16 -d $full_path;
python $code_loc --cores 1 -p $prefix --mode full --splice 16 -d $full_path -i 512;


