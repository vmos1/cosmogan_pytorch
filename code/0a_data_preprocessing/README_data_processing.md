##  Steps to perform data extraction and preparation

Steps:
1. Run `1_slice_parallel.py` via the batch scripts `batch_cori_haswell` or `batch_cori_haswell` to create data slices. This code is parallelized.
2. Copy the created .npy file to the right location
3. Run `2_pre_train.npy` on this file to generate train-validation datasets. It also generates the train and validation datasets with normalization applied (needed for lbann).

Notebooks:
- `0_explore_hdf5_file.ipynb` is useful for figuring out the contents of the input .hdf5 files
- `1b_slice_data.ipynb` performs the data extraction on a much smaller scale without parallelization