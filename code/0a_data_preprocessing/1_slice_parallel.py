# Code to extract 2D images from .hdf5 files
### June 9, 2020 
##### Venkitesh Ayyar (vpa@lbl.gov)

import numpy as np
import h5py
import os
import sys
import glob
import argparse
import time

from scipy.ndimage import gaussian_filter   ### For gaussian filtering

## modules for parallelization of python for loop
from multiprocessing import Pool
from functools import partial

#######################################
#######################################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Code to extract 2D images from 3D .hdf5 files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
#     add_arg('--batch_size','-b',type=int,default=100, help='Number of samples in each temp file')
    add_arg('--cores','-c',type=int,default=20,help='Number of parallel jobs you want to start')
    add_arg('--smoothing','-s',action='store_true',default=False,help='Whether to apply Gaussian smoothing')
    add_arg('--file_prefix','-p', type=str, default='full_', help='Prefix of the file name that stores the result')
    add_arg('--data_dir','-d', type=str, default='/global/project/projectdirs/m3363/www/cosmoUniverse_2019_08_const', help='Location of the .hdf5 files')
    add_arg('--splice','-sp',type=int,default=8,help='The splice distance between successive points along any axes.')
    add_arg('--img_dim','-i',type=int,default=128,help='Dimension of 2D image.')
    add_arg('--mode','-m', type=str, choices=['full','xaxis'], default='full', help='2 modoes of operation. full splices along all 3 axes. xaxis will splice only along x direction.')

    return parser.parse_args()


def f_get_img_slices(f_list,img_dim=128,smoothing=False,mode='full',splice_interval=8):
    '''
    Get 2D slices of 512^3 images along all 3 axes
    splice_interval is the spacing between layers 
    '''
    slices = []
    perside = 512//img_dim
    
    for fname in f_list: ### Iterate through each file
        with h5py.File(fname, 'r') as inputdata:
            img_arr=np.array(inputdata['full'])
            del(inputdata) ## Free memory
            if smoothing: 
                img_arr=img_arr.astype(np.float32)
                img_arr=gaussian_filter(img_arr,sigma=0.5,mode='wrap') ### Implement Gaussian smoothing. This takes time
             
            for i1 in range(perside): 
                for i2 in range(perside):
                    # Select slices along planes : xy,yz, zx, for redshift=0 
                    # (128 * 128 images from 512 x 512 images-> 16 images)
                    ## yz plane: 
                    data = img_arr[::splice_interval, i1*img_dim:(i1+1)*img_dim, i2*img_dim:(i2+1)*img_dim, 0]
                    data2=np.transpose(data,(0,1,2)) ### Transpose to get array in the form (samples,128,128)
                    slices.append(np.expand_dims(data2, axis=1))
                     
                    if mode=='xaxis': continue ### Use only slices from yz plane
                    
                    ## xy plane: 
                    data = img_arr[i1*img_dim:(i1+1)*img_dim,i2*img_dim:(i2+1)*img_dim,::splice_interval,0]
                    data2=np.transpose(data,(2,0,1)) ### Transpose to get array in the form (samples,128,128)
                    slices.append(np.expand_dims(data2, axis=1))      

                    ## xz plane: 
                    data = img_arr[i1*img_dim:(i1+1)*img_dim,::splice_interval,i2*img_dim:(i2+1)*img_dim,0]
                    data2=np.transpose(data,(1,0,2))  ### Transpose to get array in the form (samples,128,128)
                    slices.append(np.expand_dims(data2, axis=1))
    
        print('Sliced %s'%fname)
    slices = np.concatenate(slices)
    np.random.shuffle(slices)  ### Shuffle along first axis after extracting all slices 
    print(slices.shape)
    
    return slices
    

def f_write_temp_files(count,f_list,img_dim,save_location,smoothing,mode,splice_interval,file_prefix):
    '''
    Function to compute slices and write temporary files
    Arguments: count: index of idx array,f_list: list of files, batch_size : size of batch and save_location
    Takes in indices count*batch_size -> (count+1)*batch_size
    Can be used to run in parallel
    '''
    t3=time.time()
    prefix='temp_data_{0}_{1}'.format(file_prefix,count)
    
    files_list=[f_list[count]]
    slices=f_get_img_slices(files_list,img_dim,smoothing=smoothing,mode=mode,splice_interval=splice_interval)
    np.save(save_location+prefix+'.npy',slices)
    t4=time.time()
    print("Extraction time for count ",count,":",t4-t3)


def f_concat_temp_files(num_batches,save_location,file_prefix):
    '''
    Function to concatenate temp files to creat the full file.
    Steps: get data from temp files, stack numpy arrays and delete temp files
    '''
    if num_batches<1:
        print('zero temp files',num_batches)
        return 0
    
    for count in np.arange(num_batches):
        prefix='temp_data_%s_%s'%(file_prefix,count)
        f1=prefix+'.npy'
        
        xs=np.load(save_location+f1)
        ### Join arrays to create large array    
        if count==0:x=xs;
        else:x = np.vstack((x,xs))
        os.remove(save_location+f1) # Delete temp file
    print("Deleted temp files")
    
    return x

#######################################
#######################################
if __name__=='__main__':
    
    dest_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/temp_data/'
    args=parse_args()
    procs,file_prefix=args.cores,args.file_prefix
#     file_prefix='full_with_smoothing_1'
    print('processes {0}'.format(procs))
    
    if args.smoothing: print('Implementing Gaussian smoothing')
    
    ### Extract data
    t1=time.time()
    data_dir=args.data_dir
    print("Reading data from :",data_dir)
    # Extract list of hdf5 files
    f_list=glob.glob(data_dir+'/*.hdf5')
    t2=time.time()
    print("Setup time reading file names ",t2-t1)
    
    num_batches=len(f_list) ## Each file read in separately
    print("Number of temp files: ",num_batches)
    if num_batches<1:
        print('Exiting: Zero temp files',num_batches)
        raise SystemExit
    
    ### Get 2D slices and save to temp files
    ##### This part is parallelized
    with Pool(processes=procs) as p:
        ## Fixing the last 2 arguments of the function. The map takes only functions with one argument
        f_temp_func=partial(f_write_temp_files,f_list=f_list,img_dim=args.img_dim,save_location=dest_dir,smoothing=args.smoothing,mode=args.mode,splice_interval=args.splice,file_prefix=file_prefix)
        ### Map the function for each batch. This is the parallelization step
        p.map(f_temp_func, np.arange(num_batches))
    t3=time.time()
    
    ### Concatenate temp files
    t4=time.time()
    img=f_concat_temp_files(num_batches,save_location=dest_dir,file_prefix=file_prefix)
    t5=time.time()
    print("Time for concatenation of file:",t5-t4)
    print("total number of images",img.shape)
    
    ### Shuffle contents again
    t6=time.time()
    np.random.shuffle(img)
    t7=time.time()
    print("Time for final shuffling of entries",t7-t6)
    
    ### Save concatenated files
    fname=dest_dir+file_prefix+'.npy'
    print("Saving data at: ",fname)
    np.save(fname,img)
    t8=time.time()
    print("Total time",t8-t1)
######################################################
######################################################
