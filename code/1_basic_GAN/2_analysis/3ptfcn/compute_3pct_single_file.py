### Code to compute and store the 3 point function for a given image
#### Dec 3, 2020

import numpy as np
#import matplotlib.pyplot as plt
# from scipy.interpolate import InterpolatedUnivariateSpline

from nbodykit.lab import *
from nbodykit import setup_logging, style

from multiprocessing import Pool
from functools import partial
import time
import argparse
import os
import glob

### Modules #####

### Transformation functions for image pixel values
def f_transform(x):
    return 2.*x/(x + 4.) - 1.

def f_invtransform(s):
    return 4.*(1. + s)/(1. - s)

def f_make_catalog_2d(img):
    ''' Make catalog for 2d images'''
    x=np.arange(img.shape[0]) 
    y=np.arange(img.shape[1])

    coord=np.array([(i,j,0) for i in x for j in y]) ## Form is (x,y,0)

    ip_dict={}
    ip_dict['Position'] = coord
    ip_dict['Mass'] = img.flatten()
    catalog=ArrayCatalog(ip_dict)
    
    return catalog

def f_make_catalog_3d(img):
    
    x=np.arange(img.shape[0]) 
    y=np.arange(img.shape[1])
    z=np.arange(img.shape[2])

    coord=np.array([(i,j,k) for i in x for j in y for k in z]) ## Form is (x,y,z)

    ip_dict={}
    ip_dict['Position'] = coord
    ip_dict['Mass'] = img.flatten()
    catalog=ArrayCatalog(ip_dict)
    
    return catalog


def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--img_slice','-s', type=int, default=128, help='Size of image to slice')
    add_arg('--ncorrs','-n', type=int, default=4, help='Number of correlators to use')
    add_arg('--fname','-f',  type=str,default='/mnt/laptop/data/2d_images_50.npy', help='File name with images')
    add_arg('--suffix','-sfx',  type=str,default='sig_0.5', help='Suffix for stored file')
    add_arg('--start_i','-si', type = int, default=0, help='Start index of image input file')
    add_arg('--end_i','-ei', type = int, default=1, help='End index of image input file')
    add_arg('--invtf','-invtf',action='store_true', help='Run f_invtransform')
    add_arg('--nprocs','-np', type=int, default=32, help='Number of parallel process (=num cores)')


    return parser.parse_args()

def f_write_corr(img_index,a1,num_corrs,slice_idx,data_dir,suffix):
    '''
    Compute 3ptfcn for a given image index and write to file
    '''
    if len(a1.shape)-1==2: 
#         print("Image is 2d")
        img=a1[img_index,:slice_idx,:slice_idx]
        cat1=f_make_catalog_2d(img)
    elif len(a1.shape)-1==3:
#         print("Image is 3d")
        img=a1[img_index,:slice_idx,:slice_idx,:slice_idx]
        cat1=f_make_catalog_3d(img)
    
    print(img.shape)

    ## compute 3 ptfnc
    t1=time.time()
    obj1=SimulationBox3PCF(cat1,list(range(num_corrs)),edges=np.arange(1,5,1),BoxSize=10,weight='Mass')
    t2=time.time()
    print("Time 1 for index {0}: {1}".format(img_index,t2-t1))
    op1=obj1.run()
    t3=time.time()
    print("Time 2 for index {0}: {1}".format(img_index,t3-t2))

    ### Extract and Save correlators as 3D array to file
    corr_list=[]
    for i in op1.variables:  
        corr_list.append(op1[i]) 
    
    arr=np.array(corr_list)
    print(arr.shape)
    ## Save correlators
    fname='img_'+str(img_index)+'-corr_'+suffix+'.npy'
    np.save(data_dir+fname,arr)

def f_concat_temp_files(num_batches,save_location,file_prefix,file_suffix):
    '''
    Function to concatenate temp files to create the full file.
    Steps: get data from temp files, stack numpy arrays and delete temp files
    '''
    if num_batches<1:
        print('zero temp files',num_batches)
        return 0
    
    x = np.vstack([np.expand_dims(np.load(save_location+'%s_%s-corr_%s'%(file_prefix,count,file_suffix)+'.npy'),axis=0) for count in np.arange(num_batches)])
    print(x.shape)
    
    # Delete temp files
    for count in np.arange(num_batches):
        f1='%s_%s-corr_%s'%(file_prefix,count,file_suffix)+'.npy'
        os.remove(save_location+f1)
    print("Deleted temp files")
    
    return x
    
    
if __name__=="__main__":
    
    args=f_parse_args()
    print(args)
    img_size=args.img_slice
    num_corrs=args.ncorrs
    fname=args.fname
    name_suffix=args.suffix
    start_i=args.start_i
    end_i=args.end_i
    procs=args.nprocs # Number of parallel processes


#    main_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/128sq/'
#     fldr='20210115_133716_lambda2.0'
#     flist=glob.glob(main_dir+fldr+'/images/gen_img_epoch-*_step-110.npy')
#     data_dir=main_dir+fldr+'/3ptfnc_stored_results/'
    
#     main_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_5_4univ_cgan/'
#     fldr=''
#     flist=glob.glob(main_dir+'Om0.3_Sg{0}_H70.0.npy'.format(name))
    
#     main_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset3_smoothing_4univ_cgan_varying_sigma_128cube/'
#     fldr=''
#     fname=main_dir+'Om0.3_Sg{0}_H70.0.npy'.format(name)
#     flist=glob.glob(fname)
    
    flist=[fname]
    data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/3ptfnc_stored_results/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print(flist)
    ## Load image file
    print("Reading file",fname)
    idx_lst=np.arange(start_i,end_i,1)
    num_imgs=end_i-start_i
    
    for count,fle in enumerate(flist):
        a1=np.load(fle)[start_i:end_i]
        
        if len(a1.shape)==4:
            a1=a1[:,0,:,:]
        ## fix for pytorch code file format
#         if len(a1.shape)==3:
#             a1=a1[:,:,:] 

        elif len(a1.shape)==5:
            a1=a1[:,0,:,:,:]
        else: 
            print(a1.shape)
            raise SystemError
        
        print('Shape of input image file',a1.shape)

        if args.invtf:
            print('Invtransform:',args.invtf)
            a1=f_invtransform(a1) # Generated images need to be inv transformed
        assert np.max(a1)>100, "Incorrect scaling for images. Need to apply inv_transform"
        
#         f_write_corr(0,a1,num_corrs,slice_idx=img_size,data_dir=data_dir)
        
        # initialize the task manager to run the tasks
        with Pool(processes=procs) as p:
            f_temp_func=partial(f_write_corr,a1=a1,num_corrs=num_corrs,slice_idx=img_size,data_dir=data_dir,suffix=name_suffix)
            p.map(f_temp_func,np.array(idx_lst))
        
        ## Combine files for same input file
        
        a1=f_concat_temp_files(num_imgs,data_dir,'img',name_suffix)
        fname=data_dir+'3pt_corr_{0}_{1}.npy'.format(count,name_suffix)
        np.save(fname,a1)