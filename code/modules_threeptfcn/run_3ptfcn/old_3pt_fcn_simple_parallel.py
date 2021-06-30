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
from mpi4py import MPI

### Modules #####

### Transformation functions for image pixel values
def f_transform(x):
    return 2.*x/(x + 4.) - 1.

def f_invtransform(s):
    return 4.*(1. + s)/(1. - s + 1e-10)

def f_make_catalog_3d(img,comm):
    
    x=np.arange(img.shape[0]) 
    y=np.arange(img.shape[1])
    z=np.arange(img.shape[2])

    coord=np.array([(i,j,k) for i in x for j in y for k in z]) ## Form is (x,y,z)

    ip_dict={}
    ip_dict['Position'] = coord
    ip_dict['Mass'] = img.flatten()
    if comm.rank==0:
        catalog=ArrayCatalog(ip_dict)
    
    else:
        empty_dict={}
        empty_dict['Position']=coord[:0]
        empty_dict['Mass']=img.flatten()[:0]
        catalog = ArrayCatalog(empty_dict) 
    
    return catalog


def f_parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run script to train GAN using LBANN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arg = parser.add_argument
    
    add_arg('--img_slice','-s', type=int, default=128, help='Size of image to slice')
    add_arg('--ncorrs','-n', type=int, default=4, help='Number of correlators to use')
    add_arg('--edgesize','-e', type=int, default=10, help='Edge size')
    add_arg('--boxsize','-b', type=int, default=20, help='Size of box')
    add_arg('--fname','-f',  type=str,default='/mnt/laptop/data/2d_images_50.npy', help='File name with images')
    add_arg('--suffix','-sfx',  type=str,default='sig_0.5', help='Suffix for stored file')
    add_arg('--index','-idx', type = int, default=0, help='Index of image input file')
#     add_arg('--nprocs','-np', type=int, default=32, help='Number of parallel process (=num cores)')

    return parser.parse_args()

def f_write_corr(img_index,a1,num_corrs,edge_size,box_size,slice_idx,data_dir,suffix,comm):
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
        cat1=f_make_catalog_3d(img,comm)
    
    ## compute 3 ptfnc
    t1=time.time()
    obj1=SimulationBox3PCF(cat1,list(range(num_corrs)),edges=np.arange(1,edge_size,1),BoxSize=box_size,weight='Mass')
    t2=time.time()
    print("Time 1 for index {0}: {1}".format(img_index,t2-t1))
    op1=obj1.run()
    t3=time.time()
    print("Time 2 for index {0}: {1}".format(img_index,t3-t2))

    ### Extract and Save correlators as 3D array to file
    if comm.rank==0:
        corr_list=[]
        for i in op1.variables:  
            corr_list.append(op1[i]) 
        
        arr=np.array(corr_list)
        print(arr.shape)
        ## Save correlators
        fname='img_'+str(img_index)+'-corr_'+suffix+'.npy'
        np.save(data_dir+fname,arr)

   
if __name__=="__main__":
   
    comm=MPI.COMM_WORLD 
    args=f_parse_args()
    print(args)
    img_slice=args.img_slice
    num_corrs=args.ncorrs
    fname=args.fname
    name_suffix=args.suffix
    box_size=args.boxsize
    edge_size=args.edgesize
    idx=args.index
    
    data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/3ptfnc_stored_results/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    ## Load image file
    print("Reading file",fname)
    
    a1=np.load(fname)

    if len(a1.shape)==4:
        a1=a1[:,0,:,:]

    elif len(a1.shape)==5:
        a1=a1[:,0,:,:,:]
    else: 
        print(a1.shape)
        raise SystemError

    a1=a1[:1]
    print('Shape of input image file',a1.shape)
    a1=f_invtransform(a1) # Generated images need to be inv transformed

    # Compute correlator
    f_write_corr(0,a1,num_corrs,edge_size,box_size,slice_idx=img_slice,data_dir=data_dir,suffix=name_suffix,comm=comm)
