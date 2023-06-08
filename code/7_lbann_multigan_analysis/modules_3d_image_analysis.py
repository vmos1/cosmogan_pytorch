#!/usr/bin/env python
# coding: utf-8

# # Test post compute 3D

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import subprocess as sp
import sys
import os
import glob
import pickle 
import itertools

from matplotlib.colors import LogNorm, PowerNorm, Normalize
from ipywidgets import *


### Transformation functions for image pixel values
def f_transform(x):
    return 2.*x/(x + 4.) - 1.

def f_invtransform(s):
    return 4.*(1. + s)/(1. - s)

## Grid plot 
def f_plot_grid(arr,cols=16,fig_size=(15,5)):
    ''' Plot a grid of images
    '''
    size=arr.shape[0]    
    rows=int(np.ceil(size/cols))
    print(rows,cols)
    
    fig,axarr=plt.subplots(rows,cols,figsize=fig_size, gridspec_kw = {'wspace':0, 'hspace':0})
    if rows==1: axarr=np.reshape(axarr,(rows,cols))
    if cols==1: axarr=np.reshape(axarr,(rows,cols))
    
    for i in range(min(rows*cols,size)):
        row,col=int(i/cols),i%cols
        try: 
            axarr[row,col].imshow(arr[i],origin='lower', cmap='YlGn', extent = [0, 128, 0, 128], norm=Normalize(vmin=-1., vmax=1.))
        # Drop axis label
        except Exception as e:
            print('Exception:',e)
            pass
        temp=plt.setp([a.get_xticklabels() for a in axarr[:-1,:].flatten()], visible=False)
        temp=plt.setp([a.get_yticklabels() for a in axarr[:,1:].flatten()], visible=False)


# ## Histogram modules

def f_batch_histogram(img_arr,bins,norm,hist_range):
    ''' Compute histogram statistics for a batch of images'''

    ## Extracting the range. This is important to ensure that the different histograms are compared correctly
    if hist_range==None : ulim,llim=np.max(img_arr),np.min(img_arr)
    else: ulim,llim=hist_range[1],hist_range[0]
#         print(ulim,llim)
    ### array of histogram of each image
    hist_arr=np.array([np.histogram(arr.flatten(), bins=bins, range=(llim,ulim), density=norm) for arr in img_arr]) ## range is important
    hist=np.stack(hist_arr[:,0]) # First element is histogram array
#         print(hist.shape)
    bin_list=np.stack(hist_arr[:,1]) # Second element is bin value 
    ### Compute statistics over histograms of individual images
    mean,err=np.mean(hist,axis=0),np.std(hist,axis=0)/np.sqrt(hist.shape[0])
    bin_edges=bin_list[0]
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return mean,err,centers
    

def f_pixel_intensity(img_arr,bins=25,label='validation',mode='avg',normalize=False,log_scale=True,plot=True, hist_range=None):
    '''
    Module to compute and plot histogram for pixel intensity of images
    Has 2 modes : simple and avg
        simple mode: No errors. Just flatten the input image array and compute histogram of full data
        avg mode(Default) : 
            - Compute histogram for each image in the image array
            - Compute errors across each histogram 
    '''
    
    norm=normalize # Whether to normalize the histogram
    
    if plot: 
        plt.figure()
        plt.xlabel('Pixel value')
        plt.ylabel('Counts')
        plt.title('Pixel Intensity Histogram')

        if log_scale: plt.yscale('log')
    
    if mode=='simple':
        hist, bin_edges = np.histogram(img_arr.flatten(), bins=bins, density=norm, range=hist_range)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if plot: plt.errorbar(centers, hist, fmt='o-', label=label)
        return hist,None
    
    elif mode=='avg': 
        ### Compute histogram for each image. 
        mean,err,centers=f_batch_histogram(img_arr,bins,norm,hist_range)

        if plot: plt.errorbar(centers,mean,yerr=err,fmt='o-',label=label)  
        return mean,err
    
def f_compare_pixel_intensity(img_lst,label_lst=['img1','img2'],bkgnd_arr=[],log_scale=True, normalize=True, mode='avg',bins=25, hist_range=None):
    '''
    Module to compute and plot histogram for pixel intensity of images
    Has 2 modes : simple and avg
    simple mode: No errors. Just flatten the input image array and compute histogram of full data
    avg mode(Default) : 
        - Compute histogram for each image in the image array
        - Compute errors across each histogram 
        
    bkgnd_arr : histogram of this array is plotting with +/- sigma band
    '''
    
    norm=normalize # Whether to normalize the histogram
    
    plt.figure()
    
    ## Plot background distribution
    if len(bkgnd_arr):
        if mode=='simple':
            hist, bin_edges = np.histogram(bkgnd_arr.flatten(), bins=bins, density=norm, range=hist_range)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.errorbar(centers, hist, color='k',marker='*',linestyle=':', label='bkgnd')

        elif mode=='avg':
            ### Compute histogram for each image. 
            mean,err,centers=f_batch_histogram(bkgnd_arr,bins,norm,hist_range)
            plt.plot(centers,mean,linestyle=':',color='k',label='bkgnd')
            plt.fill_between(centers, mean - err, mean + err, color='k', alpha=0.4)
    
    ### Plot the rest of the datasets
    for img,label,mrkr in zip(img_lst,label_lst,itertools.cycle('>^*sDHPdpx_')):     
        if mode=='simple':
            hist, bin_edges = np.histogram(img.flatten(), bins=bins, density=norm, range=hist_range)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            plt.errorbar(centers, hist, fmt=mrkr+'-', label=label)

        elif mode=='avg':
            ### Compute histogram for each image. 
            mean,err,centers=f_batch_histogram(img,bins,norm,hist_range)
#             print('Centers',centers)
            plt.errorbar(centers,mean,yerr=err,fmt=mrkr+'-',label=label)

    if log_scale: 
        plt.yscale('log')
        plt.xscale('symlog',linthreshx=50)

    plt.legend()
    plt.xlabel('Pixel value')
    plt.ylabel('Counts')
    plt.title('Pixel Intensity Histogram')


# ### Spectral modules

# In[ ]:


## numpy code
def f_radial_profile_3d(data, center=(None,None)):
    ''' Module to compute radial profile of a 2D image '''
    
    z, y, x = np.indices((data.shape)) # Get a grid of x and y values
    
    center=[]
    if not centers:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0, (z.max()-z.min())/2.0]) # compute centers
        
    # get radial values of every pair of points
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2+ + (z - center[2])**2)
    r = r.astype(np.int)
    
    # Compute histogram of r values
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel()) 
    radialprofile = tbin / nr
    
    return radialprofile[1:-1]

def f_compute_spectrum_3d(arr):
    '''
    compute spectrum for a 3D image
    '''
#     GLOBAL_MEAN=1.0
#     arr=((arr - GLOBAL_MEAN)/GLOBAL_MEAN)
    y1=np.fft.fftn(arr)
    y1=np.fft.fftshift(y1)
#     print(y1.shape)
    y2=abs(y1)**2
    z1=f_radial_profile_3d(y2)
    return(z1)
   
def f_batch_spectrum_3d(arr):
    batch_pk=np.array([f_compute_spectrum_3d(i) for i in arr])
    return batch_pk

### Code ###
def f_image_spectrum_3d(x,num_channels):
    '''
    Compute spectrum when image has a channel index
    Data has to be in the form (batch,channel,x,y)
    '''
    mean=[[] for i in range(num_channels)]    
    sdev=[[] for i in range(num_channels)]    

    for i in range(num_channels):
        arr=x[:,i,:,:,:]
#         print(i,arr.shape)
        batch_pk=f_batch_spectrum_3d(arr)
#         print(batch_pk)
        mean[i]=np.mean(batch_pk,axis=0)
        sdev[i]=np.var(batch_pk,axis=0)
    mean=np.array(mean)
    sdev=np.array(sdev)
    return mean,sdev


def f_plot_spectrum_3d(img_arr,plot=False,label='input',log_scale=True):
    '''
    Module to compute Average of the 1D spectrum for a batch of 3d images
    '''
    num = img_arr.shape[0]
    Pk = f_batch_spectrum_3d(img_arr)

    mean,std = np.mean(Pk, axis=0),np.std(Pk, axis=0)/np.sqrt(Pk.shape[0])
    # mean,std = np.mean(Pk, axis=0),np.std(Pk, axis=0)
    k=np.arange(len(mean))
    
    if plot: 
        plt.figure()
        plt.plot(k, mean, 'k:')
        plt.plot(k, mean + std, 'k-',label=label)
        plt.plot(k, mean - std, 'k-')
    #     plt.xscale('log')
        if log_scale: plt.yscale('log')
        plt.ylabel(r'$P(k)$')
        plt.xlabel(r'$k$')
        plt.title('Power Spectrum')
        plt.legend()

    return mean,std


def f_compare_spectrum_3d(img_lst,label_lst=['img1','img2'],bkgnd_arr=[],log_scale=True):
    '''
    Compare the spectrum of 2 sets s: 
    img_lst contains the set of images arrays, Each is of the form (num_images,height,width)
    label_lst contains the labels used in the plot
    '''
    plt.figure()
    
    ## Plot background distribution
    if len(bkgnd_arr):
        Pk= f_batch_spectrum_3d(bkgnd_arr)
        mean,err = np.mean(Pk, axis=0),np.std(Pk, axis=0)/np.sqrt(Pk.shape[0])
        k=np.arange(len(mean))
        plt.plot(k, mean,color='k',linestyle='-',label='bkgnd')    
        plt.fill_between(k, mean - err, mean + err, color='k',alpha=0.8)
    
    
    for img_arr,label,mrkr in zip(img_lst,label_lst,itertools.cycle('>^*sDHPdpx_')): 
        Pk= f_batch_spectrum_3d(img_arr)
        mean,err = np.mean(Pk, axis=0),np.std(Pk, axis=0)/np.sqrt(Pk.shape[0])

        k=np.arange(len(mean))
#         print(mean.shape,std.shape)
        plt.fill_between(k, mean - err, mean + err, alpha=0.4)
        plt.plot(k, mean, marker=mrkr, linestyle=':',label=label)

    if log_scale: plt.yscale('log')
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')
    plt.title('Power Spectrum')
    plt.legend()  
    

if __name__=='__main__':
    # ### Read data
    fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/results_from_other_code/pytorch/results/3d/20210108_101150_lambda2.0/images/best_spec_epoch-8_step-12630.npy'
    a1=np.load(fname)

    fname='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset1_smoothing_const_params_100k/full_with_smoothing_1.npy'
    val_arr=np.load(fname,mmap_mode='r')[-500:,0,:,:,:]
    print(a1.shape,val_arr.shape)

    # Histogram
    _,_=f_pixel_intensity(val_arr)
    img_lst=[a1,f_transform(val_arr)]
    label_lst=['a1','val']
    f_compare_pixel_intensity(img_lst,label_lst=['img1','img2'],bkgnd_arr=[],log_scale=True, normalize=True, mode='avg',bins=25, hist_range=None)

    # Spectrum
    _,_=f_plot_spectrum_3d(val_arr[:5],plot=True)
    img_lst=[a1[:5],val_arr[:5]]
    f_compare_spectrum_3d(img_lst)
