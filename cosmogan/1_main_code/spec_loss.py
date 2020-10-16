
import numpy as np
import torch

############
### Numpy functions ### Not used in the code. Just to test the pytorch functions
############
def f_radial_profile(data, center=(None,None)):
    ''' Module to compute radial profile of a 2D image '''
    y, x = np.indices((data.shape)) # Get a grid of x and y values
    
    if center[0]==None and center[1]==None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0]) # compute centers
        
    # get radial values of every pair of points
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    
    # Compute histogram of r values
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel()) 
    radialprofile = tbin / nr
    
    return radialprofile

def f_compute_spectrum(arr):
    y1=np.fft.fft2(arr)
    y2=abs(y1)
    z1=f_radial_profile(y2)
    return(z1)
    
def f_compute_batch_spectrum(arr):
    batch_pk=np.array([f_compute_spectrum(i) for i in arr])
    return batch_pk


def f_image_spectrum(x,num_channels):
    '''
    Data has to be in the form (batch,channel,x,y)
    '''
    print(x.shape)
    mean=[[] for i in range(num_channels)]    
    sdev=[[] for i in range(num_channels)]    

    for i in range(num_channels):
        arr=x[:,i,:,:]
#         print(i,arr.shape)
        batch_pk=f_compute_batch_spectrum(arr)
#         print(batch_pk)
        mean[i]=np.mean(batch_pk,axis=0)
        sdev[i]=np.std(batch_pk,axis=0)
    mean=np.array(mean)
    sdev=np.array(sdev)
    return mean,sdev

####################
### Pytorch code ###
####################
def f_torch_radial_profile(img, center=(None,None)):
    ''' Module to compute radial profile of a 2D image '''
    
    y,x=torch.meshgrid(torch.arange(0,img.shape[0]),torch.arange(0,img.shape[1])) # Get a grid of x and y values
    if center[0]==None and center[1]==None:
        center = torch.Tensor([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0]) # compute centers

    # get radial values of every pair of points
    r = torch.sqrt((x - center[0])**2 + (y - center[1])**2)
    r= r.int()

#     print(r.shape,img.shape)
    # Compute histogram of r values
    tbin=torch.bincount(torch.reshape(r,(-1,)),weights=torch.reshape(img,(-1,)).type(torch.DoubleTensor))
    nr = torch.bincount(torch.reshape(r,(-1,)))
    radialprofile = tbin / nr
    
    return radialprofile


def f_torch_compute_spectrum(arr):
    y1=torch.rfft(arr,signal_ndim=2,onesided=False)
    ## Absolute value of each complex number (last index is real/imag part)
    y2=torch.sqrt(y1[:,:,0]**2+y1[:,:,1]**2)
    z1=f_torch_radial_profile(y2)
    return(z1)


def f_torch_compute_batch_spectrum(arr):
    
    batch_pk=torch.stack([f_torch_compute_spectrum(i) for i in arr])
    
    return batch_pk


def f_torch_image_spectrum(x,num_channels):
    '''
    Data has to be in the form (batch,channel,x,y)
    '''
    mean=[[] for i in range(num_channels)]    
    sdev=[[] for i in range(num_channels)]    

    for i in range(num_channels):
        arr=x[:,i,:,:]
#         print(i,arr.shape)
        batch_pk=f_torch_compute_batch_spectrum(arr)
#         print(batch_pk.shape)
        mean[i]=torch.mean(batch_pk,axis=0)
        sdev[i]=torch.std(batch_pk,axis=0)
        
    mean=torch.stack(mean)
    sdev=torch.stack(sdev)
    return mean,sdev

### Losses 
def loss_spectrum(spec_mean,spec_mean_ref,spec_std,spec_std_ref,image_size):
    ''' Loss function for the spectrum : mean + variance '''
    
    # Log ( sum( batch value - expect value) ^ 2 ))
    
    idx=int(image_size/2) ### For the spectrum, use only N/2 indices for loss calc.
    
#    spec_mean=torch.log(torch.mean(torch.pow(spec_mean[:,idx]-spec_mean_ref[:,idx],2)))
#    spec_sdev=torch.log(torch.mean(torch.pow(spec_std[:,idx]-spec_std_ref[:,idx],2)))
    spec_mean=torch.mean(torch.pow(spec_mean[:,idx]-spec_mean_ref[:,idx],2))
    spec_sdev=torch.mean(torch.pow(spec_std[:,idx]-spec_std_ref[:,idx],2))
     
    lambda1=0.002;lambda2=0.002;
    ans=lambda1*spec_mean+lambda2*spec_sdev
    return ans.item()

def loss_hist(data,hist_data):
    
    hist_sample=torch.histc(data,bins=50)
    ## A kind of normalization of histograms: divide by total sum
    hist_sample=hist_sample/torch.sum(hist_sample)
    hist_data=hist_data/torch.sum(hist_data)

    lambda1=1000.0
    #return torch.log(torch.mean(torch.pow(hist_sample-hist_data,2))).item()
    return lambda1*torch.mean(torch.pow(hist_sample-hist_data,2)).item()


