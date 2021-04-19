
import numpy as np
import torch
# import torch.fft


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
    
    return radialprofile[1:-1]

def f_compute_spectrum(arr,GLOBAL_MEAN=0.9998563):
    
    
    arr=((arr - GLOBAL_MEAN)/GLOBAL_MEAN)
    y1=np.fft.fft2(arr)
    y1=fftpack.fftshift(y1)

    y2=abs(y1)**2
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

def f_get_rad(img):
    ''' Get the radial tensor for use in f_torch_get_azimuthalAverage '''
    
    height,width,depth=img.shape[-3:]
    # Create a grid of points with x and y and z coordinates
    z,y,x = np.indices([height,width,depth])
    
    center=[]
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0, (z.max()-z.min())/2.0])

    # Get the radial coordinate for every grid point. Array has the shape of image
    r= torch.tensor(np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2))
        
    # Get sorted radii
    ind = torch.argsort(torch.reshape(r, (-1,)))

    return r.detach(),ind.detach()


def f_torch_get_azimuthalAverage(image,r,ind):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    source: https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    
#     height, width = image.shape
#     # Create a grid of points with x and y coordinates
#     y, x = np.indices([height,width])

#     if not center:
#         center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

#     # Get the radial coordinate for every grid point. Array has the shape of image
#     r = torch.tensor(np.hypot(x - center[0], y - center[1]))

#     # Get sorted radii
#     ind = torch.argsort(torch.reshape(r, (-1,)))

    r_sorted = torch.gather(torch.reshape(r, ( -1,)),0, ind)
    i_sorted = torch.gather(torch.reshape(image, ( -1,)),0, ind)
    
    # Get the integer part of the radii (bin size = 1)
    r_int=r_sorted.to(torch.int32)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = torch.reshape(torch.where(deltar)[0], (-1,))    # location of changes in radius
    nr = (rind[1:] - rind[:-1]).type(torch.float)       # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    
    csum = torch.cumsum(i_sorted, axis=-1)
    tbin = torch.gather(csum, 0, rind[1:]) - torch.gather(csum, 0, rind[:-1])
    radial_prof = tbin / nr

    return radial_prof

def f_torch_fftshift(real, imag):
    for dim in range(0, len(real.size())):
        real = torch.roll(real, dims=dim, shifts=real.size(dim)//2)
        imag = torch.roll(imag, dims=dim, shifts=imag.size(dim)//2)
    return real, imag

def f_torch_compute_spectrum(arr,r,ind):
    
    GLOBAL_MEAN=1.0
    arr=(arr-GLOBAL_MEAN)/(GLOBAL_MEAN)
    
    y1=torch.rfft(arr,signal_ndim=3,onesided=False)
    real,imag=f_torch_fftshift(y1[:,:,:,0],y1[:,:,:,1])    ## last index is real/imag part  ## Mod for 3D
    
#     # For pytorch 1.8
#     y1=torch.fft.fftn(arr,dim=(-3,-2,-1))
#     real,imag=f_torch_fftshift(y1.real,y1.imag)    
    
    y2=real**2+imag**2     ## Absolute value of each complex number
    z1=f_torch_get_azimuthalAverage(y2,r,ind)     ## Compute radial profile
    return z1

def f_torch_compute_batch_spectrum(arr,r,ind):
    
    batch_pk=torch.stack([f_torch_compute_spectrum(i,r,ind) for i in arr])
    
    return batch_pk

def f_torch_image_spectrum(x,num_channels,r,ind):
    '''
    Data has to be in the form (batch,channel,x,y)
    '''
    mean=[[] for i in range(num_channels)]    
    sdev=[[] for i in range(num_channels)]    

    for i in range(num_channels):
        arr=x[:,i,:,:,:] # Mod for 3D
        batch_pk=f_torch_compute_batch_spectrum(arr,r,ind)
        mean[i]=torch.mean(batch_pk,axis=0)
#         sdev[i]=torch.std(batch_pk,axis=0)/np.sqrt(batch_pk.shape[0])
#         sdev[i]=torch.std(batch_pk,axis=0)
        sdev[i]=torch.var(batch_pk,axis=0)
    
    mean=torch.stack(mean)
    sdev=torch.stack(sdev)
        
    return mean,sdev

def f_compute_hist(data,bins):
    
    try: 
        hist_data=torch.histc(data,bins=bins)
        ## A kind of normalization of histograms: divide by total sum
        hist_data=(hist_data*bins)/torch.sum(hist_data)
    except Exception as e:
        print(e)
        hist_data=torch.zeros(bins)

    return hist_data

### Losses 
def loss_spectrum(spec_mean,spec_mean_ref,spec_std,spec_std_ref,image_size,lambda_spec_mean,lambda_spec_var):
    ''' Loss function for the spectrum : mean + variance 
    Log(sum( batch value - expect value) ^ 2 )) '''
    
    idx=int(image_size/2) ### For the spectrum, use only N/2 indices for loss calc.
    ### Warning: the first index is the channel number.For multiple channels, you are averaging over them, which is fine.
        
    spec_mean=torch.log(torch.mean(torch.pow(spec_mean[:,:idx]-spec_mean_ref[:,:idx],2)))
    spec_sdev=torch.log(torch.mean(torch.pow(spec_std[:,:idx]-spec_std_ref[:,:idx],2)))
    
    lambda1=lambda_spec_mean;
    lambda2=lambda_spec_var;
    ans=lambda1*spec_mean+lambda2*spec_sdev
    
    if torch.isnan(spec_sdev).any():    print("spec loss with nan",ans)
    
    return ans
    
def loss_hist(hist_sample,hist_ref):
    
    lambda1=1.0
    return lambda1*torch.log(torch.mean(torch.pow(hist_sample-hist_ref,2)))

def f_FM_loss(real_output,fake_output,lambda_fm,gdict):
    '''
    Module to implement Feature-Matching loss. Reads all but last elements of Discriminator ouput
    '''
    FM=torch.Tensor([0.0]).to(gdict['device'])
    for i,j in zip(real_output[:-1],fake_output[:-1]):
#         print(i.shape,j.shape)
        real_mean=torch.mean(i)
        fake_mean=torch.mean(j)
#         print(real_mean,fake_mean)
        FM=FM.clone()+torch.sum(torch.square(real_mean-fake_mean))
    return lambda_fm*FM

def f_gp_loss(grads,l=1.0):
    '''
    Module to implement gradient penalty loss.
    '''
    loss=torch.mean(torch.sum(torch.square(grads),dim=[1,2,3]))
    return l*loss