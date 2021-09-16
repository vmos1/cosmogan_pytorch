## Code to perform the train-val split and transformation before training for 3D images ## 
### Done separately to avoid memory overload issues
## May 6, 2020. Author:Venkitesh
## June 11, 2020 - Added train-val split code
## Jan 8, 2021 - Code for 3D

import numpy as np
import time
import sys
import os

def f_scaling_transform(model,samples):
    ''' Read in the transformation function and samples array
    Perform transformation and return samples array
    '''
    
    if model==1:
        def f_transform(x,scale=4.0):
            return np.divide(2.*x, x + scale) - 1.
        
        def f_invtransform(s,scale=4.0):
            return scale*np.divide(1. + s, 1. - s)        
    
        return f_transform(samples)
    
    elif model==2: ### log-linear transformation
        def f_transform(x):
            if x<=50:
                a=0.03; b=-1.0
                return a*x+b
            elif x>50: 
                a=0.5/(np.log(15000)-np.log(50))
                b=0.5-a*np.log(50)
                return a*np.log(x)+b

        def f_invtransform(y):
            if y<=0.5:
                a=0.03;b=-1.0
                return (y-b)/a
            elif y>0.5: 
                a=0.5/(np.log(15000)-np.log(50))
                b=0.5-a*np.log(50)
                return np.exp((y-b)/a)        
        
        return np.vectorize(f_transform)(samples)
    
    
def f_concat_temp_files(num_batches,save_location,file_prefix):
    '''
    Function to concatenate temp files to create the full file.
    Steps: get data from temp files, stack numpy arrays and delete temp files
    '''
    if num_batches<1:
        print('zero temp files',num_batches)
        return 0
    
    x = np.vstack([np.load(save_location+'temp_data_%s_%s'%(file_prefix,count)+'.npy') for count in np.arange(num_batches)])
    
    # Delete temp files
    for count in np.arange(num_batches):
        prefix='temp_data_%s_%s'%(file_prefix,count)
        f1=prefix+'.npy'
        os.remove(save_location+f1)
    print("Deleted temp files")
    
    return x


if __name__=='__main__':

    model=1 # Transformation model
    t1=time.time()
    
    ip_fname=sys.argv[1]
    print("file",ip_fname)
    
    output_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/temp_data/'
    file_prefix='norm'
    
    ### Read data
    samples = np.load(ip_fname,mmap_mode='r',allow_pickle=True).copy()
    data_size=samples.shape[0]
    print(data_size)
    train_size,val_size=int(0.9*data_size),int(0.1*data_size)
    
    num_per_batch= 1000 # (total file size/ 100GB) * data_size
    num_batches=int(data_size/num_per_batch)+1
    print("Num batches",num_batches)
    np.random.seed=27705
    
    for count in range(num_batches):
#         print(count*num_per_batch,(count+1)*num_per_batch)
        arr=samples[count*num_per_batch:(count+1)*num_per_batch]
        ## Shuffle data
        t2=time.time()
        np.random.shuffle(arr)
        t3=time.time()
        print("Time for shuffle",t3-t2)
    #     np.save(data_dir+'train.npy',samples[:train_size]) ## Not saving training data
    #     np.save(data_dir+'val.npy',samples[train_size:(train_size+val_size)])

        ### Transform the images 
        t4=time.time()
        arr2=f_scaling_transform(model,arr)
        t5=time.time()
        print("Time for Applying transform",t5-t4)
        print(arr2.shape,arr.shape)
        prefix='temp_data_{0}_{1}'.format(file_prefix,count)
        np.save(output_dir+prefix+'.npy',arr2)
        t4=time.time()
        print("Total extraction time for count ",count,":",t4-t3)
    
    del samples
    ### Concatenate temp files
    t4=time.time()
    img=f_concat_temp_files(num_batches,save_location=output_dir,file_prefix=file_prefix)
    t5=time.time()
    print("Time for concatenation of file:",t5-t4)
    print("Total number of images",img.shape)
    
    ### Save to output files
    np.save(output_dir+'norm_{0}_train_val.npy'.format(model),img)
    
    t6=time.time()
    print("Time for saving file",t6-t5)
    print("Total time",t6-t1)