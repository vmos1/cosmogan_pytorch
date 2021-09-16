## Code to perform the train-val split and transformation before training for 3D images ## 
### Done separately to avoid memory overload issues
## May 6, 2020. Author:Venkitesh
## June 11, 2020 - Added train-val split code
## Jan 8, 2021 - Code for 3D

import numpy as np
import time
import sys

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
    
if __name__=='__main__':
    
    train_size,val_size=np.int(400),100
#     train_size,val_size=np.int(18e3),3000
    model=1 # Transformation model
    
    t1=time.time()
#     data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/dataset5_3dcgan_4univs_64cube_simple_splicing/'
#     ip_fname=data_dir+'Om0.3_Sg0.8_H70.0.npy'

    data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/temp_data/'
    ip_fname=sys.argv[1]

    print("file",ip_fname)

    ### Read data
    samples = np.load(ip_fname,allow_pickle=True)
    print(samples.shape)
    
    ### Random slice to extract a smaller, required sub-sample
    # Since the lbann code has typical validation ratio of 0.8, we need to grab more samples so that keras and lbann samples match
    select_size=np.int((train_size+val_size)/0.8)+1
    print(select_size)
    
    np.random.seed=27705
#     samples=np.random.choice(samples,size=select_size,replace=False)
    t2=time.time()
    np.random.shuffle(samples)
    t3=time.time()
    print("Time for shuffle",t3-t2)
    samples=samples[:select_size]
#     np.save(data_dir+'train.npy',samples[:train_size]) ## Not saving training data
    np.save(data_dir+'val.npy',samples[train_size:(train_size+val_size)])
    
    ### Transform the images 
    t4=time.time()
    samples=f_scaling_transform(model,samples)
    t5=time.time()
    print("Time for Applying transform",t5-t4)
    print(samples.shape,type(samples[0,0,0,0,0]))
    
    ### Save to output files
    np.save(data_dir+'norm_{0}_train_val.npy'.format(model),samples)
    
    t6=time.time()
    print("Time for saving file",t6-t5)
