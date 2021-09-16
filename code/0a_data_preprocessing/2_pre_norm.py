## Code to perform the train-val split and transformation before training ## 
### Done separately to avoid memory overload issues
## May 6, 2020. Author:Venkitesh
## June 11, 2020 - Added train-val split code


import numpy as np

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
    
    train_size,val_size=np.int(8e4),1000
#     train_size,val_size=np.int(18e3),3000
    model=1 # Transformation model
    
#     data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_2_smoothing_200k/'  
#     data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/512_square/dataset1_smoothing_single_universe/'
    ip_fname=data_dir+'full_with_smoothing_1.npy'
    
#     data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/128_square/dataset_5_4univ_cgan/'
#     ip_fname=data_dir+'Om0.3_Sg1.1_H70.0.npy'

#     data_dir='/global/cfs/cdirs/m3363/vayyar/cosmogan_data/raw_data/3d_data/'
#     ip_fname=data_dir+'full_with_smoothing_1.npy'

    print("file",ip_fname)
    ### Read data
    samples = np.load(ip_fname, allow_pickle=True)
    print(samples.shape)
    
    ### Random slice to extract a smaller, required sub-sample
    # Since the lbann code has typical validation ratio of 0.8, we need to grab more samples so that keras and lbann samples match
    select_size=np.int((train_size+val_size)/0.8)+1
    print(select_size)
    
    np.random.seed=27705
#     samples=np.random.choice(samples,size=select_size,replace=False)
    np.random.shuffle(samples)
    samples=samples[:select_size]
#     np.save(data_dir+'train.npy',samples[:train_size])  ## Not saving training data
    np.save(data_dir+'val.npy',samples[train_size:(train_size+val_size)])  ##
    
    ### Transform the images 
    samples=f_scaling_transform(model,samples)
    print(samples.shape,type(samples[0,0,0,0]))
    
    ### Save to output files
    np.save(data_dir+'norm_{0}_train_val.npy'.format(model),samples)
