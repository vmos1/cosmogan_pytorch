### Peter Harrington's original code
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

Ddir = '.'
out_basename = './raw'
files = os.listdir(Ddir)

slices = []
img_dim = 128
perside = 512//img_dim
for fname in files:
    if '.hdf5' not in fname:
        continue
    with h5py.File(os.path.join(Ddir,fname), 'r') as inputdata:
        for ix in range(perside):
            for iy in range(perside):
                # Select every 8 slices along some axis, for redshift=0
                data = inputdata['full'][::8, ix*img_dim:(ix+1)*img_dim, iy*img_dim:(iy+1)*img_dim, 0]
                np.random.shuffle(data)
                slices.append(np.expand_dims(data, axis=-1))
    print('Sliced %s'%fname)

slices = np.concatenate(slices)
train = slices[:17000]
val = slices[17000:]

train_name = out_basename+'_train.npy'
print('Saving file %s'%train_name)
print('shape='+str(train.shape))
np.save(train_name, train)
val_name = out_basename+'_val.npy'
print('Saving file %s'%val_name)
print('shape='+str(val.shape))
np.save(val_name, val)

