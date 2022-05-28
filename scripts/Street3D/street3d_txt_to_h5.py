import os
import h5py
import pandas as pd
import numpy as np

source = "../../data/Street3D/txt"
target = "../../data/Street3D/h5"

for set in ['train','test']:
    if not os.path.exists(os.path.join(target,set)):
        os.makedirs(os.path.join(target,set))

for set in ['train','test']:
    sourcedir = os.path.join(source,set)
    targetdir = os.path.join(target,set)
    for file in os.listdir(sourcedir):
        filesource = os.path.join(sourcedir,file)
        filetarget = os.path.join(targetdir,file.split('.')[0]+'.h5')

        df = pd.read_csv(filesource, delimiter=' ', dtype=np.float32)
        data = df.values[:]
        data = data[data[:,-1]!=0]
        data[:,-1]-=1
        hdf5 = h5py.File(filetarget,'w')

        hdf5.create_dataset('data', data=data)
        hdf5.close()
        print(f'saved file {filetarget}')
