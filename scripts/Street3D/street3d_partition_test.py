import random
from typing import Any, Dict
from tqdm import tqdm
import numpy as np
import torch.utils.data
import os
import h5py
from torchsparse.utils.collate import sparse_collate_fn



random.seed(2341)
np.random.seed(2341)
torch.manual_seed(2341)
torch.cuda.manual_seed_all(2341)




class shrec_dataset:

    def __init__(self,voxel_size,split=None) :
        self.data = []

        spath = '../../data/Street3D/h5/test'

        for file in os.listdir(spath):
          path = os.path.join(spath,file)
          hdf5 = h5py.File(path,'r')
          self.data.append(path)
        self.voxel_size = voxel_size

    def __getitem__(self, item) -> Dict[str, Any]:

        item = item % len(self.data)
        path = self.data[item]
        hdf5 = h5py.File(path,'r')
        inputs = hdf5['data'][:]
        print(inputs.shape)
        return inputs,path


    def __len__(self):
        return len(self.data)


r = 0.05
bs=1
nw=1
ne=1

## Set the number of points that each subscene will have
number_of_point = 80000


split = 'test'
dataset = shrec_dataset(voxel_size=r,split=split)

dataflow = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=1,
    collate_fn=sparse_collate_fn,
    num_workers=1
)

spath = '../../data/Street3D/h5/test_part_80k'
os.makedirs(spath)

for idx, data in enumerate(tqdm(dataflow)):

  inputs = data[0][0]
  path = data[0][1]
  indx = np.random.choice(inputs.shape[0],inputs.shape[0],replace=False)

  inputs = inputs[indx]
  indexes = np.arange(len(inputs))


  splitted = np.array_split(indexes,indexes.shape[0]//number_of_point)
  for idx,each in enumerate(splitted):


    savepath = os.path.join(spath,path.split('.')[-2].split('/')[-1]+str(idx)+'.bin')
    inputs[each].tofile(savepath)
    print(inputs[each].shape,savepath)

