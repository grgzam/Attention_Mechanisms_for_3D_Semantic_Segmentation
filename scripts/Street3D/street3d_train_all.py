import time
import random
from typing import Any, Dict
from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch import nn
#from torch.cuda import amp
import os
import h5py

from meters.s3dis import MeterS3DIS as metric
from torchsparse import SparseTensor
from torchsparse import nn as spnn
from torchsparse.utils.collate import sparse_collate_fn
from torchsparse.utils.quantize import sparse_quantize
from functools import partial
from core.schedulers import cosine_schedule_with_warmup


## Uncomment to load the proper baseline model
from core.models import SPVCNN
#from core.models import MinkUNet as SPVCNN

## or the proper attention enhanced baseline model (only one model should be used at a time)
#from models_street3d.spvcnn_se_point_global import SPVCNN
#from models_street3d.spvcnn_se_voxel_global import SPVCNN
#from models_street3d.spvcnn_cbam_point_global import SPVCNN
#from models_street3d.spvcnn_cbam_voxel_global import SPVCNN
#from models_street3d.spvcnn_knn_se_point import SPVCNN
#from models_street3d.spvcnn_knn_se_voxel import SPVCNN
#from models_street3d.spvcnn_knn_cbam_point import SPVCNN
#from models_street3d.spvcnn_knn_cbam_voxel import SPVCNN
#from models_street3d.spvcnn_pt_point import SPVCNN
#from models_street3d.spvcnn_pt_voxel import SPVCNN
#from models_street3d.spvcnn_lfa_point import SPVCNN
#from models_street3d.spvcnn_lfa_voxel import SPVCNN

#from models_street3d.minko_se_global import MinkUNet as SPVCNN
#from models_street3d.minko_cbam_global import MinkUNet as SPVCNN
#from models_street3d.minko_knn_se import MinkUNet as SPVCNN
#from models_street3d.minko_knn_se_fast import MinkUNet as SPVCNN
#from models_street3d.minko_knn_cbam import MinkUNet as SPVCNN
#from models_street3d.minko_knn_cbam_fast import MinkUNet as SPVCNN
#from models_street3d.minko_pt import MinkUNet as SPVCNN
#from models_street3d.minko_pt_fast import MinkUNet as SPVCNN
#from models_street3d.minko_lfa import MinkUNet as SPVCNN
#from models_street3d.minko_lfa_fast import MinkUNet as SPVCNN
## Use the 'fast' implementations only for inference, after the network is trained

## Define the proper savepath
model_name = "spvcnn"
savepath = "Street3D_results/"+model_name

## Define the proper training parameters
## Voxel size in meters (0.05 equals to a voxel of 0.05 X 0.05 X 0.05)
r = 0.05

## Batch Size
bs = 2

## Number of Epochs
ne = 15

## Learning rate is set to optimizer below

##-----------------------------------------------------------------------------


def calculate_weights(labels):
  counts = np.zeros(5)
  labels = labels.cpu().numpy()
  for i in range(5):
    counts[i]+=np.sum(labels==i)
  frq = counts/np.sum(counts)
  return 1/(frq+0.00001)**0.5



random.seed(2341)
np.random.seed(2341)
torch.manual_seed(2341)
torch.cuda.manual_seed_all(2341)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    print(f"the folder {savepath} exists change the savepath and try again")
    exit(0)

class shrec_dataset:

    def __init__(self,voxel_size,split=None) :
        self.data = []
        self.train=True
        self.voxel_size = voxel_size
        self.split = split

        spath = '../../data/Street3D/h5/train_part_80k'
        for file in os.listdir(spath):
            path = os.path.join(spath,file)
            self.data.append(path)



    def __getitem__(self, item) -> Dict[str, Any]:

        item = item % len(self.data)
        path = self.data[item]
        inputs = np.fromfile(path,np.float32).reshape(-1,4)
        n = 80000



        if len(inputs)>n and self.train and not self.split=='train_part':
             choices = np.random.choice(len(inputs),n,False)
             inputs = inputs[choices]

        if self.train:
            theta = np.random.uniform(0,2*np.pi)
            scale = np.random.uniform(0.95,1.05)
            rot_mat = np.array([[np.cos(theta),np.sin(theta),0],
                                [-np.sin(theta), np.cos(theta), 0],
                                [0,0,1]])
            inputs[:,:3] = np.dot(inputs[:,:3],rot_mat)*scale



        minpc = np.min(inputs[:,:3], axis=0, keepdims=True)
        inputs[:,:3] -= np.min(inputs[:,:3], axis=0, keepdims=True)

        coords, feats = inputs[:,:3], inputs[:,:3]
        all_labels = inputs[:,-1]

        coords, indices, inverse_mapping = sparse_quantize(coords,
                                          self.voxel_size,
                                          return_index=True,
                                          return_inverse=True)

        feats = feats[indices]
        labels = inputs[indices,-1]

        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats, dtype=torch.float)

        labels = torch.from_numpy(labels).long()

        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)

        if self.train:
           return {'input': input,
                  'label': label}
        else:
           return {'input': input,
                   'label': label,
                   'all_labs': all_labels,
                   'inverse': inverse_mapping}

    def __len__(self):

        if self.train and not self.split=='train_part':
            return 10*len(self.data)
        return len(self.data)
    def calculate_stats(self):
        return self.stats


source = '../../data/Street3D/h5/train_part_80k'

files = os.listdir(source)
random.shuffle(files)
sourcetrain = files


ftrain = open(savepath+"/train.txt",'w')

## Set which dataset to use
dataset = shrec_dataset(voxel_size=r,split='train_part')

metricmiou = metric(num_classes=5)
metricoa = metric(metric='overall',num_classes=5)

dataflow = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=bs,
    collate_fn=sparse_collate_fn,
    num_workers=1,
#    persistent_workers=True
)



model = SPVCNN(num_classes=5, cr=1,pres=1,vres=1,inc=3)

model.cuda()

criterion = nn.CrossEntropyLoss()
bestmiou=-1
bestoa=0

optimizer = torch.optim.SGD(model.parameters(),momentum=0.9,nesterov=True,weight_decay=1.0e-4,lr=0.024)



scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=ne,
                              batch_size=bs,
                              dataset_size=len(dataset)))

for i in range(ne):
  model.train()
  for idx, feed_dict in enumerate(tqdm(dataflow)):

      inputs = feed_dict['input'].to('cuda')
      labels = feed_dict['label'].to('cuda')
      criterion.weight = torch.from_numpy(calculate_weights(labels.feats)).cuda().float()
      outputs = model(inputs)
      loss = criterion(outputs, labels.feats)
      metricmiou.update(outputs,labels.feats)
      metricoa.update(outputs,labels.feats)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      scheduler.step()

  miou = metricmiou.compute(True)
  oa = metricoa.compute()
  print(f"epoch is {i} oa is {oa} and miou is {miou}")
  print(f"{i} {oa} {miou}",file=ftrain)
  metricmiou.reset()
  metricoa.reset()
  torch.save(model.state_dict(),savepath+'/shrec.pth')
