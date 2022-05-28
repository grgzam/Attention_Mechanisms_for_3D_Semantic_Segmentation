import random
import numpy as np
import os
from torch.utils.data import DataLoader , Dataset
import torch
from tqdm import tqdm
from meters.s3dis import MeterS3DIS as metric
import torch.nn as nn
from torchsparse.tensor import *

from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate_fn

from functools import partial
from core.schedulers import cosine_schedule_with_warmup


## Uncomment to load the proper baseline model
#from core.models import SPVCNN
from core.models import MinkUNet as SPVCNN

## or the proper attention enhanced baseline model (only one model should be used at a time)
#from models_semantic_kitti.spvcnn_se_point_global import SPVCNN
#from models_semantic_kitti.spvcnn_se_voxel_global import SPVCNN
#from models_semantic_kitti.spvcnn_cbam_point_global import SPVCNN
#from models_semantic_kitti.spvcnn_cbam_voxel_global import SPVCNN
#from models_semantic_kitti.spvcnn_knn_se_point import SPVCNN
#from models_semantic_kitti.spvcnn_knn_se_voxel import SPVCNN
#from models_semantic_kitti.spvcnn_knn_cbam_point import SPVCNN
#from models_semantic_kitti.spvcnn_knn_cbam_voxel import SPVCNN
#from models_semantic_kitti.spvcnn_pt_point import SPVCNN
#from models_semantic_kitti.spvcnn_pt_voxel import SPVCNN
#from models_semantic_kitti.spvcnn_lfa_point import SPVCNN
#from models_semantic_kitti.spvcnn_lfa_voxel import SPVCNN

#from models_semantic_kitti.minko_se_global import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_cbam_global import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_knn_se import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_knn_se_fast import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_knn_cbam import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_knn_cbam_fast import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_pt import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_pt_fast import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_lfa import MinkUNet as SPVCNN
#from models_semantic_kitti.minko_lfa_fast import MinkUNet as SPVCNN

## Use the 'fast' implementations only for inference, after the network is trained

## Save the path according to the desired location (change only the model name accordingly)
model_name = 'minko'
savepath = 'SemanticKITTI_results/'+model_name

## If load_checkpoint = True, then it continues training from latest ckpt
load_checkpoint=True

## Define the proper training parameters
## r is the voxel size, bs is the batch size,
## nw is the number of workers, ne is number of epochs
r=0.05
bs = 2
nw = 1
ne = 15

## Learning rate is set to optimizer below

##-----------------------------------------------------------------------------

def calculate_weights(labels):
  counts = np.zeros(19)
  labels = labels.cpu().numpy()
  for i in range(19):
    counts[i]+=np.sum(labels==i)
  frq = counts/np.sum(counts)
  return 1/(frq+0.00001)**0.5


random.seed(2341)
np.random.seed(2341)
torch.manual_seed(2341)
torch.cuda.manual_seed_all(2341)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



checkpoint=None
if not os.path.exists(savepath):
    os.makedirs(savepath)
else:
    if not load_checkpoint:
        print(f"save path {savepath} exists change the the path and try again")
        exit(0)
    else:
        print('continuing from last checkpoint')
        if os.path.isfile(savepath+f'/checkpoint.tar'):
             checkpoint = torch.load(savepath+f'/checkpoint.tar')


mappinglist = np.zeros(260)
mappingdict = {
    0 : 0,
    1 : 0,
    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 9,
    70: 15,
    71: 16,
    72: 17,
    80: 18,
    81: 19,
    99: 0,
    252: 1,
    253: 7,
    254: 6,
    255: 8,
    256: 5,
    257: 5,
    258: 4,
    259: 5,
}

class_name_mapping ={
    0:  "car",
    1:  "bicycle",
    2:  "motorcycle",
    3:  "truck",
    4:  "other-vehicle",
    5:  "person",
    6:  "bicyclist",
    7:  "motorcyclist",
    8:  "road",
    9:  "parking",
    10: "sidewalk",
    11: "other-ground",
    12: "building",
    13: "fence",
    14: "vegetation",
    15: "trunk",
    16: "terrain",
    17: "pole",
    18: "traffic-sign"
}

for k in mappingdict.keys():
    mappinglist[k] = mappingdict[k]


class kitti(Dataset):
    def __init__(self,train_sequences,source=None,train=True,voxel_size=0.05):
        self.voxel_size = voxel_size
        self.train = train
        self.counts = None
        self.iter = 0
        self.source = source

        ## Set the location of dataset folder
        if source is None:
            source = r'../dataset/sequences'
        self.train_seq = train_sequences
        self.filebin = []
        self.filelabel = []
        for each in self.train_seq:
            velodyne = os.path.join(source,each,'velodyne')
            labels = os.path.join(source,each,'labels')
            for file,filel in zip(sorted(os.listdir(velodyne)),sorted(os.listdir(labels))):
                filepath = os.path.join(velodyne,file)
                filepathlabel = os.path.join(labels,filel)
                self.filebin.append(filepath)
                self.filelabel.append(filepathlabel)
    def __getitem__(self, item):
        N=80000
        binpath = self.filebin[item]
        labpath = self.filelabel[item]
        xyz,labels = self.read_lab_bin(binpath,labpath)
        all_labs = labels
        if self.train:
            flag = N>len(labels)
            choices = np.random.choice(len(labels),N,flag)
            xyz = xyz[choices]
            labels = labels[choices]

        if self.train:
             theta = np.random.uniform(0,2*np.pi)
             scale = np.random.uniform(0.95,1.05)
             rot_mat = np.array([[np.cos(theta),np.sin(theta),0],
                                 [-np.sin(theta), np.cos(theta), 0],
                                 [0,0,1]])
             xyz[:,:3] = np.dot(xyz[:,:3],rot_mat)*scale

        coords, feats = xyz[:,:3], xyz.__deepcopy__(xyz)
        coords -= np.min(coords, 0)
        coords, indices, inverse = sparse_quantize(coords,
                                          self.voxel_size,
                                          return_index=True,
                                          return_inverse=True)
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(feats[indices], dtype=torch.float)
        labels = torch.from_numpy(labels[indices]).long()
        input = SparseTensor(coords=coords, feats=feats)
        label = SparseTensor(coords=coords, feats=labels)
        if self.train:
           return {'points': input,
                   'labels': label}
        else:
           return {'points': input,
                   'labels': labels,
                   'all_labs': all_labs,
                   'inverse': inverse }
    def __len__(self):
        return len(self.filelabel)
    def read_lab_bin(self,binpath,labpath):
        labels = np.fromfile(labpath, np.uint16).reshape(-1,2)[:,0]
        labels = mappinglist[labels].astype(np.int32)
        xyz = np.fromfile(binpath, np.float32).reshape(-1,4)
        xyz = xyz[np.argwhere(labels).squeeze(1)]

        labels = labels[np.argwhere(labels).squeeze(1)] - 1
        return xyz ,labels


with_save = True
np.set_printoptions(suppress=True)
source = '../../data/SemanticKITTI/sequences'

train = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
#validate = ['08']

print("train set is ",train)
#print("test set is ",validate)


ftrain = open(savepath+"/train.txt",'w')
#fval = open(savepath+"/validate.txt",'w')


model = SPVCNN(num_classes=19,cr = 1, pres=1,vres=1)

if (checkpoint is not None) and load_checkpoint:
    model.load_state_dict(checkpoint['model_state'])


model.cuda()


criterion = nn.CrossEntropyLoss()
initial_lr = 0.096
optimizer = torch.optim.SGD(
          model.parameters(),
          momentum=0.9,
          nesterov=True,
          weight_decay=1.0e-4,
          lr=initial_lr)



ds = kitti(train,source,voxel_size=r)
dl = DataLoader(ds,batch_size=bs,
                shuffle=True,
                collate_fn = sparse_collate_fn,
                num_workers=nw)

scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=ne,
                              batch_size=bs,
                              dataset_size=len(ds)))

last_epoch = 0
bestmiou = 0
bestoa = 0

if (checkpoint is not None) and load_checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    last_epoch = checkpoint['last_epoch']+1
    bestmiou = checkpoint['bestmiou']
    bestoa = checkpoint['bestoa']

metricoa = metric('overall',num_classes=19)
metricmiou = metric(num_classes=19)


for epoch in range(last_epoch,ne,1):
    model.train()
    for batch in tqdm(dl):
        pts = batch['points']
        labels = batch['labels']
        labs = labels.F.cpu().numpy()

        pts = pts.cuda()
        labels = labels.cuda()
        output = model(pts)
        criterion.weight = torch.from_numpy(calculate_weights(labels.feats)).cuda().float()
        loss = criterion(output,labels.F)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        metricoa.update(output,labels.F)
        metricmiou.update(output,labels.F)
    oa = metricoa.compute()
    miou = metricmiou.compute(False)
    print(f"{epoch} {oa} {miou}",file=ftrain)
    print(f'training: epoch is {epoch} oa is {oa:.4f} and miou is {miou:.4f}')
    metricoa.reset()
    metricmiou.reset()
    torch.save({
                  'last_epoch':epoch,
                  'model_state':model.state_dict(),
                  'optimizer_state':optimizer.state_dict(),
                  'scheduler_state':scheduler.state_dict(),
                  'bestmiou':bestmiou,
                  'bestoa':bestoa,
                 },
                 savepath+f'/checkpoint.tar')
    torch.save(model.state_dict(),savepath+f'/kitti.pth')

