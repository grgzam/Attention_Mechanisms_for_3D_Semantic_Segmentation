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
from ignite.metrics.confusion_matrix import ConfusionMatrix


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


## Load the proper weights for each model
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_se_p/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_se_v/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_cbam_p/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_cbam_v/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_knnse_p/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_knnse_v/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_knncbam_p/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_knncbam_v/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_pt_p/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_pt_v/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_lfa_p/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/spvcnn_lfa_v/checkpoint.tar')


checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko_se/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko_cbam/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko_knnse/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko_knncbam/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko_pt/checkpoint.tar')
#checkpoint = torch.load('../../pretrained_weights/SemanticKITTI/minko_lfa/checkpoint.tar')

##----------------------------------------------------------------------------------
##-----------------------------------------------------------------------------




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
        if source is None:
            source = r'../SemanticKITTI_dataset/sequences'
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

        binpath = self.filebin[item]
        labpath = self.filelabel[item]
        xyz,labels = self.read_lab_bin(binpath,labpath)
        all_labs = labels

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
torch.set_printoptions(sci_mode=False)

source = '../../data/SemanticKITTI/sequences'


validate = ['08']
print("test set is ",validate)
model = SPVCNN(num_classes=19,cr = 1, pres=1,vres=1)
model.cuda()

# Initial voxel size (0.05m X 0.05m X 0.05m)
r=0.05

# Batch size (keep this to 1 for inference)
bs = 1

nw = 1
#ne = 15

dsval = kitti(validate,source,train=False,voxel_size=r)
dlval = DataLoader(dsval,batch_size=1,
                   collate_fn = sparse_collate_fn,
                   num_workers=nw)

bestmiou = 0
bestoa = 0

metricoa = metric('overall',num_classes=19)
metricmiou = metric(num_classes=19)
cmatrix = ConfusionMatrix(19,'recall')

model.load_state_dict(checkpoint['model_state'],strict=False)

model.eval()
for batch in tqdm(dlval):
  pts = batch['points']
  labels = batch['labels']
  all_labs = batch['all_labs'].cuda()
  inverse = batch['inverse']
  pts = pts.cuda()
  output = model(pts)
  output = output[inverse]
  metricoa.update(output[0],all_labs[0])
  metricmiou.update(output[0],all_labs[0])
  cmatrix.update((output[0].view(-1,19),all_labs[0].view(-1).int().cuda()))
oa = metricoa.compute()
miou = metricmiou.compute(True,class_name_mapping)
print(f'oa is {oa:.4f} and miou is {miou:.4f}')
metricoa.reset()
metricmiou.reset()
print ("\n Confusion matrix is:")
print(cmatrix.compute())

print("params are:", sum(p.numel() for p in model.parameters() if p.requires_grad))

