import torch
import pointops_cuda
import torchsparse
import torchsparse.nn as spnn
from torch import nn
from torchsparse import PointTensor, SparseTensor

from core.models.utils import initial_voxelize, point_to_voxel, voxel_to_point


__all__ = ['SPVCNN']


class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


def offs(x):
    offset = []
    offset.append((x.C[:,-1]==0).sum().item())
    offset.append(x.C.shape[0])
    offset = torch.cuda.IntTensor(offset)
    return offset

def knn(x,k=16):
    pos = x.C[:,:3].float()
    offset = offs(x)
    m = x.F.shape[0]
    idx = torch.cuda.IntTensor(m,k).zero_()
    dist = torch.cuda.FloatTensor(m,k).zero_()
    pointops_cuda.knnquery_cuda(m,k,pos,pos,offset,offset,idx,dist)
    idx = idx.long()
    return idx, dist

class LSE_AP(nn.Module):
    def __init__(self,inc,outc,k=16):
        super().__init__()
        self.k = k
        self.posmlp = nn.Sequential(nn.Linear(10,inc),
                                 nn.BatchNorm1d(inc),
                                 nn.ReLU(True))
        self.attmlp = nn.Sequential(nn.Linear(2*inc,2*inc),
                                    nn.BatchNorm1d(2*inc),
                                    nn.ReLU(True))
        self.mlpout = nn.Sequential(nn.Linear(2*inc,outc),
                                    nn.BatchNorm1d(outc),
                                    nn.ReLU(True))

        self.softmax = nn.Softmax(dim=1)

    def forward(self,x,pos,idx,dist):

        xyzk = pos[:,None,:3].tile(1,self.k,1).float()

        posenc = torch.dstack([dist[:,:,None],xyzk-pos[:,:3].float()[idx],xyzk,pos[:,:3].float()[idx]])
        for i, layer in enumerate(self.posmlp):
           posenc = layer(posenc.transpose(1,2)).transpose(1,2) if (i==1) else layer(posenc)
        fxyz = posenc
        ffeats = x[idx]
        stackfeat = torch.dstack([fxyz,ffeats])

        scores = self.attmlp[0](stackfeat)

        for i, layer in enumerate(self.attmlp[1:]):
            scores = layer(scores.transpose(1,2)).transpose(1,2) if (i==0) else layer(scores)
        scores = self.softmax(scores)
        stackfeat = (stackfeat*scores).sum(1)
        out = self.mlpout(stackfeat)
        return out

class LFA(nn.Module):

    def __init__(self, inc, outc, k =16):
        super().__init__()
        self.k = k
        mid = int(outc/4)
        self.lseap1 = LSE_AP(mid,outc)
        self.in_lin = nn.Linear(inc,mid,bias=False)
        self.lrelu = nn.LeakyReLU(0.1)
        self.skip = nn.Linear(inc,outc,bias=False)
    def forward(self,x,idx=None,dist=None):


        idx_flag = False
        if idx==None:
           idx_flag = True
           idx,dist = knn(x)

        x1 = self.in_lin(x.F)
        x1 = self.lseap1(x1,x.C,idx,dist)

        out = self.lrelu(self.skip(x.F)+x1)
        if idx_flag:
            return out,idx,dist

        return out


class SPVCNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']
        self.inc = 4
        if 'inc' in kwargs:
            self.inc = kwargs['inc']

        self.stem = nn.Sequential(
            spnn.Conv3d(self.inc, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.lfastem = LFA(cs[0],cs[0])
        self.lfa1 = LFA(cs[1],cs[1])
        self.lfa2 = LFA(cs[2],cs[2])
        self.lfa3 = LFA(cs[3],cs[3])
        self.lfa4 = LFA(cs[4],cs[4])
        self.lfau1 = LFA(cs[5],cs[5])
        self.lfau2 = LFA(cs[6],cs[6])
        self.lfau3 = LFA(cs[7],cs[7])
        self.lfau4 = LFA(cs[8],cs[8])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, self.pres, self.vres)

        x0 = self.stem(x0)
        x0.F, idx0, dist0 = self.lfastem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x1.F,idx1,dist1 = self.lfa1(x1)
        x2 = self.stage2(x1)
        x2.F,idx2,dist2 = self.lfa2(x2)
        x3 = self.stage3(x2)
        x3.F,idx3,dist3 = self.lfa3(x3)
        x4 = self.stage4(x3)
        x4.F,_,__ = self.lfa4(x4)
        z1 = voxel_to_point(x4, z0)

        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)
        y1.F = self.lfau1(y1,idx3,dist3)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        y2.F = self.lfau2(y2,idx2,dist2)
        z2 = voxel_to_point(y2, z1)

        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)
        y3.F = self.lfau3(y3,idx1,dist1)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        y4.F = self.lfau4(y4,idx0,dist0)
        z3 = voxel_to_point(y4, z2)

        z3.F = z3.F + self.point_transforms[2](z2.F)

        out = self.classifier(z3.F)
        return out
