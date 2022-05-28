import torch
import pointops_cuda
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import PointTensor
from core.models.utils import voxel_to_point, initial_voxelize

__all__ = ['MinkUNet']


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
            self.downsample = nn.Sequential()
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
    return idx

class SE(nn.Module):
    def __init__(self,inc,reduction=4):
        super().__init__()
        hidden = int(inc/reduction)
        self.MLP = nn.Sequential(nn.Linear(inc,hidden),
                                 nn.ReLU(True),
                                 nn.Linear(hidden,inc),
                                 nn.Sigmoid())


    def forward(self,x,idx=None):

        idx_flag = False
        if idx==None:
           idx_flag = True
           idx = knn(x)
        n = x.F.shape[0]
        avg = x.F[idx].mean(1).view(n,-1)
        mlpout = self.MLP(avg)
        out = x.F*mlpout
        if idx_flag:
           return out,idx
        return out

class MinkUNet(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

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
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

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

        self.sestem = SE(cs[0])
        self.se1 = SE(cs[1])
        self.se2 = SE(cs[2])
        self.se3 = SE(cs[3])
        self.se4 = SE(cs[4])
        self.seu1 = SE(cs[5])
        self.seu2 = SE(cs[6])
        self.seu3 = SE(cs[7])
        self.seu4 = SE(cs[8])


        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))


        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = PointTensor(x.F, x.C.float())

        x0 = initial_voxelize(z, 1,1)

        x0 = self.stem(x0)
        x0.F,idx0 =self.sestem(x0)
        x1 = self.stage1(x0)
        x1.F,idx1 = self.se1(x1)
        x2 = self.stage2(x1)
        x2.F,idx2 = self.se2(x2)
        x3 = self.stage3(x2)
        x3.F,idx3 = self.se3(x3)
        x4 = self.stage4(x3)
        x4.F,idx4 = self.se4(x4)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)
        y1.F = self.seu1(y1,idx3)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        y2.F = self.seu2(y2,idx2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)
        y3.F = self.seu3(y3,idx1)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        y4.F = self.seu4(y4,idx0)
        z1 = voxel_to_point(y4,z)

        out = self.classifier(z1.F)



        return out
