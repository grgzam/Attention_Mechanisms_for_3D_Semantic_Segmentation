![](imgs/SemanticKITTI_viz.jpg)
![](imgs/Street3D_viz.jpg)

# Improving performance of deep learning models for 3D point cloud semantic segmentation via attention mechanisms
This is the official implementation of "Improving performance of deep learning models for 3D point cloud semantic segmentation via attention mechanisms" paper

Our implemented network "SPVCNN with Point Transformer in the voxel branch", achieves State Of the Art Results in Street3D dataset

## Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 18.04)
- Python 3.9.7
- PyTorch 1.10
- CUDA 11.4



## Install 

1. Construct an anaconda environment with python 3.9.7
2. Install pytorch 1.10 `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge`
3. Install [torchsparse](https://github.com/mit-han-lab/torchsparse) with `pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0`
4. For the k-NN, we use the operations as implemented in [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer). Execute the lib\pointops\setup.py file, downloaded from [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer),  with `python3.9 setup.py install` 
5. Install [h5py](https://docs.h5py.org/en/latest/build.html) with `conda install h5py`
6. Install tqdm with `pip install tqdm`
7. Install ignite with `pip install pytorch-ignite`
8. Install numba with `pip install numba`

## Supported Datasets
- [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download)
- [Street3D](https://kutao207.github.io/shrec2020)

### SemanticKITTI


