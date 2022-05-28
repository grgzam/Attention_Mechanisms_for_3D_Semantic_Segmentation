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
- Please follow the instructions from [here](http://www.semantic-kitti.org/dataset.html#download) to download the SemanticKITTI dataset (both KITTI Odometry dataset and SemanticKITTI labels) and extract all the files in the sequences folder to `data/SemanticKITTI`. You should see 22 folders. Folders 00-10 should have subfolders named `velodyne` and `labels`. The rest 11-21 folders are used for online testing and should not contain any `labels` folder, only the `velodyne` folder.

### Street3D
- Plese follow the instructions from [here](https://kutao207.github.io/shrec2020) to download the Street3D dataset. It is in a `.txt` form. Place it in the `data/Street3D/txt` folder, where you should have two folders, `train` and `test` with 60 and 20 `.txt` files, respectively.
- Next, execute the pre-processing scripts as follows:
 ```
 python scripts/Streed3D/street3d_txt_to_h5.py
 python scripts/Streed3D/street3d_partition_train.py
 python scripts/Streed3D/street3d_partition_test.py
 
 ```
 The first script converts the dataset to h5 format and places it in the `data/Street3D/h5` folder
 The following scripts split each scene into subscenes of around 80k points and save them in `.bin` format into proper folders, `train_part_80k` and `test_part_80k` sets respectively. The `train_part_80k` folder should contain 2458 files and the `test_part_80k` folder should contain 845 files. Training and testing is performed based on these split subscenes of 80k points. 
 
 The final structure for both datasets should look like this:
 
 data/
 - SemanticKITTI/
   - sequences/
     - 00/
       - poses.txt
       - labels/
         - 000000.label
         - ...
         
       - velodyne/
         - 000000.bin
         - ...
         
     - ...
     
     - 21/
       - poses.txt
       - velodyne/
         - 000000.bin
         - ...
       
       
       
 - Street3D/
   - txt/
     - train/
       - 5D4KVPBP.txt
       - ...
     - test/
       - 5D4KVPG4.txt
       - ...
    
   - h5/
     - train/
       - 5D4KVPBP.h5
       - ...
       
     - test/
       - 5D4KVPG4.h5
       - ...
       
     - train_part_80k/
       - 5D4KVPBP0.bin
       - ...
       
     - test_part_80k/
       - 5D4KVPG40.bin
       - ...
       
       
## Training

To train the networks, check the following scripts for each dataset:
```
python scripts/SemanticKITTI/kitti_train_all.py
python scripts/Street3D/street3d_train_all.py
```

Inside each file, you can select the proper network to train, as well as training parameters.


## Inference

To test the networks in SemanticKITTI validation set or Street3D test set, check the following scripts for each dataset:
```
python scripts/SemanticKITTI/kitti_inference_all.py
python scripts/Street3D/street3d_inference_all.py
```
Inside each file, you can select the proper network to inference, as well as to load the proper weights.


## Pretrained weights

The pretrained weights used in our paper are provided [here](https://vc.ee.duth.gr:6960/index.php/s/iWh3sHqIwyfTAQ2)
The size is around 4.8 GB for the weights for all networks

Next, unzip the `pretrained_weights.zip` file in the main folder of the repository



