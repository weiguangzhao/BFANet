# BFANet
## [CVPR2025] BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis



Backbones
- [x] Support OctFomer backbone;
- [x] Support MinkowskiEngine backbone;
- [ ] Support PTv3 backbone;

Datasets
- [x] Support ScanNetv2 Dataset;
- [ ] Support ScanNet200 Dataset;
- [ ] Support ScanNet++ Dataset;
- [ ] Support S3DIS Dataset;
- [ ] Support SemanticKITTI Dataset;

Others
- [x] Release training and testing code of BFANet;
- [ ] Support TTA (Test Time Augmentation);
- [ ] Evaluation of Four Proposed Metrics 

## Environments
Our code was verified on Four RTX 4090 with CUDA 11.8 and Python 3.8.


### Creat Conda Environment
    conda create -n BFANet python=3.8
    conda activate BFANet

### Install OctFormer
    conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch
    pip install -r requirements.txt
    cd lib
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv

Further OctFormer information can be found in [OctFormer](https://github.com/octree-nn/octformer)

### Install Segmentator 

```
cd segmentator
cd csrc && mkdir build && cd build
conda install cmake==3.26.4 cudnn

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`

make && make install # after install, please do not delete this folder (as we only create a symbolic link)
cd ../../../../
```
Further segmentator information can be found in [DKNet](https://github.com/W1zheng/DKNet) and [Segmentator](https://github.com/Karbo123/segmentator).

### Install Our Pseudo-label Lib
    cd lib/BFANet_lib
    python setup.py develop
    cd ../

### Install Minkowski Engine
    conda install openblas-devel -c anaconda
    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
    cd ../../

Further MinkowskiEngine information can be found in [MinkowskiEngine](https://github.com/W1zheng/DKNet)


### PointTransformerV3 environment 
    pip install spconv-cu118
    conda install yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
    conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

Further PointTransformerV3 information can be found in [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)

## Dataset Preparation
(1) Download the [ScanNet](http://www.scan-net.org/) dataset.

(2) Put the data in the corresponding folders. The dataset files are organized as follows.
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`,  `[scene_id].aggregation.json`  and `[scene_id]_vh_clean_2.labels.ply`  into the `datasets/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).

* Copy the files `[scene_id]_vh_clean_2.ply` into the `datasets/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).

* Put the file `scannetv2-labels.combined.tsv` in the `datasets/scannetv2` folder.


```
BFANet
├── data
│   ├── ScanNet
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json & [scene_id]_vh_clean_2.labels.ply
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json & [scene_id]_vh_clean_2.labels.ply
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```
(3) Decode the files to the "BFANet/datasets/ScanNetv2/npy/", if you don't want to use shared memory, please set the "use_shm_flag" as False in "BFANet/datasets/ScanNetv2/data_decode.py" 
    
    cd BFANet
    export PYTHONPATH=./
    python datasets/ScanNetv2/data_decode.py

## Environments
    cd BFANet
    python train.py


## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{zhao2025bfanet,
  title={BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis},
  author={Zhao, Weiguang and Zhang, Rui and Wang, Qiufeng and Cheng, Guangliang and Huang, Kaizhu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
  pages={29395--29405},
  year={2025}
}
```

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples: 
[OctFormer](https://github.com/dvlab-research/PointGroup), [Seg-Aliasing](https://github.com/Linwei-Chen/Seg-Aliasing), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3),[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [Superpoint Transformer](https://github.com/drprojects/superpoint_transformer), 
[Mix3d](https://github.com/kumuji/mix3d), [DKNet](https://github.com/W1zheng/DKNet), etc.
    

