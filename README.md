# BFANet
## [CVPR2025] BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis



Backbones
- [ ] Support OctFomer backbone;
- [ ] Support MinkowskiEngine backbone;
- [ ] Support PTv3 backbone;

Datasets
- [ ] Support ScanNetv2 Dataset;
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
conda install cmake cudnn

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
    cd CCP_lib
    python setup.py develop
    cd ../

### Install Minkowski Engine
    conda install openblas-devel -c anaconda
    git clone https://github.com/NVIDIA/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

Further MinkowskiEngine information can be found in [MinkowskiEngine](https://github.com/W1zheng/DKNet)


### PointTransformerV3 environment 
    pip install spconv-cu118
    conda install yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
    conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

Further PointTransformerV3 information can be found in [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3)



## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{zhao2025BFANet,
  title={BFANet: Revisiting 3D Semantic Segmentation with Boundary Feature Analysis},
  author={Zhao, Weiguang and Zhang, Rui and Wang, Qiufeng and Cheng, Guangliang and Huang, Kaizhu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples: 
[OctFormer](https://github.com/dvlab-research/PointGroup), [Seg-Aliasing](https://github.com/Linwei-Chen/Seg-Aliasing), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3),[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [Superpoint Transformer](https://github.com/drprojects/superpoint_transformer), 
[DKNet](https://github.com/W1zheng/DKNet), etc.
    

