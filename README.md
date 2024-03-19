# SE3ET: SE(3)-Equivariant Transformer for Low-Overlap Point Cloud Registration

This work proposes exploiting equivariant learning from 3D point clouds to improve registration robustness. We propose SE3ET, an SE(3)-equivariant registration framework that employs equivariant point convolution and equivariant transformer design to learn expressive and robust geometric features.

## Installation

Please use the following command for installation.

Build docker environment or conda environment
- docker
```
cd docker
docker build --tag umcurly/geotransformer .
bash build_docker_container.bash [container_name]
```
- conda
```
conda create -n se3et python==3.8
conda activate se3et
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

```
Build extension package for GeoTransformer (might need root access)
```
python setup.py build develop
```
Build extension package for E2PN
```
cd geotransformer/modules/e2pn/vgtk
python setup.py build_ext -i
```

Code has been tested with Ubuntu 20.04, GCC 9.3.0, Python 3.8, PyTorch 1.7.1, CUDA 11.1 and cuDNN 8.1.0.

## Pre-trained Weights

Upcoming!
Please download the latest weights and put them in `weights` directory.

## 3DMatch

### Data Preparation

The dataset can be downloaded from [PREDATOR](https://github.com/prs-eth/OverlapPredator). The data should be organized as follows:

```text
--data--3DMatch--metadata
              |--data--train--7-scenes-chess--cloud_bin_0.pth
                    |      |               |--...
                    |      |--...
                    |--test--7-scenes-redkitchen--cloud_bin_0.pth
                          |                    |--...
                          |--...
```

You may create symlink at `data/3DMatch/data` to the actual dataset path, and symlink at `output` to the path you want to save outputs and log files. 

### Training

The code for 3DMatch is in `experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```
#### Training with E2PN

The code for 3DMatch is in `experiments/e2pntransformer.3dmatchsmall.inv.exp4`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```
For experiment, you can copy the `experiments/e2pntransformer.3dmatchsmall.inv.exp4` folder or create a new folder under `experiments`. Inside the folder, change `config.py` and `backbone.py` accordingly. If GPU memory is limited, lower `train.point_limit`, `test.point_limit`, `backbone.init_dim`, `backbone.output_dim`, `geotransformer.input_dim`, `geotransformer.hidden_dim`, `geotransformer.output_dim`.

For invariant feature encoder, `geotransformer.blocks` is set as `['self', 'cross', 'self', 'cross', 'self', 'cross']` and the initialization of the `GeometricTransformer` in `model.py` doesn't include `kanchor` arguement. 

### Testing

Use the following command for testing.

```bash
# 3DMatch
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH 3DMatch
# 3DLoMatch
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH 3DLoMatch
```

`EPOCH` is the epoch id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/geotransformer-3dmatch.pth.tar --benchmark=3DMatch
CUDA_VISIBLE_DEVICES=0 python eval.py --benchmark=3DMatch --method=lgr
```

Replace `3DMatch` with `3DLoMatch` to evaluate on 3DLoMatch.

Use `--cfg_file` option to point to a `yaml` config file to override the options in `experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/config.py`. 

## Kitti odometry

### Data preparation

Download the data from the [Kitti official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) into `data/Kitti` and run `data/Kitti/downsample_pcd.py` to generate the data. The data should be organized as follows:

```text
--data--Kitti--metadata
            |--sequences--00--velodyne--000000.bin
            |              |         |--...
            |              |...
            |--downsampled--00--000000.npy
                         |   |--...
                         |--...
```

### Training

The code for Kitti is in `experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 ./eval.sh EPOCH
```

`EPOCH` is the epoch id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/geotransformer-kitti.pth.tar
CUDA_VISIBLE_DEVICES=0 python eval.py --method=lgr
```

## ModelNet

### Data preparation

Download the [data](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) and run `data/ModelNet/split_data.py` to generate the data. The data should be organized as follows:

```text
--data--ModelNet--modelnet_ply_hdf5_2048--...
               |--train.pkl
               |--val.pkl
               |--test.pkl
```

### Training

The code for ModelNet is in `experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn`. Use the following command for training.

```bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
```

### Testing

Use the following command for testing.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --test_iter=ITER
```

`ITER` is the iteration id.

We also provide pretrained weights in `weights`, use the following command to test the pretrained weights.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/geotransformer-modelnet.pth.tar
```

## Multi-GPU Training

As the point clouds usually have different sizes, we organize them in the *pack* mode. This causes difficulty for batch training as we need to convert the data between *batch* mode and *pack* mode frequently. For this reason, we limit the batch size to 1 per GPU at this time and support batch training via `DistributedDataParallel`. Use `torch.distributed.launch` for multi-gpu training:

```bash
CUDA_VISIBLE_DEVICES=GPUS python -m torch.distributed.launch --nproc_per_node=NGPUS trainval.py
```

Note that the learning rate is multiplied by the number of GPUs by default as the batch size increased. In our experiments, multi-gpu training slightly improves the performance.

## Acknowledgements
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)

We thank the authors for open sourcing their methods, the code is heavily borrowed from [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)
