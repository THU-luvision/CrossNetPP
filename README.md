# CrossNet++: Cross-Scale Large-Parallax Warping for Reference-Based Super-Resolution

This repo contains the implementation of the 'CrossNet++: Cross-Scale Large-Parallax Warping for Reference-Based Super-Resolution'. The improvements include multistage warping, keypoint guidence as well as extend real-world training set using hikvision dual camera.

This is a project from LuVision SIGMA, Tsinghua University. Visit our website for more interesting works: http://www.luvision.net/

## License
This project is released under the [GPLv3 license](LICENSE). We only allow free use for academic use. For commercial use, please contact us to negotiate a different license by: `fanglu at tsinghua.edu.cn`

## Usage

### Install Envs and Libs

```
conda env create -f env.yaml
```

### Download pretrained_models

1. download pretrained_models from [TSCloud](https://cloud.tsinghua.edu.cn/d/98f87f0a2e894c7d9ddc/)
2. move downloaded pretrained_models into source root
    ```
    - evalutation
    - ...
    - pretrained_models
      - dual_camera
        - CP250000.pth
        - ...
    ```

### Inference Only

 `python evaluation/eval.py`


### Training from scratch

1. prepare an hdf5 file (the same as [crossnet](https://github.com/htzheng/ECCV2018_CrossNet_RefSR)), which contains /img_HR, /img_LR, /img_MDSR, /img_LR_upsample. /img_HR is used as reference input and ground truth, /img_LR is used as low resolution input, /img_MDSR is the MDSR upsampled image, and /img_LR_upsample is bicubically upsampled image. (Different from the original paper, in this version of code, we use Flownet with bicubically upsampled image and reference image to generate optical flow)
2. `sh train/train.sh`

## Citation

Please cite our paper if you find it useful.

```
@inproceedings{zheng2018crossnet,
  title={Crossnet: An end-to-end reference-based super resolution network using cross-scale warping},
  author={Zheng, Haitian and Ji, Mengqi and Wang, Haoqian and Liu, Yebin and Fang, Lu},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={88--104},
  year={2018}
}
```

```
@article{tan2020crossnet++,
  title={Crossnet++: Cross-scale large-parallax warping for reference-based super-resolution},
  author={Tan, Yang and Zheng, Haitian and Zhu, Yinheng and Yuan, Xiaoyun and Lin, Xing and Brady, David and Fang, Lu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={43},
  number={12},
  pages={4291--4305},
  year={2020},
  publisher={IEEE}
}
```
