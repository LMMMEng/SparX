# Applying SparX to Object Detection and Instance Segmentation   

## 1. Requirements

```
mmcv==2.1.0
mmengine=0.10.4
mmdet==3.30
```

## 2. Data Preparation

Prepare COCO 2017 according to the [guidelines](https://github.com/open-mmlab/mmdetection/blob/2.x/docs/en/1_exist_data_model.md).  

## 3. Main Results on COCO using Mask R-CNN framework


|    Backbone   |   Pretrain  | Schedule | AP_b | AP_m | Config | Download |
|:-------------:|:-----------:|:--------:|--------|:-------:|:------:|:----------:|
| SparX-Mamba-T | [ImageNet-1K](https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_tiny_in1k.pth)|    1x    |  48.1  |43.1     |[config](config/maskrcnn_sparx_mamba_t_fpn_1x_coco.py)        |[model](https://github.com/LMMMEng/SparX/releases/download/v1/maskrcnn_sparx_mamba_t_fpn_1x_coco.pth)          |
|               |             |    3x    |50.2        |44.7         |[config](config/maskrcnn_sparx_mamba_t_fpn_3x_coco.py)        |[model](https://github.com/LMMMEng/SparX/releases/download/v1/maskrcnn_sparx_mamba_t_fpn_3x_coco.pth)          |
| SparX-Mamba-S | [ImageNet-1K](https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_small_in1k.pth)|    1x    |49.4        |44.1         |[config](config/maskrcnn_sparx_mamba_s_fpn_1x_coco.py)        |[model](https://github.com/LMMMEng/SparX/releases/download/v1/maskrcnn_sparx_mamba_s_fpn_1x_coco.pth)           |
|               |             |    3x    |51.0        |45.2         |[config](config/maskrcnn_sparx_mamba_s_fpn_3x_coco.py)        |[model](https://github.com/LMMMEng/SparX/releases/download/v1/maskrcnn_sparx_mamba_s_fpn_3x_coco.pth)          |
| SparX-Mamba-B | [ImageNet-1K](https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_base_in1k.pth) |    1x    |49.7        |44.3         |[config](config/maskrcnn_sparx_mamba_b_fpn_1x_coco.py)        |[model](https://github.com/LMMMEng/SparX/releases/download/v1/maskrcnn_sparx_mamba_b_fpn_1x_coco.pth)           |
|               |             |    3x    |51.8       |45.8         |[config](config/maskrcnn_sparx_mamba_b_fpn_3x_coco.py)        |[model](https://github.com/LMMMEng/SparX/releases/download/v1/maskrcnn_sparx_mamba_b_fpn_3x_coco.pth)          |


## 4. Train
To train ``SparX-Mamba-T + Mask R-CNN 1x`` models on COCO dataset with 8 gpus (single node), run:
```
bash scripts/dist_train.sh config/maskrcnn_sparx_mamba_t_fpn_1x_coco.py 8
```

## 5. Validation
To evaluate ``SparX-Mamba-T + Mask R-CNN 1x`` models on COCO dataset, run:
```
bash scripts/dist_test.sh config/maskrcnn_sparx_mamba_t_fpn_1x_coco.py path-to-checkpoint 8
```

## Citation
If you find this project useful for your research, please consider citing:

```
@article{lou2024sparx,
  title={SparX: A Sparse Cross-Layer Connection Mechanism for Hierarchical Vision Mamba and Transformer Networks},
  author={Lou, Meng and Fu, Yunxiang and Yu, Yizhou},
  journal={arXiv preprint arXiv:2409.09649},
  year={2024}
}
```
