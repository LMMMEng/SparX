# Applying SparX to Semantic Segmentation   

## 1. Requirements

```
mmcv==2.1.0
mmengine=0.10.4
mmsegmentation==1.2.2
```

## 2. Data Preparation

Prepare ADE20K dataset according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md).  

## 3. Main Results on ADE20K using UperNet framework

|    Backbone   |   Pretrain  | Schedule | mIoU |                         Config                          | Download |
|:-------------:|:-----------:|:--------:|--------|:-------------------------------------------------------:|:----------:|
| SparX-Mamba-T | [ImageNet-1K](https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_tiny_in1k.pth)|    160K    |  50.2     |    [config](configs/upernet_sparx_mamba_t_ade20k.py)    |[model](https://github.com/LMMMEng/SparX/releases/download/v1/upernet_sparx_mamba_tiny.pth)          |
| SparX-Mamba-S | [ImageNet-1K](https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_small_in1k.pth)|    160K    |51.4       |    [config](configs/upernet_sparx_mamba_s_ade20k.py)    |[model](https://github.com/LMMMEng/SparX/releases/download/v1/upernet_sparx_mamba_small.pth)           |
| SparX-Mamba-B | [ImageNet-1K](https://github.com/LMMMEng/SparX/releases/download/v1/sparx_mamba_base_in1k.pth) |    160K    |52.5        |    [config](configs/upernet_sparx_mamba_b_ade20k.py)    |[model](https://github.com/LMMMEng/SparX/releases/download/v1/upernet_sparx_mamba_base.pth)           |
> 💡 We retrained all models after paper acceptance, achieving slightly better performance.

## 4. Train
To train ``SparX-Mamba-T + UperNet`` models on ADE20K dataset with 8 gpus (single node), run:
```
bash scripts/dist_train.sh configs/upernet_sparx_mamba_t_ade20k.py 8
```

## 5. Validation
To evaluate ``SparX-Mamba-T + UperNet`` models on COCO dataset, run:
```
bash scripts/dist_test.sh configs/upernet_sparx_mamba_t_ade20k.py path-to-checkpoint 8
```

## Citation
If you find this project useful for your research, please consider citing:

```
@inproceedings{lou2024sparx,
  title={SparX: A Sparse Cross-Layer Connection Mechanism for Hierarchical Vision Mamba and Transformer Networks},
  author={Lou, Meng and Fu, Yunxiang and Yu, Yizhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```
