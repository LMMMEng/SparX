_base_ = [
    'mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='sparx_mamba_b',
        pretrained=True,
        drop_path_rate=0.45
    ),
    neck=dict(in_channels=[120, 240, 396, 636]))
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, save_best='auto'))