_base_ = [
    'mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='sparx_mamba_s',
        pretrained=True,
        drop_path_rate=0.3
    ),
    neck=dict(in_channels=[96, 192, 328, 544]))
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, save_best='auto'))