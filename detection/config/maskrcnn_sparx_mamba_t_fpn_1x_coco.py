_base_ = [
    'mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='sparx_mamba_t',
        pretrained=True,
        drop_path_rate=0.2
    ),
    neck=dict(in_channels=[96, 192, 320, 512]))
default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto'))