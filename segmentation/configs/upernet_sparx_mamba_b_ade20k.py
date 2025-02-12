_base_ = [
    'swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='sparx_mamba_b',
        pretrained=True,
        drop_path_rate=0.5,
    ),
    decode_head=dict(
        in_channels=[120, 240, 396, 636],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=396,
        num_classes=150
    ))

runner = dict(type='IterBasedRunner', max_iters=8000)
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3, save_best='mIoU'))
train_dataloader = dict(batch_size=2)