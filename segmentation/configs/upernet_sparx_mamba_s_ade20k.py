_base_ = [
    'swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='sparx_mamba_s',
        pretrained=True,
        drop_path_rate=0.4,
    ),
    decode_head=dict(
        in_channels=[96, 192, 328, 544],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=328,
        num_classes=150
    ))

runner = dict(type='IterBasedRunner', max_iters=8000)
default_hooks = dict(checkpoint=dict(interval=8000, max_keep_ckpts=3, save_best='mIoU'))
train_dataloader = dict(batch_size=2)