#!/usr/bin/env bash
# Please disable amp training if you encounter loss=nan
python3 -m torch.distributed.launch \
--master_port=$((RANDOM+10000)) \
--nproc_per_node=8 \
train.py \
--data-dir /data/dataset/imagenet/ \
--batch-size 256 \
--model sparx_mamba_b \
--lr 4e-3 \
--drop-path 0.5 \
--epochs 300 \
--warmup-epochs 20 \
--workers 10 \
--output output/sparx_mamba_b/ \
--model-ema \
--model-ema-decay 0.9998 \
--native-amp \
--clip-grad 5