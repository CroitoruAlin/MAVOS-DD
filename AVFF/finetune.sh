#!/bin/bash

contrast_loss_weight=0.01
mae_loss_weight=1.0
norm_pix_loss=True

lr=1e-5
head_lr=50
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa_start=1
wa_end=10
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
batch_size=42
lr_adapt=False
pretrain_path=checkpoints/stage-3.pth


save_dir=./exp/stage-3
mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_VISIBLE_DEVICES=0 python -W ignore ./src/run_ft.py --input_path /mnt/d/projects/datasets/MAVOS-DD
