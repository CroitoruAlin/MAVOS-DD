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
batch_size=6
lr_adapt=False
pretrain_path=/home/eivor/biodeep/Detection/MAVOS-DD/AVFF/checkpoints/stage-3.pth
use_effort=True


save_dir=/home/eivor/biodeep/Detection/MAVOS-DD/AVFF/checkpoints_trained/effort_r523_no_cross
mkdir -p $save_dir
mkdir -p ${save_dir}/models

CUDA_VISIBLE_DEVICES=0 python -m memory_profiler  ./src/run_ft.py --input_path /home/eivor/data/MAVOS-DD \
--save-dir $save_dir --n_classes 2 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--loss BCE --metrics mAP --warmup True \
--wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--head_lr ${head_lr} \
--pretrain_path ${pretrain_path} --use_effort ${use_effort} --num_workers 10
