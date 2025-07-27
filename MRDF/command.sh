CUDA_VISIBLE_DEVICES=0 python test.py \
  --data_root /mnt/d/projects/datasets/MAVOS-DD \
  --checkpoint /mnt/d/projects/MAVOS-DD/MRDF/checkpoints/MRDF_CE.ckpt \
  --model_type MRDF_CE


  CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_type MRDF_CE --save_name MRDF_CE \
  --data_root /mnt/d/projects/datasets/MAVOS-DD \
  --dataset MAVOS-DD \
  --outputs ./output