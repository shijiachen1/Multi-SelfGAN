#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_derived.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_iter 65000 \
--gen_model shared_gan_block \
--dis_model shared_gan_block \
--teacher_gen_b teacher_gen_a_pre \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--tea_df_dim 128 \
--tea_gf_dim 256 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 20 \
--arch 0 1 0 1 1 1 1 1 0 1 0 2 1 3 \
--exp_name derive
