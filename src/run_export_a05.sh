#!/bin/bash
source /opt/dtk/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct
cd /public/home/dongshou/fedETF/ETF-pesuade
export HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5
python -u export_align_feats.py --alpha 0.5 --K 10 --ckpt_dir saved_models/a0.5_k10_s42 --tag ERL --out tmp/align_ERL_a0.5_k10.npz
echo ALLDONE05
