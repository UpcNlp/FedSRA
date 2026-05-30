#!/bin/bash
cd /public/home/dongshou/fedETF/ETF-pesuade
export LD_LIBRARY_PATH='/opt/dtk/cuda/cuda-12/lib64:/usr/local/lib/python3.10/dist-packages/torch/lib:/opt/ucx/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/mpi/lib:/opt/hwloc/lib:' HIP_PATH=/opt/dtk ROCM_PATH=/opt/dtk
export HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2
/public/home/dongshou/anaconda/envs/ct/bin/python -u ce_train.py --alpha 0.05 --K 20 --epochs 150 --out saved_models/ce_a0.05_k20_s42
