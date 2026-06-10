#!/bin/bash
source /opt/dtk/env.sh
source /public/home/dongshou/anaconda/etc/profile.d/conda.sh
conda activate ct
cd /public/home/dongshou/fedETF/ETF-pesuade
mkdir -p tmp
export HIP_VISIBLE_DEVICES=5 CUDA_VISIBLE_DEVICES=5
which python
python -c "import torch;print(\"torch\",torch.__version__,\"gpu\",torch.cuda.is_available())"
for a in 0.1 0.3; do
  echo "##### alpha=$a #####"
  python -u export_align_feats.py --alpha $a --K 10 --ckpt_dir saved_models/a${a}_k10_s42 --tag ERL --out tmp/align_ERL_a${a}_k10.npz
done
echo "ALLDONE"
