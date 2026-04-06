#!/bin/bash
# ═══════════════════════════════════════════════════════════
# ICDE 论文完整实验脚本
# ═══════════════════════════════════════════════════════════

# ── 实验一: 主实验 α×K 网格 (CIFAR-10, 多 seed) ──────────
# 3 台机器各跑一个 seed
# 机器 1:
python run_full_experiments.py --mode grid --dataset cifar10 --seed 42 --gpu 0 \
    --alphas 0.05,0.1,0.3,0.5,1.0 --clients 5,10,20,50

# 机器 2:
python run_full_experiments.py --mode grid --dataset cifar10 --seed 0 --gpu 0 \
    --alphas 0.05,0.1,0.3,0.5,1.0 --clients 5,10,20,50

# 机器 3:
python run_full_experiments.py --mode grid --dataset cifar10 --seed 123 --gpu 0 \
    --alphas 0.05,0.1,0.3,0.5,1.0 --clients 5,10,20,50

# K=100 单独跑 (耗时长)
python run_full_experiments.py --mode grid --dataset cifar10 --seed 42 --gpu 0 \
    --alphas 0.05,0.1,0.3,0.5,1.0 --clients 100


# ── 实验二: CIFAR-100 (对齐 FAFI Table 1) ────────────────
python run_full_experiments.py --mode grid --dataset cifar100 --seed 42 --gpu 0 \
    --alphas 0.05,0.1,0.3,0.5 --clients 5,10,20,50

python run_full_experiments.py --mode grid --dataset cifar100 --seed 0 --gpu 1 \
    --alphas 0.05,0.1,0.3,0.5 --clients 5,10,20,50


# ── 实验三: FAFI 对比 (K=5, 对齐 FAFI 设置) ──────────────
python run_full_experiments.py --mode fafi_compare --dataset cifar10 --seed 42 --gpu 0
python run_full_experiments.py --mode fafi_compare --dataset cifar100 --seed 42 --gpu 0


# ── 实验四: 消融实验 ──────────────────────────────────────
# 在代表性设置下做消融
python run_full_experiments.py --mode ablation --dataset cifar10 --alpha 0.1 --n_clients 5 --gpu 0
python run_full_experiments.py --mode ablation --dataset cifar10 --alpha 0.3 --n_clients 10 --gpu 0
python run_full_experiments.py --mode ablation --dataset cifar10 --alpha 0.5 --n_clients 20 --gpu 0
python run_full_experiments.py --mode ablation --dataset cifar10 --alpha 0.1 --n_clients 50 --gpu 0


# ── 汇总结果 ──────────────────────────────────────────────
python collect_results.py --results_dir results --output summary