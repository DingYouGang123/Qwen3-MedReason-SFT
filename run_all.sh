#!/usr/bin/env bash
# =============================================================================
# Qwen3 医学推理对话大模型 - 对照实验一键脚本
#
# 串联执行：数据准备 -> 训练 Exp A/B/C -> 一键评估并汇总对比表。
# 通过环境变量 USE_LORA / MODEL_ID 覆盖 train.py 的配置。
#
# 用法：
#   bash run_all.sh              # 完整跑三组实验 + 对比
#   SKIP_TRAIN=1 bash run_all.sh # 跳过训练，仅评估+汇总（模型已就绪时）
#   NUM_SAMPLES=50 bash run_all.sh  # 评估阶段抽样 50 条快速对比
#
# 注意：训练非常耗时且吃显存，请确认在带 GPU 的环境（如 AutoDL）中执行。
# =============================================================================
set -e  # 任一步骤出错立即终止

NUM_SAMPLES="${NUM_SAMPLES:--1}"   # 评估样本数，-1 表示全部

echo "==================== [0/5] 环境与数据准备 ===================="
# 生成 train.jsonl / val.jsonl（若已存在则跳过）
if [ ! -f "train.jsonl" ] || [ ! -f "val.jsonl" ]; then
    echo "[data] 下载并切分数据集..."
    # download.py 使用相对路径写入当前工作目录，故从项目根目录执行，直接生成到根目录
    python data/download.py
else
    echo "[data] 已存在 train.jsonl / val.jsonl，跳过下载。"
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    echo "==================== [1/5] 训练 Exp A: 1.7B 全量微调 ===================="
    USE_LORA=false MODEL_ID="Qwen/Qwen3-1.7B" python train.py

    echo "==================== [2/5] 训练 Exp B: 1.7B LoRA ===================="
    USE_LORA=true MODEL_ID="Qwen/Qwen3-1.7B" python train.py

    echo "==================== [3/5] 训练 Exp C: 8B LoRA（可选，显存不足可注释） ===================="
    USE_LORA=true MODEL_ID="Qwen/Qwen3-8B" python train.py
else
    echo "[train] SKIP_TRAIN=1，跳过训练阶段。"
fi

echo "==================== [4/5] 一键评估 + 汇总对比 ===================="
# compare.py 会自动跳过不存在的模型（如未训练的 Exp C）
python compare.py --num_samples "${NUM_SAMPLES}"

echo "==================== [5/5] 完成 ===================="
echo "对比表: comparison.md"
echo "各实验明细: eval_results_*.json"
