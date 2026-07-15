"""
Qwen3 医学推理对话大模型 - 对照实验一键评估与对比脚本

对应项目计划「对照实验设计」，在同一验证集上依次评估 Baseline / Exp A / Exp B / Exp C，
复用 evaluate.py 的评估逻辑，最终汇总为一张横向对比表（控制台 + Markdown）。

用法：
    # 评估下方 EXPERIMENTS 中所有存在的实验
    python compare.py

    # 抽样 50 条快速对比
    python compare.py --num_samples 50

    # 已经跑过 evaluate.py，只想从已有 eval_results_*.json 汇总，不重新推理
    python compare.py --from_cache

注意：
    - 只会评估 model_path 真实存在的实验，缺失的自动跳过（并给出提示）。
    - 模型路径需与 train.py 的 OUTPUT_DIR/best 及基座模型缓存路径保持一致。
"""

import os
import json
import argparse

from evaluate import evaluate

# ==================== 对照实验清单（按需修改路径） ====================
# tag 用于结果文件命名与表格行名；model_path 为待评估权重目录。
EXPERIMENTS = [
    {"tag": "baseline",       "model_path": "models/Qwen/Qwen3-1.7B"},
    {"tag": "expA-1.7B-full", "model_path": "output/Qwen3-1.7B-full/best"},
    {"tag": "expB-1.7B-lora", "model_path": "output/Qwen3-1.7B-lora/best"},
    {"tag": "expC-8B-lora",   "model_path": "output/Qwen3-8B-lora/best"},
]

# 汇总表展示的指标列（key -> 表头）
METRIC_COLUMNS = [
    ("perplexity", "PPL↓"),
    ("format_compliance_rate", "格式合规率↑"),
    ("avg_semantic_similarity", "语义相似度↑"),
    ("avg_latency_s", "延迟/样本(s)↓"),
    ("decode_throughput_tok_s", "吞吐(tok/s)↑"),
    ("avg_new_tokens", "平均生成长度"),
    ("num_samples", "样本数"),
]


def load_cached_result(tag):
    """从 eval_results_<tag>.json 读取已有评估结果。"""
    path = f"eval_results_{tag}.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def fmt(value):
    return "-" if value is None else str(value)


def render_table(results):
    """将结果列表渲染为对齐的控制台表格与 Markdown 表格。"""
    headers = ["实验(tag)"] + [h for _, h in METRIC_COLUMNS]
    rows = []
    for r in results:
        row = [r["tag"]] + [fmt(r.get(k)) for k, _ in METRIC_COLUMNS]
        rows.append(row)

    # 计算每列宽度
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(cells):
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    print("\n==================== 对照实验对比 ====================")
    print(line(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(line(row))
    print("=====================================================\n")

    # Markdown 表格，便于粘贴进项目计划/报告
    md = ["| " + " | ".join(headers) + " |",
          "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        md.append("| " + " | ".join(str(c) for c in row) + " |")
    md_text = "\n".join(md)
    with open("comparison.md", "w", encoding="utf-8") as f:
        f.write("# 对照实验对比结果\n\n" + md_text + "\n")
    print("[compare] Markdown 对比表已写入: comparison.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_path", default="val_format.jsonl", help="格式化后的验证集")
    parser.add_argument("--num_samples", type=int, default=-1, help="评估样本数，-1 表示全部")
    parser.add_argument("--from_cache", action="store_true",
                        help="仅从已有 eval_results_*.json 汇总，不重新推理")
    args = parser.parse_args()

    results = []
    for exp in EXPERIMENTS:
        tag, model_path = exp["tag"], exp["model_path"]

        if args.from_cache:
            summary = load_cached_result(tag)
            if summary is None:
                print(f"[skip] 无缓存结果: eval_results_{tag}.json")
                continue
            results.append(summary)
            continue

        if not os.path.exists(model_path):
            print(f"[skip] 模型不存在，跳过 {tag}: {model_path}")
            continue

        print(f"\n########## 评估 {tag} ##########")
        summary = evaluate(model_path, args.val_path, args.num_samples, tag)
        results.append(summary)

    if not results:
        print("[compare] 没有可汇总的结果。请先训练/评估，或检查 EXPERIMENTS 中的路径。")
        return

    render_table(results)


if __name__ == "__main__":
    main()
