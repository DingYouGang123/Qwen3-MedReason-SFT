"""
Qwen3 医学推理对话大模型 - 验证集离线定量评估脚本

在同一验证集上计算同一套指标，便于横向对比：
    1. 困惑度 PPL          —— 语言建模拟合程度（越低越好）
    2. 格式合规率          —— 输出是否成对且顺序正确地包含 <think>...</think> 且答案非空
    3. 语义相似度          —— 生成答案(去 think 段) 与参考 answer 的 Qwen3-Embedding 余弦相似度
    4. 推理时间            —— 端到端生成延迟(越低越好) 与解码吞吐 tokens/s(越高越好)，
                             用于对比不同规模(1.7B/8B)与不同微调方式(全量/LoRA)的部署成本

用法示例：
    # 评估某个微调 checkpoint
    python evaluate.py --model_path /root/autodl-tmp/output/Qwen3-1.7B-full/best

    # 评估原始模型作为 Baseline
    python evaluate.py --model_path /root/autodl-tmp/Qwen/Qwen3-1.7B --tag baseline

    # 仅抽样 50 条快速评估
    python evaluate.py --model_path <path> --num_samples 50
"""

import os
import re
import json
import math
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048
MAX_NEW_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 语义相似度所用 embedding 模型（Qwen3-Embedding 系列，可用环境变量覆盖）
# 默认 0.6B：评估 8B 时若用 4B 且 embedder 以 fp32 加载会 OOM（8B+4B 峰值超 32GB）；
# 0.6B 在短答案余弦相似度场景精度足够。显存充裕(仅评 1.7B)可设 EMBED_MODEL_ID=Qwen/Qwen3-Embedding-4B
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "Qwen/Qwen3-Embedding-0.6B")


# ---------------------------------------------------------------- 文本工具
def strip_think(text: str) -> str:
    """去除 <think>...</think> 段，只保留最终答案部分。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def check_format(text: str) -> bool:
    """格式合规：恰好一对 <think></think>，顺序正确，且 </think> 之后答案非空。"""
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False
    open_idx = text.find("<think>")
    close_idx = text.find("</think>")
    if open_idx > close_idx:  # 顺序错误
        return False
    answer = text.split("</think>", 1)[1].strip()
    return len(answer) > 0


# ---------------------------------------------------------------- 生成 & PPL
def generate(model, tokenizer, user_input: str):
    """生成回答，并测量端到端推理延迟与新生成 token 数。

    返回 (text, latency_s, num_new_tokens)。
    计时要点：
    - CUDA 是异步执行，必须在计时前后 torch.cuda.synchronize()，否则测到的
      只是 kernel 下发耗时而非真实执行耗时；
    - 冷启动（首次 kernel 编译）开销由外层 warmup 排除，此处不额外处理。
    """
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    prompt_len = model_inputs.input_ids.shape[1]

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS
    )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    latency_s = time.perf_counter() - t0

    new_ids = generated_ids[0][prompt_len:]
    num_new_tokens = int(new_ids.shape[0])  # 含 eos，作为解码步数
    answer = tokenizer.decode(new_ids, skip_special_tokens=True)
    return answer, latency_s, num_new_tokens


def compute_sample_ppl(model, tokenizer, user_input: str, target_output: str):
    """按训练时相同的拼接方式，计算单条样本在 assistant 段上的困惑度。"""
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(target_output, add_special_tokens=False)
    eos = tokenizer.eos_token_id

    input_ids = instruction["input_ids"] + response["input_ids"] + [eos]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [eos]
    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]

    input_tensor = torch.tensor([input_ids], device=DEVICE)
    label_tensor = torch.tensor([labels], device=DEVICE)
    with torch.no_grad():
        out = model(input_ids=input_tensor, labels=label_tensor)
    return out.loss.item()  # 该样本 assistant 段的平均交叉熵


# ---------------------------------------------------------------- embedding
def load_embedder():
    """加载 Qwen3-Embedding 模型；未安装 sentence-transformers 时返回 None。

    需要 transformers>=4.51.0 与 sentence-transformers>=2.7.0。
    如需加速/省显存，可在 model_kwargs 中启用 flash_attention_2：
        SentenceTransformer(
            EMBED_MODEL_ID,
            model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
            tokenizer_kwargs={"padding_side": "left"},
        )
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBED_MODEL_ID, device=DEVICE)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 语义相似度不可用（{e}）；将跳过该指标。")
        return None


def cosine_sim(embedder, generated: str, reference: str) -> float:
    """用 Qwen3-Embedding 计算生成答案与参考答案的余弦相似度。

    生成答案视为 query（使用内置 "query" prompt 获得更优表征），参考答案视为
    document，再用 model.similarity 计算余弦相似度。
    """
    query_emb = embedder.encode([generated], prompt_name="query")
    doc_emb = embedder.encode([reference])
    return embedder.similarity(query_emb, doc_emb)[0][0].item()


# ---------------------------------------------------------------- 模型加载
def load_model_and_tokenizer(model_path: str):
    """加载模型与分词器，自动兼容「完整模型」与「LoRA adapter」两种目录。

    - 若目录下存在 adapter_config.json，识别为 LoRA：读取其中的
      base_model_name_or_path 加载基座，再用 PeftModel 挂载 adapter，
      最后 merge_and_unload 合并权重——既能正常评估，又使推理速度与全量模型
      一致（去掉未合并 adapter 的额外算子，保证推理时间可比）；
    - 否则按普通完整模型加载；
    - 统一强制 use_cache=True：训练时为兼容 gradient_checkpointing 会设成 False
      并写入 config，若不覆盖会导致生成不走 KV cache，既慢又与 Baseline 口径不一致。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        from peft import PeftModel
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_path = json.load(f).get("base_model_name_or_path")
        if not base_path or not os.path.exists(base_path):
            raise FileNotFoundError(
                f"LoRA adapter 的基座路径不存在: {base_path}（来自 {adapter_cfg}）。"
                f"请确认基座模型已下载，或修正 adapter_config.json。"
            )
        print(f"[load] 检测到 LoRA adapter，加载基座并合并: {base_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重，去除额外算子
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model.config.use_cache = True  # 生成需 KV cache（训练存的 False 必须覆盖）
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------- 主流程
def evaluate(model_path: str, val_path: str, num_samples: int, tag: str):
    print(f"[eval] 加载模型: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)

    embedder = load_embedder()

    # 读取验证集（{instruction, input, output}）
    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if num_samples > 0:
        samples = samples[:num_samples]
    print(f"[eval] 评估样本数: {len(samples)}")

    # 预热：首次 generate 含 CUDA kernel 编译等冷启动开销，先跑一次并丢弃，
    # 避免第 1 条样本的推理时间被高估，保证各实验计时口径一致。
    if samples:
        print("[eval] warmup ...")
        _ = generate(model, tokenizer, samples[0]["input"])

    losses, format_ok, sims, records = [], 0, [], []
    latencies, new_token_counts = [], []
    for i, s in enumerate(samples):
        user_input = s["input"]
        ref_output = s["output"]
        ref_answer = strip_think(ref_output)

        # 1) PPL（teacher-forcing loss）
        loss = compute_sample_ppl(model, tokenizer, user_input, ref_output)
        losses.append(loss)

        # 2) 生成 + 格式合规 + 3) 语义相似度 + 4) 推理时间
        gen, latency_s, num_new_tokens = generate(model, tokenizer, user_input)
        latencies.append(latency_s)
        new_token_counts.append(num_new_tokens)
        is_ok = check_format(gen)
        format_ok += int(is_ok)

        sim = None
        if embedder is not None:
            gen_answer = strip_think(gen)
            if gen_answer:
                sim = cosine_sim(embedder, gen_answer, ref_answer)
                sims.append(sim)

        records.append({
            "input": user_input,
            "generated": gen,
            "format_ok": is_ok,
            "similarity": sim,
            "sample_loss": loss,
            "latency_s": round(latency_s, 4),
            "new_tokens": num_new_tokens,
            "tokens_per_sec": round(num_new_tokens / latency_s, 2) if latency_s > 0 else None,
        })
        if (i + 1) % 10 == 0:
            print(f"  progress: {i + 1}/{len(samples)}")

    avg_loss = sum(losses) / len(losses)
    total_time = sum(latencies)
    total_tokens = sum(new_token_counts)
    result = {
        "tag": tag,
        "model_path": model_path,
        "num_samples": len(samples),
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(math.exp(avg_loss), 4),
        "format_compliance_rate": round(format_ok / len(samples), 4),
        "avg_semantic_similarity": round(sum(sims) / len(sims), 4) if sims else None,
        # 推理时间：平均端到端延迟 + 平均生成长度 + 聚合解码吞吐(总token/总耗时)。
        # 同时报延迟与吞吐，避免"答得更长=更慢"这一生成长度混淆。
        "avg_latency_s": round(total_time / len(latencies), 4),
        "avg_new_tokens": round(total_tokens / len(new_token_counts), 1),
        "decode_throughput_tok_s": round(total_tokens / total_time, 2) if total_time > 0 else None,
    }

    print("\n==================== 评估结果 ====================")
    for k, v in result.items():
        print(f"{k:>26}: {v}")
    print("=================================================")

    out_path = f"eval_results_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": result, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"[eval] 明细已写入: {out_path}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="待评估模型或 checkpoint 目录")
    p.add_argument("--val_path", default="val_format.jsonl", help="格式化后的验证集")
    p.add_argument("--num_samples", type=int, default=-1, help="评估样本数，-1 表示全部")
    p.add_argument("--tag", default=None, help="实验标签，用于结果文件命名")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tag = args.tag or os.path.basename(os.path.normpath(args.model_path))
    evaluate(args.model_path, args.val_path, args.num_samples, tag)
