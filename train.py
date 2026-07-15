"""
Qwen3 医学推理对话大模型 - 监督微调训练脚本

支持两种微调方式（通过 USE_LORA 切换），对应对照实验：
    - Exp A: Qwen3-1.7B + 全量微调   (USE_LORA=False)
    - Exp B: Qwen3-1.7B + LoRA 微调   (USE_LORA=True)
    - Exp C: Qwen3-8B  + LoRA 微调   (USE_LORA=True, MODEL_ID 改为 Qwen3-8B)

相比原始实现的关键改进：
    1. 全量微调学习率由 1e-4 降到 1e-5，LoRA 才用 1e-4；
    2. 增加 warmup + cosine 衰减 + weight_decay + 梯度裁剪，稳定训练；
    3. load_best_model_at_end 自动按 eval_loss 选取最优 checkpoint；
    4. 数据结尾使用 eos_token，保证模型学会自然停止；
    5. 固定随机种子，训练侧显式 bf16。
"""

import os
import json

import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
import swanlab

# ==================== 实验配置====================
# 环境变量优先，便于 run_all.sh 串联多组实验：
#   MODEL_ID / CACHE_DIR / OUTPUT_ROOT / USE_LORA(true|false)
SEED = 42
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")   
CACHE_DIR = os.environ.get("CACHE_DIR", "models")  
USE_LORA = os.environ.get("USE_LORA", "false").lower() == "true"  

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048

# 学习率：全量微调建议 1e-5(7e-6~2e-5)，LoRA 建议 1e-4(1e-4~2e-4)
LEARNING_RATE = 1e-4 if USE_LORA else 1e-5

# 数据集路径
TRAIN_DATASET_PATH = "train.jsonl"
VAL_DATASET_PATH = "val.jsonl"
TRAIN_FORMAT_PATH = "train_format.jsonl"
VAL_FORMAT_PATH = "val_format.jsonl"

_model_tag = MODEL_ID.split("/")[-1]
_ft_tag = "lora" if USE_LORA else "full"
RUN_NAME = f"{_model_tag}-{_ft_tag}"
OUTPUT_DIR = os.path.join(os.environ.get("OUTPUT_ROOT", "output"), RUN_NAME)

os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"
set_seed(SEED)


def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需的 {instruction, input, output} 格式。
    同时做基础脏数据过滤：字段缺失或 think/answer 为空的样本直接跳过。
    """
    messages = []
    skipped = 0
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = data.get("question", "").strip()
            think = data.get("think", "").strip()
            answer = data.get("answer", "").strip()
            # 脏数据过滤
            if not question or not think or not answer:
                skipped += 1
                continue
            output = f"<think>{think}</think> \n {answer}"
            messages.append({
                "instruction": PROMPT,
                "input": question,
                "output": output,
            })

    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

    print(f"[data] {origin_path} -> {new_path} | 有效 {len(messages)} 条, 过滤 {skipped} 条")


def build_process_func(tokenizer):
    """
    返回一个绑定了 tokenizer 的预处理函数（Dataset.map 使用）。
    - 采用 Qwen ChatML 模板拼接 system/user/assistant；
    - 仅在 assistant 段计算 loss（instruction 段用 -100 屏蔽）；
    - 结尾使用 eos_token 而非 pad_token，保证模型学会自然停止；
    - 超过 MAX_LENGTH 的样本对三个张量同步截断。
    """
    eos_token_id = tokenizer.eos_token_id
    truncated = {"count": 0}

    def process_func(example):
        instruction = tokenizer(
            f"<|im_start|>system\n{PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)

        input_ids = instruction["input_ids"] + response["input_ids"] + [eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [eos_token_id]

        if len(input_ids) > MAX_LENGTH:  # 截断
            truncated["count"] += 1
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    process_func.truncated = truncated
    return process_func


def main():
    swanlab.config.update({
        "model": MODEL_ID,
        "prompt": PROMPT,
        "data_max_length": MAX_LENGTH,
        "finetune": "lora" if USE_LORA else "full",
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
    })

    # 下载并加载模型/分词器
    model_dir = snapshot_download(MODEL_ID, cache_dir=CACHE_DIR, revision="master")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()  # 梯度检查点需要
    model.config.use_cache = False       # 与 gradient_checkpointing 兼容

    # LoRA 包装
    if USE_LORA:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 数据格式转换（幂等）
    if not os.path.exists(TRAIN_FORMAT_PATH):
        dataset_jsonl_transfer(TRAIN_DATASET_PATH, TRAIN_FORMAT_PATH)
    if not os.path.exists(VAL_FORMAT_PATH):
        dataset_jsonl_transfer(VAL_DATASET_PATH, VAL_FORMAT_PATH)

    process_func = build_process_func(tokenizer)

    train_df = pd.read_json(TRAIN_FORMAT_PATH, lines=True)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

    eval_df = pd.read_json(VAL_FORMAT_PATH, lines=True)
    eval_ds = Dataset.from_pandas(eval_df)
    eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)

    print(f"[data] 截断样本数: {process_func.truncated['count']} "
          f"(train+val 共 {len(train_dataset) + len(eval_dataset)} 条)")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,        # 等效 batch size = 4
        num_train_epochs=3,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.05,                    # 学习率预热
        lr_scheduler_type="cosine",           # 余弦衰减
        weight_decay=0.01,
        max_grad_norm=1.0,                    # 梯度裁剪
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,                       # 与 eval_steps 对齐
        save_total_limit=3,
        logging_steps=10,
        load_best_model_at_end=True,          # 自动选取最优 checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_on_each_node=True,
        report_to="swanlab",
        run_name=RUN_NAME,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()

    # 保存最优模型与分词器，便于 evaluate.py / predict.py 直接加载
    best_dir = os.path.join(OUTPUT_DIR, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"[train] 最优模型已保存到: {best_dir}")

    swanlab.finish()


if __name__ == "__main__":
    main()
