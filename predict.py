"""
Qwen3 医学推理对话大模型 - 单条推理 / 人工抽查脚本

加载微调后保存的最优权重（train.py 训练结束会保存到 output/<run>/best），
对给定问题生成「先思考后作答」的结构化回答。

复用 evaluate.py 的 load_model_and_tokenizer，因此同时兼容：
    - 全量微调的完整模型目录；
    - LoRA adapter 目录（自动加载基座并合并）。

用法：
    python predict.py --model_path output/Qwen3-1.7B-full/best
    python predict.py --model_path <path> --question "我血糖偏高应该注意什么？"
"""

import argparse

from evaluate import load_model_and_tokenizer, DEVICE

PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_NEW_TOKENS = 2048

# 默认权重路径：与 train.py 的 OUTPUT_DIR/best 保持一致
DEFAULT_MODEL_PATH = "output/Qwen3-1.7B-full/best"
DEFAULT_QUESTION = (
    "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，"
    "我应该选择什么样的碳水化合物呢？"
)


def predict(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS
    )
    generated_ids = [
        out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="微调后权重目录（best）")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="待咨询的医学问题")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": args.question},
    ]
    response = predict(messages, model, tokenizer)
    print(response)


if __name__ == "__main__":
    main()
