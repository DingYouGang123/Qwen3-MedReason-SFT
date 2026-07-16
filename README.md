# 基于监督微调的医学推理对话大模型

> 基于 **Qwen3** 的医学推理对话大模型监督微调（SFT）项目 —— 让模型学会「**先思考（`<think>`）后作答**」的结构化医学问答；支持**全量微调**与 **LoRA**，内置 *数据处理 → 训练 → 评估 → 对照实验* 完整流水线，并提供 PPL / 格式合规率 / 语义相似度 / 推理性能的**定量对比**。

---

## ✨ 核心特点

- 🩺 **医学垂域微调**：基于 `delicate_medical_r1_data` 中文医学推理数据集，含 R1 风格思维链。
- 🧠 **结构化推理**：通过数据侧强约束 `<think>...</think> \n {answer}`，让模型隐式学会「先思考后作答」。
- 🔀 **多微调方法**：全量微调（Full Fine-tuning）与 LoRA 通过环境变量 `USE_LORA` 一键切换，无代码分叉。
- 🔬 **严谨对照实验**：Baseline + 控制变量设计，隔离「微调方法」与「模型规模」两个变量，结论可归因。
- 📊 **一站式定量评估**：PPL、格式合规率、语义相似度、推理延迟 / 吞吐 / 生成长度，同一验证集横向对比。
- ⚡ **一键流水线**：`run_all.sh` 串联 数据 → 训练三组 → 评估汇总；固定随机种子，结果可复现。

---

## 📁 目录结构

```
Qwen3_SFT/
├── download.py           # 下载数据集并按 9:1 切分 train/val（seed=42）
├── train.py              # 训练脚本（全量微调 / LoRA，环境变量切换）
├── evaluate.py           # 验证集离线定量评估（PPL / 格式 / 语义 / 性能）
├── compare.py            # 一键评估 Baseline/A/B/C 并汇总对比表
├── predict.py            # 单条推理 / 人工抽查
├── run_all.sh            # 一键流水线：数据 → 训练三组 → 评估对比
├── requirements.txt      # Python 依赖清单
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- **硬件**：带 GPU 的 Linux 环境（如 AutoDL）；1.7B 全量微调建议显存 ≥ 24GB。
- **Python**：3.10+

### 依赖安装

```bash
pip install -r requirements.txt
```

> `requirements.txt` 为完整环境快照（含 CUDA 12.8 版 `torch`）。其中 `peft` 用于 LoRA 微调（Exp B/C），`sentence-transformers` 用于语义相似度评估（缺失会自动跳过该指标）。
>
> **SwanLab**：首次使用需登录 `swanlab login`（训练监控）。

### 方式一：一键流水线（推荐）

```bash
bash run_all.sh                 # 数据准备 + 训练 A/B/C + 评估汇总
SKIP_TRAIN=1 bash run_all.sh    # 已训练完，仅评估 + 汇总
NUM_SAMPLES=50 bash run_all.sh  # 评估阶段抽样 50 条快速验证流程
```

产物：`comparison.md`（对比表）、`eval_results_*.json`（各实验明细）。

### 方式二：分步执行

**1. 准备数据**（在项目根目录执行，生成 `train.jsonl` / `val.jsonl`）

```bash
python download.py
```

**2. 训练**（用环境变量切换实验；不设则默认 1.7B 全量微调）

```bash
USE_LORA=false MODEL_ID="Qwen/Qwen3-1.7B" python train.py   # Exp A
USE_LORA=true  MODEL_ID="Qwen/Qwen3-1.7B" python train.py   # Exp B
USE_LORA=true  MODEL_ID="Qwen/Qwen3-8B"  python train.py    # Exp C
```

训练结束，最优权重保存到 `output/<run_name>/best`。

**3. 评估单个模型**（`evaluate.py` 一次评估一个模型）

```bash
# 微调模型（以 Exp A 为例，B/C 同理，换成对应 best 目录）
python evaluate.py --model_path output/Qwen3-1.7B-full/best
# Baseline（原始模型）
python evaluate.py --model_path models/Qwen/Qwen3-1.7B --tag baseline
```

**4. 汇总对比**（一次性评估 Baseline/A/B/C 全部 4 个并出对比表）

```bash
python compare.py                # 全量对比
python compare.py --from_cache   # 仅从已有 eval_results_*.json 汇总
```

**5. 人工抽查**

```bash
python predict.py --model_path output/Qwen3-1.7B-full/best \
                  --question "我最近血糖偏高，饮食上应该注意什么？"
```

> **LoRA 自动合并**：`evaluate.py` / `predict.py` 会自动识别 `best` 是否为 LoRA adapter（含 `adapter_config.json`），若是则加载基座并 `merge_and_unload` 合并后再评估，合并后与全量模型的推理速度口径一致。
>
> **显存提示**：评估会同时加载「被评估模型 + Qwen3-Embedding」。默认 embedder 用 **Qwen3-Embedding-0.6B**，评估 8B 时也安全（峰值 ~21GB）。仅评估 1.7B 且显存充裕时可用更大的：`EMBED_MODEL_ID=Qwen/Qwen3-Embedding-4B python compare.py`。

---

## 🗂️ 数据处理

### 数据来源

- **数据集**：`krisfu/delicate_medical_r1_data`，含 R1 风格思维链的中文医学问答，由基于 MetaGPT 的造数 workflow + QwQ 模型生成。
- **规模**：约 2490 条，固定种子 `42` 按 9:1 切分为训练集（约 2251 条）与验证集（约 249 条），可复现。
- **原始字段**：`question`（输入）、`think`（思考过程）、`answer`（结论）、`metrics`（F1）。

### 处理流程

1. **下载与切分**（`download.py`）：`random.seed(42)` → `shuffle` → 9:1 切分，`ensure_ascii=False` 保留中文。
2. **格式转换**（`dataset_jsonl_transfer`）：映射为 `{instruction, input, output}` 三元组，其中 `output = f"<think>{think}</think> \n {answer}"`；同步做**脏数据过滤**（字段缺失 / think / answer 为空即跳过）。
3. **Tokenize + Label Mask**（`process_func`）：Qwen ChatML 模板拼接，`labels` 将 system+user 段置为 `-100`，**仅在 assistant 段计算 loss**；结尾使用 **`eos_token`** 保证模型学会自然停止；超 `MAX_LENGTH=2048` 同步截断。
4. **动态 Padding**：`DataCollatorForSeq2Seq(padding=True)` 在 batch 内动态补齐。

---

## 🔬 对照实验设计

采用 Baseline + 控制变量法，确保「模型规模」与「微调方法」互相隔离、结论可归因：

| 实验 | 模型 | 微调方法 | 是否训练 | 作用 |
| --- | --- | --- | --- | --- |
| **Baseline** | Qwen3-1.7B（原始） | 不微调（zero-shot） | ❌ | 参照基准，衡量 SFT 净增益 |
| **Exp A** | Qwen3-1.7B | 全量微调 | ✅ | 主方案（A vs Baseline = 全量微调收益） |
| **Exp B** | Qwen3-1.7B | LoRA | ✅ | 与 A 同规模，隔离「微调方法」变量 |
| **Exp C** | Qwen3-8B | LoRA | ✅ | 与 B 同方法，隔离「模型规模」变量 |

> **归因逻辑**：`A vs Baseline` 量化 SFT 净收益；`A vs B` 隔离「全量 vs LoRA」；`B vs C` 隔离「1.7B vs 8B」。
>
> **Baseline 从哪来？** 并非训练产物，而是 `snapshot_download` 下载的**原始 Qwen3-1.7B**，直接用 `evaluate.py` 在同一验证集上评估。

### 关键训练配置

| 配置 | 全量微调（Exp A） | LoRA（Exp B/C） |
| --- | --- | --- |
| 学习率 | `1e-5` | `1e-4` |
| batch size / 梯度累积 | 1 / 4（等效 bs=4） | 1 / 4 |
| epoch | 3 | 3 |
| 精度 | bf16 | bf16 |
| 显存优化 | 梯度检查点 | 梯度检查点 + LoRA(r=8, α=32) |

通用：`warmup_ratio=0.05` + `cosine` 衰减 + `weight_decay=0.01` + `max_grad_norm=1.0`（梯度裁剪）；`load_best_model_at_end` 按 `eval_loss` 自动选最优 checkpoint；`seed=42` 可复现。

---

## 📈 评估指标

| 指标 | 含义 | 方向 | 计算方法 |
| --- | --- | --- | --- |
| **PPL（困惑度）** | 语言建模拟合程度 | 越低越好 | 参考答案 teacher-forcing loss 取 `exp()` |
| **格式合规率** | 是否成对且顺序正确含 `<think>...</think>` 且答案非空 | 越高越好 | 正则校验 |
| **语义相似度** | 生成答案（去 think）与参考 answer 的接近度 | 越高越好 | Qwen3-Embedding-0.6B 余弦相似度 |
| **平均延迟** | 单样本端到端生成耗时 | 越低越好 | `avg_latency_s` |
| **解码吞吐** | 总生成 token / 总耗时 | 越高越好 | `decode_throughput_tok_s`（tok/s） |

> 同时报「延迟」和「吞吐」：延迟受生成长度影响，吞吐剔除长度混淆才是模型本身速度。计时已做 `torch.cuda.synchronize()` + warmup，排除 CUDA 异步与冷启动误差。训练过程指标（train/eval loss、learning_rate、grad_norm）由 **SwanLab** 实时记录。

---

## 📊 实验结果

在验证集（约 250 条）上的对照实验结果：

| 实验(tag) | PPL↓ | 格式合规率↑ | 语义相似度↑ | 延迟/样本(s)↓ | 吞吐(tok/s)↑ | 平均生成长度 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 7.4243 | 1.0 | 0.8297 | 17.6872 | 71.54 | 1265.3 |
| expA-1.7B-full | 3.4050 | 1.0 | 0.8151 | 10.0700 | 71.30 | 718.0 |
| expB-1.7B-lora | 3.1991 | 1.0 | 0.8221 | 9.6133 | 71.27 | 685.1 |
| expC-8B-lora | **2.4504** | 1.0 | **0.8369** | 12.4015 | 55.89 | 693.1 |

**核心结论：**

- **SFT 净收益显著**：PPL 从 7.42 降至 2.45~3.41（↓ 54%~67%）；生成长度砍半、延迟下降约 43%（学会精简作答）。
- **LoRA ≥ 全量微调**（同 1.7B）：Exp B 在 PPL、语义相似度、延迟上全面略优于 Exp A —— 小数据集下全量微调有轻微过拟合，LoRA 性价比更高。
- **8B 换质量、牺牲效率**：Exp C 的 PPL 与语义相似度最佳，但吞吐下降约 22%、延迟上升约 29%。
- **选型建议**：质量最优 → **Exp C（8B LoRA）**；性价比最优 → **Exp B（1.7B LoRA）**。

> ⚠️ **两个指标陷阱**：① 格式合规率四组均为 1.0（基座已饱和，无区分度）；② 语义相似度存在「冗长度混淆」—— Baseline 因输出超长（1265 token）反而拿到虚高相似度，应与 PPL、生成长度联合解读。

---

## ⚙️ 关键配置说明

- **模型下载目录**：默认 `models/`，可用环境变量 `CACHE_DIR` 覆盖（如 `CACHE_DIR=/root/autodl-tmp/`）。
- **输出目录**：`output/<模型名>-<full|lora>/`，可用 `OUTPUT_ROOT` 覆盖，最优权重在其下 `best/`。
- **Baseline 路径**：`compare.py` 中 `baseline` 通过 `snapshot_download` 动态解析原始模型目录，如与实际不符可修改 `EXPERIMENTS`。

---