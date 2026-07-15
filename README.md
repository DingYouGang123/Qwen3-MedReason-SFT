# 基于监督微调的医学推理对话大模型

基于开源数据集 `delicate_medical_r1_data` 对 **Qwen3** 系列模型进行监督微调（SFT），
让模型学会「先思考（`<think>...</think>`）后作答」的结构化医学问答能力，并通过对照实验
（全量微调 vs LoRA、不同模型规模）评估微调收益。

> 详细的技术方案、参数取舍与评估设计见 [项目计划.md](项目计划.md)。

---

## 目录结构

```
Qwen3_SFT/
├── download.py       # 下载数据集并按 9:1 切分 train/val
├── train.py              # 训练脚本（支持全量微调 / LoRA，环境变量可切换）
├── evaluate.py           # 验证集离线定量评估（PPL / 格式合规率 / 语义相似度）
├── compare.py            # 一键评估 Baseline/A/B/C 并汇总对比表
├── predict.py            # 单条推理 / 人工抽查
├── run_all.sh            # 一键流水线：数据 -> 训练三组 -> 评估对比
├── 项目计划.md            # 完整技术方案文档
└── README.md
```

---

## 环境要求

- **硬件**：带 GPU 的 Linux 环境（如 AutoDL）；1.7B 全量微调建议显存 ≥ 24GB。
- **Python**：3.10+
- **依赖安装**：

```bash
pip install torch transformers datasets modelscope swanlab pandas
pip install peft                      # LoRA 微调（Exp B/C）需要
pip install sentence-transformers     # 语义相似度评估需要（Qwen3-Embedding，需 sentence-transformers>=2.7.0；缺失会自动跳过该指标）
```

- **SwanLab**：首次使用需登录 `swanlab login`（训练监控）。

---

## 对照实验设计

| 实验 | 模型 | 微调方法 | 是否训练 | 作用 |
| --- | --- | --- | --- | --- |
| Baseline | Qwen3-1.7B（原始） | 不微调（zero-shot） | ❌ 直接评估原始模型 | 参照基准，衡量 SFT 的净增益 |
| Exp A | Qwen3-1.7B | 全量微调 | ✅ | 主方案（A vs Baseline = 全量微调收益） |
| Exp B | Qwen3-1.7B | LoRA | ✅ | 与 A 同规模，隔离「微调方法」变量 |
| Exp C | Qwen3-8B | LoRA | ✅ | 与 B 同方法，隔离「模型规模」变量 |

> **Baseline 从哪来？** 它不是训练出来的，而是 `snapshot_download` 下载的**原始 Qwen3-1.7B**
> （即 Exp A/B 微调前的样子），直接用 `evaluate.py` 在同一验证集上评估。有了它，才能量化
> A/B/C 相对原始模型「提升了多少」。
>
> 关键参数：全量微调 `lr=1e-5`、LoRA `lr=1e-4`；均使用 warmup + cosine 衰减、
> 梯度检查点、bf16、`load_best_model_at_end` 自动选最优 checkpoint。

---

## 快速开始

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
USE_LORA=true  MODEL_ID="Qwen/Qwen3-8B"  python train.py   # Exp C
```

训练结束，最优权重保存到 `/root/autodl-tmp/output/<run_name>/best`。

**3. 评估单个模型**（`evaluate.py` 一次只评估一个模型，下面仅为两种路径写法示例）

```bash
# 微调模型（以 Exp A 为例，B/C 同理，换成对应 best 目录即可）
python evaluate.py --model_path /root/autodl-tmp/output/Qwen3-1.7B-full/best
# Baseline（原始模型）
python evaluate.py --model_path /root/autodl-tmp/Qwen/Qwen3-1.7B --tag baseline
```

> 要一次性评估 **Baseline/A/B/C 全部 4 个** 并出对比表，用下一步的 `compare.py`，无需逐个手动跑。

**4. 汇总对比**（编辑 `compare.py` 顶部 `EXPERIMENTS` 路径后执行）

```bash
python compare.py                # 全量对比
python compare.py --from_cache   # 仅从已有 eval_results_*.json 汇总
```

> **LoRA 自动合并**：`evaluate.py` / `predict.py` 会自动识别 `best` 目录是否为 LoRA
> adapter（含 `adapter_config.json`），若是则加载基座并 `merge_and_unload` 合并后再评估，
> 无需手动处理；合并后 LoRA 与全量模型的推理速度口径一致。
>
> **显存提示**：评估会同时加载「被评估模型 + Qwen3-Embedding」，且 `SentenceTransformer`
> 默认以 fp32 加载。默认 embedder 用 **Qwen3-Embedding-0.6B**，评估 8B 时也安全（峰值 ~21GB）。
> 若仅评估 1.7B 且显存充裕，可用更大的：`EMBED_MODEL_ID=Qwen/Qwen3-Embedding-4B python compare.py`。

**5. 人工抽查**

```bash
python predict.py --model_path /root/autodl-tmp/output/Qwen3-1.7B-full/best \
                  --question "我最近血糖偏高，饮食上应该注意什么？"
```

---

## 评估指标

| 指标 | 含义 | 方向 |
| --- | --- | --- |
| **PPL（困惑度）** | 语言建模拟合程度（由 `eval_loss` 换算） | 越低越好 |
| **格式合规率** | 输出是否成对且顺序正确包含 `<think>...</think>` 且答案非空 | 越高越好 |
| **语义相似度** | 生成答案（去 think 段）与参考 answer 的 Qwen3-Embedding 余弦相似度 | 越高越好 |
| **平均延迟** | 单样本端到端生成耗时（`avg_latency_s`） | 越低越好 |
| **解码吞吐** | 总生成 token 数 / 总耗时（`decode_throughput_tok_s`，tok/s） | 越高越好 |

> 推理时间同时报"延迟"和"吞吐"：延迟是端到端体感，会受生成长度影响；吞吐（tok/s）
> 剔除长度混淆，才是模型本身的速度。计时已做 `torch.cuda.synchronize()` + warmup，
> 排除 CUDA 异步与冷启动误差。用于对比 1.7B/8B、全量/LoRA 的**质量-延迟性价比**。

训练过程指标（train/eval loss、learning_rate、grad_norm）由 **SwanLab** 实时记录。

---

## 关键配置说明

- **模型下载目录**：`train.py` 中 `CACHE_DIR`（默认 `/root/autodl-tmp/`），也可用环境变量 `CACHE_DIR` 覆盖。
- **输出目录**：`/root/autodl-tmp/output/<模型名>-<full|lora>/`，最优权重在其下 `best/`。
- **Baseline 路径**：`compare.py` 中 `baseline` 指向 `snapshot_download` 落盘的原始模型目录，
  首次运行 `train.py` 后即存在，如与实际不符请修改 `EXPERIMENTS`。

---

## 数据说明

- **来源**：`krisfu/delicate_medical_r1_data`（ModelScope），含 R1 风格思维链的中文医学问答。
- **字段**：`question`（输入）、`think`（思考过程）、`answer`（结论）、`metrics`（F1）。
- **切分**：固定随机种子 `42`，按 9:1 划分训练/验证集，可复现。
- **训练目标格式**：`<think>{think}</think> \n {answer}`，仅在 assistant 段计算 loss。
