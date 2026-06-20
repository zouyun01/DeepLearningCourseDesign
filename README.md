# 基于 GRPO 与思维链蒸馏的小型大语言模型数学推理能力增强

> 深度学习课程设计 — 以 **Qwen2.5-1.5B-Instruct** 为基座，在 **GSM8K** 小学数学应用题上，
> 系统比较 **思维链监督微调（CoT-SFT）** 与 **GRPO 规则奖励强化学习** 两类后训练方法及其奖励函数消融，
> 并从准确率、格式规范性、答案抽取率、输出长度与错误类型等多维度诊断模型行为。

---

## 1. 项目简介

本项目围绕「小模型数学推理增强」这一问题，构建了一套**可复现**的实验工程，共六组对比实验：

| 实验 | 名称 | 说明 |
|------|------|------|
| E1 | Base | 直接评测原始 Qwen2.5-1.5B-Instruct |
| E2 | CoT-SFT | 用 GSM8K 风格蒸馏数据做 LoRA 监督微调 |
| E3 | GRPO | 从 Base 直接 GRPO（仅正确性奖励 R1） |
| E4-R1 | SFT-GRPO-R1 | SFT 合并模型 + GRPO（R1：正确性） |
| E4-R2 | SFT-GRPO-R2 | SFT 合并模型 + GRPO（R2：正确性 + 格式） |
| E5-R3 | SFT-GRPO-R3 | SFT 合并模型 + GRPO（R3：正确性 + 格式 − 长度惩罚） |

核心方法：**SFT**（思维链蒸馏的监督微调）与 **GRPO**（组相对策略优化、基于规则可验证奖励 RLVR），
训练统一使用 **LoRA** 参数高效微调。

---

## 2. 主要结果（完整 GSM8K test，1319 题）

| 方法 | 严格准确率 | 宽松准确率 | 格式成功率 | 平均输出长度(词) | 过度推理率 |
|------|:---:|:---:|:---:|:---:|:---:|
| Base | 42.38% | 48.67% | 53.22% | 139.8 | 0.0% |
| CoT-SFT | 63.08% | 63.15% | 99.17% | 107.9 | 0.7% |
| **GRPO** | **69.98%** | **70.43%** | 97.50% | 157.9 | 4.2% |
| SFT-GRPO-R1 | 64.06% | 64.06% | 99.47% | 107.8 | 0.6% |
| SFT-GRPO-R2 | 64.14% | 64.22% | 99.32% | 108.1 | 0.8% |
| SFT-GRPO-R3 | 64.75% | 64.75% | 99.24% | 109.4 | 0.7% |

**主要结论：**
1. CoT-SFT 主要收益是**输出格式规范化**（格式成功率 53%→99%，严格准确率 42%→63%）。
2. 单独 GRPO 取得**最高准确率（70%）**，但带来最长输出与最高过度推理率。
3. SFT-GRPO 系列格式最稳、输出更短，但准确率未超过单独 GRPO。
4. 所有方法的「输出长度—正确率」相关系数均为**负值**：更长推理 ≠ 更强推理。
5. Base 约 **81%** 的严格错误其实是格式/抽取问题而非推理错误，其推理能力被严重低估。

完整指标见 [results/metrics_summary.csv](results/metrics_summary.csv)，图表见 [results/figures/](results/figures/)。

---

## 3. 仓库结构

```
.
├── README.md                 本文件
├── requirements.txt          Python 依赖
├── configs/                  各实验 YAML 配置（SFT / GRPO / 评测）
├── data/                     数据目录（raw/processed/debug，大文件已 gitignore）
├── src/                      核心库
│   ├── prompts.py            ChatML prompt 构造、strip_think
│   ├── answer_extraction.py  严格/宽松答案抽取
│   ├── reward.py             R1/R2/R3 规则奖励
│   ├── metrics.py            多维指标计算
│   ├── error_analysis.py     错误案例收集
│   └── test_extraction.py    抽取单元测试
├── scripts/                  训练 / 评测 / 分析 / 出图脚本
│   ├── run_all.sh            一键跑通 E1–E5 + 汇总出图
│   ├── prepare_data.py       下载并构造 SFT/GRPO/test 数据
│   ├── train_sft.py          CoT-SFT 训练
│   ├── train_grpo.py         GRPO 训练
│   ├── merge_lora.py         合并 SFT LoRA（供 E4/E5 初始化）
│   ├── evaluate.py           评测 + 生成 predictions.jsonl
│   ├── analyze_outputs.py    汇总指标 → metrics_summary.csv
│   ├── classify_errors.py    错误类型自动归因
│   ├── plot_results.py / plot_analysis.py / plot_training_curves.py
│   ├── plot_combined_results.py   训练曲线 / 长度分析组图
│   ├── plot_style.py         统一绘图风格
│   └── plot_arch_h.py / plot_methods_combined.py / plot_formula.py  报告插图
├── outputs/                  各实验输出（predictions、trainer_state 等小文件已跟踪；
│                             模型权重 *.safetensors 已 gitignore）
├── results/                  指标 CSV、错误表、图表（svg + png）
└── docs/                     报告（docx）与交接文档 handoff.md（docx 未纳入 git）
```

---

## 4. 实验环境

- **硬件**：单卡 NVIDIA A800 80GB（显存 ≥ 24GB 即可复现，1.5B + LoRA）。
- **系统/驱动**：Linux + CUDA 12.x。
- **Python**：3.10（建议用 conda 新建独立环境）。

```bash
conda create -n dl python=3.10 -y
conda activate dl
pip install -r requirements.txt
```

主要依赖：`torch>=2.1`、`transformers>=4.45`、`trl>=0.17`、`peft>=0.12`、`datasets`、
`accelerate`、`pandas`、`numpy`、`matplotlib`、`scikit-learn`、`tensorboard`（出训练曲线用）。

---

## 5. 数据集与下载

全部数据通过 Hugging Face `datasets` 自动下载，无需手动准备：

| 用途 | 数据集 | 说明 |
|------|--------|------|
| CoT-SFT 训练 | `camel-ai/gsm8k_distilled` | GSM8K 风格思维链蒸馏数据（取 1500 训练 / 300 验证） |
| GRPO 训练 | `openai/gsm8k` (main, train) | 取 800 条做规则奖励强化学习 |
| 最终评测 | `openai/gsm8k` (main, test) | 完整 1319 题 |

一键构造数据（输出到 `data/processed/`）：

```bash
python scripts/prepare_data.py \
    --sft_dataset camel-ai/gsm8k_distilled --sft_size 1500 --val_size 300 \
    --grpo_size 800 --test_size -1 --out_dir data/processed --seed 42
```

> 基座模型 `Qwen2.5-1.5B-Instruct` 需自行下载（Hugging Face 或 ModelScope），
> 并通过 `MODEL_PATH` 指定本地路径。

---

## 6. 运行方式

### 6.1 一键复现全部实验

```bash
# 指定基座模型路径后，一次跑完 E1–E5 + 汇总出图
MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct bash scripts/run_all.sh
```

流程：`data → e1(Base) → e2(CoT-SFT) → merge(合并 LoRA) → e3(GRPO) → e4r1 / e4r2 / e5r3 → report`。
- 已完成的阶段会自动跳过；`FORCE=1` 强制重跑。
- 每个阶段日志写入 `logs/<stage>.log`。

### 6.2 只跑部分阶段

```bash
# 只跑 GRPO 相关阶段 + 汇总
STAGES="merge e3 e4r1 e4r2 e5r3 report" bash scripts/run_all.sh
```

### 6.3 单步运行（示例）

```bash
# CoT-SFT 训练
python scripts/train_sft.py --config configs/sft_qwen_1.5b.yaml
# GRPO 训练（R3：正确性 + 格式 + 长度惩罚）
python scripts/train_grpo.py --config configs/grpo_sft_r3_length.yaml
# 合并 SFT LoRA（供 GRPO 初始化）
python scripts/merge_lora.py --adapter outputs/sft_qwen_1.5b --out outputs/sft_qwen_1.5b_merged
```

### 6.4 仅重新生成指标与图表（已有 predictions 时）

```bash
PRED="outputs/base_1.5b/predictions.jsonl outputs/sft/predictions.jsonl \
      outputs/grpo_base_r1/predictions.jsonl outputs/grpo_sft_r1_correct/predictions.jsonl \
      outputs/grpo_sft_r2_format/predictions.jsonl outputs/grpo_sft_r3_length/predictions.jsonl"
LAB="Base CoT-SFT GRPO SFT-GRPO-R1 SFT-GRPO-R2 SFT-GRPO-R3"

# 汇总指标
python scripts/analyze_outputs.py --prediction_files $PRED --labels $LAB \
    --output_csv results/metrics_summary.csv --error_csv results/error_cases.csv

# 出图（柱状图 / 分析图 / 训练曲线 / 错误类型 / 组图）
python scripts/plot_results.py    --metrics_csv results/metrics_summary.csv --output_dir results/figures
python scripts/plot_analysis.py   --prediction_files $PRED --labels $LAB --fig_dir results/figures --table_dir results
python scripts/plot_training_curves.py
python scripts/classify_errors.py --error_csv results/error_cases.csv \
    --dist_csv results/error_type_distribution.csv --fig results/figures/error_type_distribution.svg
python scripts/plot_combined_results.py     # 训练过程组图 + 长度分析组图
```

> 训练曲线优先读取 `outputs/grpo_*/runs/` 下的 TensorBoard 事件；若不存在，会自动回退到
> `trainer_state.json`（已随 outputs 一并跟踪），因此**克隆仓库后即可重绘训练曲线**。

---

