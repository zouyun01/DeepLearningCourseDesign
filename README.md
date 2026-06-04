# GRPO × CoT 蒸馏：小型 LLM 数学推理增强实验工程

本工程实现课程设计题目：**基于思维链蒸馏与 GRPO 的小型大语言模型数学推理能力增强研究**。

核心实验路线：

| 组别 | 方法 | 说明 |
|---|---|---|
| E1 | Base | 原始 Qwen2.5 小模型直接推理 |
| E2 | CoT-SFT | 使用开源 CoT 蒸馏数据进行 LoRA 监督微调 |
| E3 | GRPO | 原始模型直接做 GRPO，用于观察“无 SFT 初始化”的 RL 效果 |
| E4 | CoT-SFT + GRPO | 先 CoT-SFT，再 GRPO，是主实验组 |
| E5 | E4 + 长度惩罚 | 在复合奖励中加入长度惩罚，分析过度推理 |

工程已包含：数据准备、SFT、GRPO、答案抽取、奖励函数、评测、错误分析、画图脚本和报告模板。

---

## 1. 推荐环境

建议在 A800 80GB 单卡上运行。使用数据盘并把 Hugging Face 缓存放到数据盘：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

建议数据盘不少于 100GB，最好 150GB。

创建环境：

```bash
conda create -n grpo-cot python=3.10 -y
conda activate grpo-cot
pip install --upgrade pip
pip install -r requirements.txt
```

> 说明：`vllm` 是可选加速包，默认代码使用 Transformers 进行评测，便于稳定复现。

---

## 2. 数据与模型

默认模型：

- 主实验：`Qwen/Qwen2.5-1.5B-Instruct`
- 调试兜底：`Qwen/Qwen2.5-0.5B-Instruct`

默认数据：

- GRPO 与最终测试：`openai/gsm8k`
- CoT-SFT 蒸馏数据：`open-r1/OpenR1-Math-220k`

你可以让脚本自动下载，也可以提前下载模型：

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct

huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct
```

如果使用本地模型路径，把配置文件里的 `model_name_or_path` 改成本地目录即可。

---

## 3. 数据准备

使用开源 CoT 数据构造 SFT 训练集，并用 GSM8K 构造 GRPO 与测试数据：

```bash
python scripts/prepare_data.py \
  --sft_dataset open-r1/OpenR1-Math-220k \
  --sft_size 1500 \
  --grpo_size 800 \
  --val_size 300 \
  --out_dir data/processed \
  --seed 42
```

时间紧时可先小规模调试：

```bash
python scripts/prepare_data.py \
  --sft_size 80 \
  --grpo_size 40 \
  --val_size 30 \
  --test_size 50 \
  --out_dir data/debug \
  --seed 42
```

---

## 4. 答案抽取单元测试

GRPO 奖励严重依赖答案抽取，训练前必须先跑测试：

```bash
python -m pytest src/test_extraction.py
```

如果服务器未安装 pytest，也可以：

```bash
python src/test_extraction.py
```

---

## 5. E1：Base 评测

```bash
python scripts/evaluate.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --data_file data/processed/gsm8k_test.jsonl \
  --output_jsonl outputs/baseline/predictions.jsonl \
  --metrics_file results/baseline_metrics.json \
  --batch_size 8 \
  --max_new_tokens 512
```

也可以用封装脚本：

```bash
python scripts/run_baseline_eval.py --config configs/eval_base_qwen_1.5b.yaml
```

---

## 6. E2：CoT-SFT

```bash
python scripts/train_sft.py --config configs/sft_qwen_1.5b.yaml
```

评测 SFT LoRA adapter：

```bash
python scripts/evaluate.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/sft_qwen_1.5b \
  --data_file data/processed/gsm8k_test.jsonl \
  --output_jsonl outputs/sft/predictions.jsonl \
  --metrics_file results/sft_metrics.json \
  --batch_size 8 \
  --max_new_tokens 512
```

为了后续 E4 更稳，建议把 SFT adapter 合并成一个模型：

```bash
python scripts/merge_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/sft_qwen_1.5b \
  --output_dir outputs/sft_qwen_1.5b_merged
```

---

## 7. E3：GRPO 单独训练

```bash
python scripts/train_grpo.py --config configs/grpo_base_r1.yaml
```

评测：

```bash
python scripts/evaluate.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/grpo_base_r1 \
  --data_file data/processed/gsm8k_test.jsonl \
  --output_jsonl outputs/grpo_base_r1/predictions.jsonl \
  --metrics_file results/grpo_base_r1_metrics.json \
  --batch_size 8 \
  --max_new_tokens 512
```

---

## 8. E4/E5：CoT-SFT + GRPO 与奖励函数消融

E4-R1：仅正确性奖励：

```bash
python scripts/train_grpo.py --config configs/grpo_sft_r1_correct.yaml
```

E4-R2：正确性 + 格式奖励：

```bash
python scripts/train_grpo.py --config configs/grpo_sft_r2_format.yaml
```

E5-R3：正确性 + 格式 + 长度惩罚：

```bash
python scripts/train_grpo.py --config configs/grpo_sft_r3_length.yaml
```

分别评测时，将 `--model_name_or_path` 指向 SFT merged 模型，将 `--adapter_path` 指向对应 GRPO 输出目录。例如：

```bash
python scripts/evaluate.py \
  --model_name_or_path outputs/sft_qwen_1.5b_merged \
  --adapter_path outputs/grpo_sft_r3_length \
  --data_file data/processed/gsm8k_test.jsonl \
  --output_jsonl outputs/grpo_sft_r3_length/predictions.jsonl \
  --metrics_file results/grpo_sft_r3_length_metrics.json \
  --batch_size 8 \
  --max_new_tokens 512
```

---

## 9. 汇总结果与画图

把多个 metrics JSON 合成 CSV：

```bash
python scripts/analyze_outputs.py \
  --prediction_files \
    outputs/baseline/predictions.jsonl \
    outputs/sft/predictions.jsonl \
    outputs/grpo_base_r1/predictions.jsonl \
    outputs/grpo_sft_r1_correct/predictions.jsonl \
    outputs/grpo_sft_r2_format/predictions.jsonl \
    outputs/grpo_sft_r3_length/predictions.jsonl \
  --labels Base CoT-SFT GRPO SFT-GRPO-R1 SFT-GRPO-R2 SFT-GRPO-R3 \
  --output_csv results/metrics_summary.csv \
  --error_csv results/error_cases.csv
```

画图：

```bash
python scripts/plot_results.py \
  --metrics_csv results/metrics_summary.csv \
  --output_dir results/figures
```

输出图包括：

- 准确率对比柱状图；
- 格式成功率对比；
- 平均输出长度对比；
- 过度推理率对比。

---

## 10. 建议运行顺序

最稳顺序：

```text
1. prepare_data.py
2. test_extraction.py
3. evaluate.py 跑 Base
4. train_sft.py
5. evaluate.py 跑 SFT
6. merge_lora.py 合并 SFT
7. train_grpo.py 跑 E3 / E4 / E5
8. evaluate.py 评测所有模型
9. analyze_outputs.py + plot_results.py
10. 人工补 error_cases.csv 中的错误类型
```

---

## 11. 注意事项

1. 先用 0.5B 和小样本跑通，再上 1.5B 正式实验。
2. GRPO 单独训练组 E3 方差较大，结果下降也正常，报告中重点分析奖励稀疏、格式破坏和输出长度变化。
3. SFT 与 GRPO 默认使用 LoRA；全参训练不是本项目主线。
4. 最终测试尽量用 GSM8K 官方完整 test split（1319 条），不要只用 300 条。
5. `max_new_tokens` 默认 512；如果输出过长或训练太慢，可先改成 256 调试。

---

## 12. 文件说明

```text
grpc-cot-reasoning/
├── configs/               # SFT/GRPO/评测配置
├── scripts/               # 可执行脚本
├── src/                   # 可复用模块
├── data/processed/        # 由 prepare_data.py 生成
├── outputs/               # 模型输出、adapter、prediction jsonl
├── results/               # 指标、图表、错误案例
└── report/                # 报告模板
```
