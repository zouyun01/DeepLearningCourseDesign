# E2 (CoT-SFT) 崩溃诊断与修复 — 工作交接文档

> 日期：2026-06-05　·　范围：E2（CoT-SFT, Qwen2.5-1.5B + LoRA）训练崩溃的根因定位与代码修复
> 状态：✅ 代码已修复、离线验证通过、**服务器重训+评测已完成，结果达标（见第 8 节）**

---

## 1. 问题现象

E1（Base 直推）正常，但 E2（CoT-SFT）评测结果相对 Base **大幅崩溃**：

| 指标 | E1 Base (1.5B) | E2 CoT-SFT (1.5B) | 说明 |
|---|---|---|---|
| `accuracy_strict` | **42.4%** | **3.6%** | 暴跌 |
| `format_success_rate` | 53.2% | **4.3%** | 几乎不输出 `Final Answer:` |
| `has_final_answer` | — | 57/1319 | 同上 |
| `avg_response_length_words` | 139.8 | 170.1 | 更啰嗦 |

> 已核对 `outputs/sft_qwen_1.5b_eval/predictions.jsonl`：3.6% 不是指标文件过期或统计误差，是真实结果（1319 条里仅 47 条 strict 正确）。
> 典型失败样本的 `completion` 结尾停在 `…\boxed{18`——即用 `\boxed{}` 风格作答且被 `max_new_tokens` 截断，根本没产出 `Final Answer:`。

**结论：E2 不是「推理能力下降」，而是训练与评测的协议错位，导致 LoRA adapter 被用在它从未见过的输入分布上、且训练时根本没学会按要求收尾。**

---

## 2. 根因（三重错位 + 一个放大因素）

| # | 问题 | 证据 | 后果 |
|---|---|---|---|
| 1 | **Chat 模板错位（主因）** | 训练用裸文本拼接（`train_sft.py` 旧 `dataset_text_field="text"`，无 ChatML、无 system）；评测用 `tokenizer.apply_chat_template`（`evaluate.py:62-64`） | LoRA 学到的输入分布（`Question:\n…\n\nReasoning:`）在评测时完全不出现（评测是 `<\|im_start\|>system…user…assistant`） |
| 2 | **Prompt 正文错位** | 训练 prompt = `Question:…\n\nReasoning:`；评测 user 内容 = `Question:…\n\nPlease solve the problem step by step.\nEnd your answer with: Final Answer: <number>` | 模型学的是「续写 `Reasoning:`」，而非「响应指令并以 `Final Answer:` 收尾」 |
| 3 | **序列截断丢标签** | 训练文本 token 估计：中位 918 / p90 1888 / max 3657，但 `max_seq_length=1024`，**38%+ 样本**结尾的 `Final Answer:` 被截掉 | 模型从未在完整目标上训练，学不到如何收尾 → `format_success` 仅 4.3% |
| 放大 | **蒸馏数据风格** | `camel-ai/gsm8k_distilled` 的 `reasoning_solution` 是 R1 风格：`<think>…</think>` + `\boxed{}`，又长又啰嗦 | 即便模板对齐，过长输出在评测时撞 `max_new_tokens=512` 被截断；`\boxed` 风格也与抽取协议不符 |

数据侧观察（基于 `data/debug/sft_train.jsonl` 80 条）：
- 100% 的 `reasoning` 含 `<think>` 和 `\boxed`。
- 每条 `</think>` 之后都跟着一段**简洁的分步解题**（中位 ~139 token），结构清晰、自然收尾——非常适合作为干净的 CoT 训练目标。

---

## 3. 修复方案（已落地）

核心思路：**让 SFT 的训练输入分布与评测输入分布逐字节一致，并用 completion-only 只在回答上算 loss，同时保证 `Final Answer:` 永不被截断。**

### 改动文件

#### `src/prompts.py`
- 新增 `build_sft_messages(question, reasoning, answer, strip_think=True)`：
  - 复用评测同款 `build_chat_messages()`（system + user 完全一致）；
  - assistant 段 = 清洗后的 CoT + `\n\nFinal Answer: {answer}`。
- 新增 `clean_reasoning(reasoning, strip_think=True)`：取 `</think>` 之后那段简洁解（太短则回退到去标签全文），并移除残留 `<think>/</think>` 标签。
- 新增常量 `RESPONSE_TEMPLATE = "<|im_start|>assistant\n"`（completion-only 掩码用）。
- 保留旧 `format_sft_text()` 仅作向后兼容，不再是主路径。

#### `scripts/train_sft.py`
- 训练时用 `tokenizer.apply_chat_template` **重新渲染** `text`（从 `question/reasoning/answer` 字段重建，**忽略 jsonl 里旧的 `text`**，因此**无需重跑 `prepare_data.py`**）。
- 接入 `trl.DataCollatorForCompletionOnlyLM(response_template=RESPONSE_TEMPLATE)`，做 **completion-only 掩码**（只在 assistant 回答上计算 loss，屏蔽掉 system+user 前缀）。
- 两条 TRL 版本分支（`SFTConfig` / 旧 `TrainingArguments`）都接上了该 collator。
- 训练启动时会打印首条渲染样本的前 600 字符，便于肉眼确认格式。

#### `configs/sft_qwen_1.5b.yaml` 与 `configs/sft_qwen_1.5b_local.yaml`
- `max_seq_length: 1024 → 2048`（留安全余量）。
- 新增 `strip_think: true`。

---

## 4. 离线验证（不需 GPU，已通过）

由于本机（win32）无 transformers，验证用 `outputs/sft_qwen_1.5b/tokenizer_config.json` 里**真实的 Qwen2.5 chat template** + jinja2 渲染完成。

**单样本对齐检查（5/5 通过）：**
1. ✅ 训练文本以评测 prompt 前缀**逐字节开头** → 训练分布 == 评测分布
2. ✅ response template `<|im_start|>assistant\n` 存在 → 掩码可定位
3. ✅ 结尾为 `Final Answer: 13<|im_end|>`，目标完整
4. ✅ 无 `<think>` 泄漏进训练目标
5. ✅ 文本长度 7647 → 1582 字符

**全量（80 条）检查：**
- token 估计 max **3657 → 442**，中位 262，p90 359
- 丢失 `Final Answer:` 的样本：**0**
- 超过 2048 会丢答案的样本：**0**

`py_compile src/prompts.py scripts/train_sft.py` 通过。

---

## 5. 服务器重跑步骤（A800）

> 数据无需重新准备：`data/processed/sft_train.jsonl` 已含 `question/reasoning/answer` 字段，训练脚本会即时重渲染。

```bash
# 1. 重训 SFT
python scripts/train_sft.py --config configs/sft_qwen_1.5b.yaml

# 2. 评测 SFT adapter
python scripts/evaluate.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/sft_qwen_1.5b \
  --data_file data/processed/gsm8k_test.jsonl \
  --output_jsonl outputs/sft/predictions.jsonl \
  --metrics_file results/sft_metrics.json \
  --batch_size 8 --max_new_tokens 512

# 3.（为 E4 做准备）合并 LoRA
python scripts/merge_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path outputs/sft_qwen_1.5b \
  --output_dir outputs/sft_qwen_1.5b_merged
```

---

## 6. 预期与回归判据

- `format_success_rate`：4.3% → **≥90%**
- `accuracy_strict`：3.6% → **应超过 Base 的 42.4%**（CoT-SFT 正常增益区间约 50–60%）

**若重训后仍偏低，先看 `outputs/sft/predictions.jsonl` 的 `completion`：**
- 若仍以 `\boxed` 结尾或被截断 → 说明 `DataCollatorForCompletionOnlyLM` 没匹配上 response template（个别 TRL 版本对以特殊 token 开头的子串匹配较挑剔）。
  **处理：** 把 `RESPONSE_TEMPLATE` 改为传入 token id 列表（`tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)`）再传给 collator。
- 训练日志里那条「Example rendered SFT text」也应确认前缀是 ChatML、结尾是 `Final Answer: X<|im_end|>`。

---

## 7. 对后续实验（E3/E4/E5）的影响

- E4（CoT-SFT → GRPO，主实验组）依赖一个**好的 SFT 初始化**。修复前的崩溃 adapter 会让整条主线不可信，**必须先用修复后的 E2 重训并合并**，再做 E4/E5。
- 建议（可选）：在重训 E4 前，仿照 `src/test_extraction.py` 加一个轻量单测，断言「训练渲染文本以评测 prompt 前缀开头 + 含 `Final Answer:` + 不超长」，防止格式再次漂移。

---

## 8. 修复后实测结果（2026-06-05 02:28 服务器重训 + 评测，GSM8K test 1319 条）

崩溃版 adapter 已重命名保留为 `outputs/sft_qwen_1.5b_bad_old`（及其评测 `..._eval_bad_old`）作为对照；新结果在 `results/sft_metrics.json`，预测在 `outputs/sft/predictions.jsonl`。

| 指标 | 旧（崩溃） | **新（修复后）** | Base 基线 | 判定 |
|---|---|---|---|---|
| `accuracy_strict` | 3.6% | **63.1%** | 42.4% | ✅ 超 Base +20.7pt |
| `accuracy_lenient` | 13.3% | **63.2%** | 48.7% | ✅ |
| `format_success_rate` | 4.3% | **99.2%** | 53.2% | ✅ |
| `answer_extraction_rate_strict` | 8.5% | **99.1%** | 56.3% | ✅ |
| `avg_response_length_words` | 170.1 | **107.9** | 139.8 | ✅ 比 Base 更短 |
| `overthinking_rate` | 0 | 0.7% | 0 | 正常 |

**结论：三项判据全部达成。** 另一个健康信号：strict 与 lenient 准确率几乎相等（63.07% vs 63.15%），说明答案格式已稳定，不再依赖 lenient 兜底。

### 8.1 工程上必需的额外补丁：`NumericOnlyCollator`（已合入 `train_sft.py`）
- 现象：部分 TRL 版本在 SFT 分词后仍保留源 `text` 字符串列，HF 基础 padding collator 会试图把字符串转 tensor，报 `too many dimensions 'str'`。
- 处理：用 `NumericOnlyCollator` 包一层，在交给 `DataCollatorForCompletionOnlyLM` 前剔除非数值/列表字段。重训前若换 TRL 版本，保留此包装即可。

### 8.2 已知的良性现象：输出仍含 `\boxed{}`
- 实测 1304/1319 条 `completion` 仍带 `\boxed{X}`，因为 `strip_think` 保留的 `</think>` 后简洁解本身就含 `\(\boxed{X}\)`，模型学了下来。
- **对指标无影响**：评测抽取的是最后一行 `Final Answer:`，抽取率 99.1%。
- 如需更干净输出（可选）：在 `clean_reasoning()` 里用正则去掉 `\boxed{...}` 包裹即可，不影响其余逻辑。

---

## 9. GRPO（E3/E4/E5）跑前对齐审查与预防性修复（2026-06-05）

在开跑 GRPO 前，对「rollout 采样 prompt / 奖励抽取 / 评测」三者做了一次对齐审查，目的是把 E2 同类隐患在浪费算力之前堵掉。

### 9.1 已对齐、无需改动
- **抽取逻辑单一来源**：`src/reward.py` 与 `src/metrics.py` 都调用 `answer_extraction.completion_stats`/`extract_answer_value`，奖励判对与评测判对用同一套规则，不存在「奖励算对、评测算错」的偏差。
- **completion 格式兼容**：`reward.completion_to_text` 同时处理字符串与 chat 消息列表，对不同 TRL 返回格式健壮。
- **合并模型带模板**：`merge_lora.py` 从 base 保存 tokenizer，合并目录含 chat template，E4/E5 的 GRPO 能正常套模板。

### 9.2 🔴 致命隐患（E2 同款）：GRPO rollout prompt 不套 chat template
- 根因：`prepare_data.py` 把 `prompt` 写成 `build_math_prompt()` 的**纯字符串**；TRL GRPOTrainer 只有当 prompt 是**消息列表**时才 `apply_chat_template(add_generation_prompt=True)`，纯字符串会被直接裸喂给 policy。
- 后果：E4/E5 的 policy 是 ChatML 训练出来的 SFT 合并模型，却收到裸文本 → 采样退化；且评测走 ChatML，等于在「非评测分布」上做 RL，E3/E4 极可能白跑。
- **修复（`scripts/train_grpo.py`，无需重跑 prepare_data）**：加载数据后就地把 `prompt` 列重建为 `build_chat_messages(question)`（复用评测同款函数）。这样 GRPOTrainer 会套与评测**逐字节一致**的模板。启动时打印首条 rollout prompt 便于核对。
- **离线验证**：用真实 Qwen 模板渲染，`GRPO rollout prompt == EVAL prompt` 为 `True`，且均以 `<|im_start|>assistant` 收尾。

### 9.3 🟡 次要风险：prompt 截断
- `max_prompt_length: 256` 叠加 ChatML 模板 + system prompt 后，个别长 GSM8K 题可能被截断（且截的是开头的 system/题干）。
- **修复**：四个 GRPO 配置 `max_prompt_length: 256 → 512`。

### 9.4 GRPO 重跑命令
```bash
# E3：纯 base + GRPO（correctness-only）
python scripts/train_grpo.py --config configs/grpo_base_r1.yaml
# E4-R1 / E4-R2 / E5-R3（需先有 outputs/sft_qwen_1.5b_merged）
python scripts/train_grpo.py --config configs/grpo_sft_r1_correct.yaml
python scripts/train_grpo.py --config configs/grpo_sft_r2_format.yaml
python scripts/train_grpo.py --config configs/grpo_sft_r3_length.yaml
```
评测各组时：`--model_name_or_path` 指向 base（E3）或 `outputs/sft_qwen_1.5b_merged`（E4/E5），`--adapter_path` 指向对应 GRPO 输出目录（见 README 第 7/8 节）。

> 自检建议：每次 GRPO 启动看一眼日志里打印的「Example GRPO rollout prompt」，确认是 `system/user` + 末尾 `<|im_start|>assistant`，没有再退化成裸 `Question:`。

---

## 10. 一键运行脚本 `scripts/run_all.sh`（2026-06-05）

把 E1–E5 全流程串成一个可断点续跑的脚本。

### 10.1 用法
```bash
bash scripts/run_all.sh                                   # 跑全部阶段
STAGES="merge e3 e4r1 e4r2 e5r3 report" bash scripts/run_all.sh  # 只跑 GRPO + 汇总
MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct bash scripts/run_all.sh # 覆盖基座模型路径
FORCE=1 bash scripts/run_all.sh                           # 无视"已完成"标记强制重跑
```

### 10.2 阶段与产物
顺序：`data → test → e1 → e2 → merge → e3 → e4r1 → e4r2 → e5r3 → report`
- 每个阶段日志写到 `logs/<stage>.log`（同时打印到控制台）。
- **断点续跑**：每阶段有"完成标记"（对应 metrics/模型文件），已完成则自动跳过，除非 `FORCE=1`。
- **模型路由**：E2 SFT 与 E3 GRPO 用 `--model_name_or_path "$MODEL_PATH"` 覆盖为基座；E4/E5 不覆盖，用配置里的 `outputs/sft_qwen_1.5b_merged`。为此给 `train_sft.py` / `train_grpo.py` 各加了可选 `--model_name_or_path` 覆盖参数。
- `report` 只聚合**实际存在**的预测文件，因此即便只跑了部分组也能出 CSV/图。
- 结尾打印各组 `strict / format / len` 一览。

### 10.3 关键环境变量
`MODEL_PATH`（基座模型，默认 `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct`）、`STAGES`、`FORCE`、`BATCH_SIZE`（默认 8）、`MAX_NEW_TOKENS`（默认 512）、`DATA_DIR`（默认 `data/processed`）。

> 实现注记：脚本用 `set -euo pipefail`；已修两个 `set -e` 陷阱——`eval_model` 里 E1 无 adapter 的分支、以及 `report` 聚合时对缺失文件的判断，均改成显式 `if` 以保证返回 0，避免误退出。已用「假 python」干跑校验过命令路由与引号无误。

---

## 11. 报告用分析产物与脚本（2026-06-05）

为把报告写得比参考更充实，新增 3 个分析脚本，已全部用真实数据本地验证出图。`run_all.sh` 的 `report` 阶段已自动调用 `plot_analysis.py` 与 `classify_errors.py`（`error_limit` 提到 100000 以导出全部错误）。`plot_training_curves.py` 需先把服务器的 `outputs/grpo_*/runs/` sync 到本地再单独跑。

### 11.1 三个脚本
| 脚本 | 产物 | 依赖 |
|---|---|---|
| `scripts/plot_analysis.py` | `accuracy_vs_length` / `length_distribution` / `accuracy_by_length` / `accuracy_grouped` / `radar` 共 5 图 + `results/length_stats.csv` | 仅 predictions（无需 GPU/日志） |
| `scripts/classify_errors.py` | `results/error_cases_classified.csv`（加 `auto_error_type` 列）+ `results/error_type_distribution.csv` + `error_type_distribution.png` | 仅 error_cases.csv |
| `scripts/plot_training_curves.py` | `train_reward` / `train_loss` / `train_kl` / `train_completion_length` 共 4 图 | **需 sync `outputs/grpo_*/runs/`** + `pip install tensorboard` |

> 本地运行用 conda `dl` 环境：`E:\anaconda3\envs\dl\python.exe`（已在该环境装好 tensorboard）。

### 11.2 图表清单（共 16 图 + 4 表，远超参考报告）
- **图（16）**：6 张基础柱状图（plot_results）+ 5 张深度分析图（plot_analysis）+ 1 张错误构成图 + 4 张训练曲线。
- **表（4）**：`metrics_summary.csv`（主结果）、`length_stats.csv`（长度统计）、`error_type_distribution.csv`（错误类型分布）、外加报告手写的环境/超参表。

### 11.3 错误预分类口径
`classify_errors.py` 按优先级给每条 strict 错误打一个 `auto_error_type`：
1. `格式/抽取问题(答案实际正确)`：lenient 答案==gold（非推理错）
2. `答案格式错误(缺Final Answer)`：无 Final Answer 标记
3. `过度推理/疑似截断`：`word_length >= 300`
4. `计算或推理错误(待人工细分)`：其余——**只有这一类需人工再细分**（算术/推理路径/条件理解/幻觉条件）。

### 11.4 可直接进报告的关键结论
- **错误构成（3.9）**：Base 的错误 81% 是格式/抽取问题、仅 19% 是真·推理错误 → 定量证明「Base 瓶颈在格式」；后训练后 95%+ 错误转为真实推理错误。
- **长度-准确率（3.7）**：全方法负相关；`accuracy_by_length.png` 显示越长越易错，GRPO 在每个长度段都最高、Base 下降最陡。
- **训练过程（3.8）**：R2/R3 绝对 reward 更高是因公式含格式加分项，**仅 GRPO 与 R1 同尺度可比**（均 ~0.7）；曲线噪声大且平稳，说明 GRPO 收敛快、增益有限。
- **奖励消融（3.6）**：格式奖励因 SFT 后格式已饱和而几乎无效；长度惩罚阈值 256 词、而 SFT 输出仅 ~108 词，几乎从不触发，故为「惰性」。
