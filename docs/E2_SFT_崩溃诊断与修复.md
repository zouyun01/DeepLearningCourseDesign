# E2 (CoT-SFT) 崩溃诊断与修复 — 工作交接文档

> 日期：2026-06-05　·　范围：E2（CoT-SFT, Qwen2.5-1.5B + LoRA）训练崩溃的根因定位与代码修复
> 状态：✅ 代码已修复并通过离线验证；⏳ 待在服务器（A800）重跑 E2 训练 + 评测

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
