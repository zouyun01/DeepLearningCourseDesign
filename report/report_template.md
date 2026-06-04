# 基于思维链蒸馏与 GRPO 的小型大语言模型数学推理能力增强研究

## 摘要

（200–300 字。写背景、方法、实验、结果和结论。）

关键词：大语言模型；数学推理；思维链蒸馏；GRPO；强化学习；后训练

---

## 1. 引言

### 1.1 研究背景

### 1.2 问题提出

### 1.3 CoT 蒸馏与 GRPO 的基本思想

### 1.4 本文研究内容

### 1.5 主要贡献

---

## 2. 研究方法

### 2.1 任务定义

### 2.2 基座模型选择

### 2.3 数据集与样本格式

### 2.4 CoT-SFT 思维链蒸馏

### 2.5 GRPO 强化学习后训练

### 2.6 奖励函数设计

核心公式：

\[
R = R_{correct} + \alpha R_{format} - \beta R_{length}
\]

### 2.7 训练行为诊断方法

---

## 3. 实验与结果分析

### 3.1 实验环境

### 3.2 实验设置

### 3.3 对比方法

### 3.4 评价指标

### 3.5 主实验结果

### 3.6 奖励函数消融实验

### 3.7 输出长度与准确率关系分析

### 3.8 训练过程分析

### 3.9 错误类型分析

### 3.10 典型案例分析

### 3.11 结果讨论

---

## 4. 结论

### 4.1 主要结论

### 4.2 不足之处

### 4.3 未来工作

---

## 5. 参考文献

[1] Guo D., et al. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948, 2025.

[2] Shao Z., et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.

[3] Cobbe K., et al. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168, 2021.

[4] Hugging Face. TRL GRPO Trainer Documentation.

[5] Wei J., et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS, 2022.

[6] Hu E. J., et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

[7] Schulman J., et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
