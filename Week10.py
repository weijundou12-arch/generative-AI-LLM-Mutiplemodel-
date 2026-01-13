Week10 实战代码：金融研究助理对齐项目（RM + PPO + DPO + GRPO）
场景设定（你面试时可以这样说）

我们要把一个“金融研究助理”对齐到：

不乱编数据（不会凭空给 PE、营收等）；

输出结构稳定（给结论、依据、风险提示）；

如果信息不足，会明确说无法确认并给“下一步如何查证”。

实现路线：

先做偏好数据（chosen/rejected），用 LLaMAFactory 的偏好数据格式；

训练 RM（stage=rm）；

用 PPO（stage=ppo）做 RLHF；

用 DPO（stage=dpo）做低成本对齐；

用 GRPO（单独脚本演示组相对优势）做 RLVR 风格训练（规则打分）。

LLaMAFactory 的 stage=rm/ppo/dpo、以及偏好数据 chosen/rejected 的格式在官方文档里有明确说明。

代码总览（文件树）
week10_rlhf_alignment_lab/
  data/
    make_preference_dataset.py
    finance_pref.json
  dataset_info.json
  configs/
    rm_lora.yaml
    ppo_lora.yaml
    dpo_lora.yaml
  grpo/
    grpo_rlvr_toy.py
  README.md

1）生成偏好数据（符合 LLaMAFactory Preference Dataset 格式）

偏好数据必须包含：instruction / input / chosen / rejected。

# data/make_preference_dataset.py
"""
生成一个“金融研究助理”偏好数据集：
- chosen: 可信、结构化、承认不确定性、给出可执行的查证路径
- rejected: 口吻肯定但乱编数据/来源
输出：finance_pref.json（Alpaca preference 格式）
"""

import json
from pathlib import Path

def build_examples():
    examples = []

    # 示例 1：用户问某公司最新财务数据，但我们不接网时应避免编造
    examples.append({
        "instruction": "你是金融研究助理。回答要：结构化（结论/依据/风险），不能编造数据；若缺信息要说明并给查证步骤。",
        "input": "请给出公司A最新一季营收、同比增速，并解释主要驱动因素。",
        "chosen": (
            "【结论】我目前无法在离线环境中确认“公司A最新一季营收/同比增速”的准确数值，因此不直接给出具体数字。\n"
            "【依据】财务指标属于时效性数据，必须以公司季报/官方公告或可信金融终端为准。\n"
            "【建议的查证步骤】\n"
            "1) 打开公司A最新 Form 10-Q/季报，定位 Income Statement 的 Revenue。\n"
            "2) 同比增速 = (本期营收-去年同期营收)/去年同期营收。\n"
            "3) 结合 Management Discussion & Analysis（MD&A）提到的产品/地区/价格/销量变化，提炼驱动因素。\n"
            "【风险提示】若仅引用二手媒体口径，可能存在口径不一致（GAAP/Non-GAAP）或更新滞后。"
        ),
        "rejected": (
            "公司A最新一季营收是 128.7 亿美元，同比 +23.4%。主要原因是云业务爆发、成本下降、AI 需求强劲。"
        )
    })

    # 示例 2：需要给结构化输出而不是“散文式”
    examples.append({
        "instruction": "你是金融研究助理。回答要：结构化（结论/依据/风险），不能编造数据；若缺信息要说明并给查证步骤。",
        "input": "请用三点总结：为什么某股票近期波动大？",
        "chosen": (
            "【结论】波动加大通常来自“信息不确定性上升 + 流动性变化 + 交易拥挤”。\n"
            "【三点拆解】\n"
            "1) 信息面：财报/指引/监管/并购传闻等事件提高未来现金流不确定性 → 风险溢价波动。\n"
            "2) 流动性：成交量、买卖价差、做市深度变化会放大价格冲击。\n"
            "3) 仓位与情绪：高杠杆/期权 Gamma/拥挤交易导致被动对冲与止损链式反应。\n"
            "【风险提示】若只看价格不看成交与期权持仓，容易把“结构性波动”误判为“基本面变化”。"
        ),
        "rejected": (
            "波动大就是因为市场情绪不好，资金乱跑，大家都很恐慌，所以涨跌都大。"
        )
    })

    return examples

def main():
    out_path = Path(__file__).parent / "finance_pref.json"
    data = build_examples()
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(data)} examples -> {out_path}")

if __name__ == "__main__":
    main()

2）在 LLaMAFactory 注册数据集（dataset_info.json 片段）

Preference dataset 需要 ranking: true，并映射 chosen/rejected 字段。

{
  "finance_pref_demo": {
    "file_name": "data/finance_pref.json",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}

3）Reward Model 训练（stage = rm）

LLaMAFactory 文档：训练 RM 时设置 stage: rm，并使用 Preference Dataset。

# configs/rm_lora.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

stage: rm
do_train: true
finetuning_type: lora
lora_target: all

dataset: finance_pref_demo
cutoff_len: 2048
max_samples: 2000
preprocessing_num_workers: 4
overwrite_cache: true

output_dir: saves/llama3-8b/lora/reward_finance
logging_steps: 10
save_steps: 200
overwrite_output_dir: true
plot_loss: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true


运行：

llamafactory-cli train configs/rm_lora.yaml

4）PPO（stage = ppo）做 RLHF

LLaMAFactory：PPO 阶段设置 stage: ppo，并指定 reward_model 路径。

# configs/ppo_lora.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

stage: ppo
do_train: true
finetuning_type: lora
lora_target: all

# 关键：接入上一步训练好的 Reward Model
reward_model: saves/llama3-8b/lora/reward_finance

dataset: finance_pref_demo
cutoff_len: 2048
max_samples: 2000
preprocessing_num_workers: 4
overwrite_cache: true

output_dir: saves/llama3-8b/lora/ppo_finance
logging_steps: 10
save_steps: 200
overwrite_output_dir: true
plot_loss: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true


运行：

llamafactory-cli train configs/ppo_lora.yaml


面试你要会解释： PPO 阶段要同时跑“策略模型生成 → RM 打分 → PPO 更新”，工程更复杂；这也是 DPO 受欢迎的原因之一。

5）DPO（stage = dpo）低成本对齐（强烈建议你做对比实验）

LLaMAFactory：DPO 设置 stage: dpo，pref_beta 和 pref_loss: sigmoid（即 DPO）。

# configs/dpo_lora.yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

stage: dpo
do_train: true
finetuning_type: lora
lora_target: all

pref_beta: 0.1
pref_loss: sigmoid   # dpo

dataset: finance_pref_demo
cutoff_len: 2048
max_samples: 2000
preprocessing_num_workers: 4
overwrite_cache: true

output_dir: saves/llama3-8b/lora/dpo_finance
logging_steps: 10
save_steps: 200
overwrite_output_dir: true
plot_loss: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true


运行：

llamafactory-cli train configs/dpo_lora.yaml

6）GRPO（RLVR 风格）——组相对优势、无 critic 的“可验证奖励”演示脚本

GRPO 最关键的点：同一 prompt 采样一组回答，baseline 用组均值，省掉 critic；DeepSeekMath 提出并在 DeepSeek-R1 等推理模型中采用。

下面脚本用“数学可验证奖励”演示 GRPO 的核心计算（你面试时讲得清楚这段就很加分）：

# grpo/grpo_rlvr_toy.py
"""
GRPO/RLVR 核心思想演示（不追求能在你本机立刻跑通大模型训练，而是把“核心函数”完整写出来）：
- 对每个 prompt 采样 K 个回答（group sampling）
- 用 verifier 给每个回答奖励 r_i（可验证：算术正确=1，否则0）
- baseline = mean(r_i)
- advantage a_i = r_i - baseline （组相对优势）
- 用 policy gradient + KL(参考策略) 做更新（这里给出损失形式与关键 logprob 计算）

你真正工程化训练 GRPO 时，一般用 OpenRLHF 等框架来做分布式与高吞吐。:contentReference[oaicite:21]{index=21}
"""

import re
import math
from dataclasses import dataclass
from typing import List, Tuple

# -------------------------
# 1) 可验证奖励：简单算术题 verifier
# -------------------------
def parse_answer(text: str) -> str:
    # 约定模型输出包含 "Answer: <number>"
    m = re.search(r"Answer:\s*([-+]?\d+(\.\d+)?)", text)
    return m.group(1) if m else ""

def reward_verifier(prompt: str, completion: str) -> float:
    """
    规则奖励：如果答案正确给 1，否则 0
    prompt 形式： "Compute 23 * 47"
    """
    nums = re.findall(r"(-?\d+)", prompt)
    if len(nums) < 2:
        return 0.0
    a, b = int(nums[0]), int(nums[1])
    gold = a * b
    ans = parse_answer(completion)
    if ans == "":
        return 0.0
    try:
        pred = float(ans)
    except:
        return 0.0
    return 1.0 if abs(pred - gold) < 1e-9 else 0.0

# -------------------------
# 2) GRPO 核心：组内 baseline & advantage
# -------------------------
def group_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO 的关键：baseline 直接用组均值（省 critic）
    """
    if not rewards:
        return []
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]

# -------------------------
# 3) 训练时需要的 logprob / KL（这里写成接口，真正实现依赖 HF 模型）
# -------------------------
@dataclass
class LogProbPack:
    logp_pi: float       # 当前策略对该 completion 的 logprob
    logp_ref: float      # 参考策略对该 completion 的 logprob

def kl_term(lp: LogProbPack) -> float:
    # token 级 KL 更常见；这里用序列级近似演示
    return (lp.logp_pi - lp.logp_ref)

def grpo_loss_for_group(
    logprobs: List[LogProbPack],
    advantages: List[float],
    kl_coef: float = 0.02
) -> float:
    """
    一个最小可解释版本的 GRPO loss（示意）：
      maximize  E[ A_i * log pi(y_i|x)  - kl_coef * KL(pi || ref) ]
    训练中通常还会加入 clip（类似 PPO）来限制更新幅度。:contentReference[oaicite:22]{index=22}
    """
    assert len(logprobs) == len(advantages)
    total = 0.0
    for lp, adv in zip(logprobs, advantages):
        total += -(adv * lp.logp_pi) + kl_coef * kl_term(lp)
    return total / max(1, len(logprobs))

# -------------------------
# 4) 演示：对一个 prompt 采样 K 个 completion，算奖励与优势
# -------------------------
def demo():
    prompt = "Compute 23 * 47"

    # 假设采样 K=4 个回答（真实训练这里来自模型 generate）
    completions = [
        "Let’s calculate. Answer: 1081",
        "Answer: 1000",
        "We get 23*47=1081. Answer: 1081",
        "Answer: 1090"
    ]

    rewards = [reward_verifier(prompt, c) for c in completions]
    advs = group_advantages(rewards)

    # 假设我们已经从模型算出了 logp（这里只做演示填值）
    lps = [
        LogProbPack(logp_pi=-12.3, logp_ref=-12.6),
        LogProbPack(logp_pi=-10.1, logp_ref=-10.0),
        LogProbPack(logp_pi=-11.0, logp_ref=-11.4),
        LogProbPack(logp_pi=-9.8,  logp_ref=-9.7),
    ]

    loss = grpo_loss_for_group(lps, advs, kl_coef=0.02)

    print("rewards:", rewards)
    print("advantages:", advs)
    print("grpo_loss:", loss)

if __name__ == "__main__":
    demo()

7）README（你跑实验/写项目汇报直接用）
# Week10 RLHF Alignment Lab (PPO / DPO / GRPO)

## Goal
Align a "financial research assistant" to be:
- factual (no hallucinated numbers)
- structured (Conclusion/Evidence/Risks)
- explicit about uncertainty + gives verification steps

## 1. Build preference dataset
python data/make_preference_dataset.py

## 2. Register dataset
Edit dataset_info.json to include `finance_pref_demo`

## 3. Train Reward Model (RM)
llamafactory-cli train configs/rm_lora.yaml

## 4. RLHF with PPO
llamafactory-cli train configs/ppo_lora.yaml

## 5. Preference tuning with DPO (baseline compare)
llamafactory-cli train configs/dpo_lora.yaml

## 6. GRPO (toy RLVR demonstration)
python grpo/grpo_rlvr_toy.py

RLHF：SFT 先让模型可用；用偏好数据训练 RM；再用 PPO + KL 做策略优化，避免漂移与 reward hacking。
DPO：不用 RM、不用 PPO，直接用偏好对做稳定对齐（更便宜更简单）。
GRPO：PPO 变体，去 critic，用组内均值做 baseline，特别适合 可验证 reward（RLVR） 的推理任务；DeepSeekMath / DeepSeek-R1 使用过。
