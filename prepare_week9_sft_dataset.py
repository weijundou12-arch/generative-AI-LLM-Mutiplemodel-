Week 9 “全覆盖项目”：金融场景 SFT + LoRA/QLoRA（可复制）

目标：把模型微调成 “金融简报助手”：
输入：问题 +（可选）证据片段
输出：结构化简报 + 明确不确定性（证据不足就拒答）

1）数据准备脚本（生成 LlamaFactory 可用的 ShareGPT 风格 JSONL）

"""
prepare_week9_sft_dataset.py
生成一个小型、可跑通的 SFT 数据集（ShareGPT 风格），用于 LlamaFactory 的 SFT/LoRA/QLoRA。

WHY（面试可说）：
- ShareGPT 风格天然支持多轮对话（system/user/assistant），更贴近 chat/instruct 模型。
- 数据里刻意加入：证据不足时的拒答样本，防止模型“硬编”。

输出：
- data/finance_sft_sharegpt.jsonl
"""

import json
import os
from typing import Dict, List


def example(system: str, user: str, assistant: str) -> Dict:
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "user", "value": user},
            {"from": "assistant", "value": assistant},
        ]
    }


def build_finance_dataset() -> List[Dict]:
    sys = (
        "You are a finance brief assistant.\n"
        "Rules:\n"
        "1) If evidence is provided, use it. If not enough evidence, say 'insufficient evidence'.\n"
        "2) Output must be structured with headings: Key Findings / Numbers / Caveats.\n"
        "3) Never invent numbers.\n"
    )

    data = []

    # 1) 有证据：抽数+报告
    user1 = (
        "Question: What is FY2024 EBITDA margin and FY2025 forecast EBITDA margin?\n"
        "Evidence:\n"
        "- FY2024 EBITDA margin: 22%\n"
        "- FY2025 forecast EBITDA margin: 23%\n"
        "Write a brief."
    )
    asst1 = (
        "Key Findings\n"
        "- EBITDA margin is expected to improve.\n\n"
        "Numbers\n"
        "- FY2024 EBITDA margin: 22%\n"
        "- FY2025 forecast EBITDA margin: 23%\n"
        "- Change: +1 percentage point\n\n"
        "Caveats\n"
        "- Forecast depends on operating expense discipline.\n"
    )
    data.append(example(sys, user1, asst1))

    # 2) 证据不足：拒答
    user2 = (
        "Question: What is the 2025 EBITDA margin?\n"
        "Evidence:\n"
        "- Revenue grew by 18% year-over-year.\n"
        "Write a brief with the margin."
    )
    asst2 = (
        "Key Findings\n"
        "- insufficient evidence\n\n"
        "Numbers\n"
        "- No EBITDA margin value is provided in the evidence.\n\n"
        "Caveats\n"
        "- Please provide the exact EBITDA margin figure or an excerpt containing it.\n"
    )
    data.append(example(sys, user2, asst2))

    # 3) 格式遵循：只输出指定字段
    user3 = (
        "Question: Summarize risk factors mentioned.\n"
        "Evidence:\n"
        "- Risk: FX volatility\n"
        "- Risk: supply chain disruptions\n"
        "Only output JSON with keys: risks (list), confidence (0-1)."
    )
    asst3 = json.dumps(
        {"risks": ["FX volatility", "supply chain disruptions"], "confidence": 0.85},
        ensure_ascii=False
    )
    data.append(example(sys, user3, asst3))

    return data


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    rows = build_finance_dataset()
    out = "data/finance_sft_sharegpt.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} samples -> {out}")
