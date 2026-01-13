Week13 工程闭环：一套可复制的“推理加速实验室”

目标：你能在面试中把“KV cache/分页/连续批处理/量化/服务化监控”讲清，并且能跑一个 demo：

KV cache 显存估算器

连续批处理调度模拟器（无 GPU 也能跑）

一个最小推理服务骨架（FastAPI：prefill/decode 分离统计）

量化导出策略（给出可复制命令与注意点）

目录结构
week13_inference_accel_lab/
  kv_cache/
    kv_estimator.py
  batching/
    continuous_batching_sim.py
  serve/
    llm_infer_fastapi.py
  quant/
    quant_plan.md
  README.md

1）KV Cache 显存估算器（面试必杀：你能当场算）

保存为：kv_cache/kv_estimator.py

"""
KV Cache 显存估算器
WHY：工程上你必须在上线前算清楚：
- max_seq_len
- max_concurrency
- dtype
不然服务一定 OOM 或吞吐灾难。
"""

from dataclasses import dataclass

@dataclass
class ModelKVSpec:
    layers: int
    heads: int
    head_dim: int
    dtype_bytes: int  # fp16/bf16=2, fp32=4

def kv_bytes_per_token(spec: ModelKVSpec) -> int:
    # 每 token 每层的 KV：2 * heads * head_dim * dtype_bytes
    return 2 * spec.layers * spec.heads * spec.head_dim * spec.dtype_bytes

def kv_total_bytes(spec: ModelKVSpec, seq_len: int, concurrency: int) -> int:
    return kv_bytes_per_token(spec) * seq_len * concurrency

def pretty_gb(x: int) -> float:
    return x / (1024**3)

if __name__ == "__main__":
    # 示例：类似 32 层、32 heads、head_dim=128 的模型（只是示例）
    spec = ModelKVSpec(layers=32, heads=32, head_dim=128, dtype_bytes=2)  # bf16/fp16
    seq_len = 8192
    concurrency = 16

    per_tok = kv_bytes_per_token(spec)
    total = kv_total_bytes(spec, seq_len, concurrency)

    print("KV bytes/token:", per_tok, f"({pretty_gb(per_tok):.6f} GB)")
    print("Total KV:", f"{pretty_gb(total):.2f} GB",
          f"(seq_len={seq_len}, concurrency={concurrency})")


你面试说法（20 秒）：
“KV cache 规模线性随层数、头数、序列长度、并发增长。上线前我会先用估算器算出 KV 占用，决定最大上下文和并发，再选 paged KV 或压缩策略。”

2）连续批处理模拟器（你能讲得非常工程）

保存为：batching/continuous_batching_sim.py

"""
Continuous Batching 调度模拟（无 GPU 也能跑）
WHY：
- 线上请求流式到达，静态 batch 会造成 GPU 空转 & padding 浪费
- 连续批处理通过动态加入/退出保持 GPU 高利用率

模拟：
- 每个请求有 remaining_tokens（需要 decode 的步数）
- 每个 step，batch 最多容纳 B 个请求
- 新请求会被动态加入空位
- 统计吞吐（tokens/step）与平均等待
"""

import random
from dataclasses import dataclass, field
from typing import List, Deque
from collections import deque

@dataclass
class Req:
    rid: int
    arrive_step: int
    remaining: int
    start_step: int = -1
    finish_step: int = -1

def simulate(steps: int = 200, batch_cap: int = 8, p_arrive: float = 0.25):
    q: Deque[Req] = deque()
    active: List[Req] = []
    rid = 0
    done: List[Req] = []

    tokens_done = 0

    for t in range(steps):
        # 新请求到达
        if random.random() < p_arrive:
            rid += 1
            q.append(Req(rid=rid, arrive_step=t, remaining=random.randint(20, 120)))

        # 填充 batch 空位（连续批处理的关键）
        while len(active) < batch_cap and q:
            r = q.popleft()
            r.start_step = t
            active.append(r)

        # decode 1 step：每个 active 请求生成 1 token
        for r in list(active):
            r.remaining -= 1
            tokens_done += 1
            if r.remaining <= 0:
                r.finish_step = t
                active.remove(r)
                done.append(r)

        # 也可以在这里模拟 prefill 成本/长上下文导致每 step 更慢，这里略

    # 统计
    if done:
        avg_wait = sum((r.start_step - r.arrive_step) for r in done) / len(done)
        avg_latency = sum((r.finish_step - r.arrive_step) for r in done) / len(done)
    else:
        avg_wait = avg_latency = 0

    print(f"Done reqs={len(done)}, tokens_done={tokens_done}")
    print(f"Avg queue wait (steps): {avg_wait:.2f}")
    print(f"Avg end-to-end latency (steps): {avg_latency:.2f}")
    print(f"Throughput (tokens/step): {tokens_done/steps:.2f}")

if __name__ == "__main__":
    random.seed(7)
    simulate()


面试说法（15 秒）：
“连续批处理本质是把 decode 的 token 级循环做成动态调度：每步在 batch 中补空位，让 GPU 一直有活干，减少 padding 和空转。”

3）推理服务骨架：prefill/decode 分离统计（工程必备）

保存为：serve/llm_infer_fastapi.py

"""
FastAPI 推理服务骨架（不依赖具体推理引擎）
目的：展示你上线时会做的关键点：
- request_id/trace_id
- prefill/decode 分阶段计时
- 限流/超时/日志（可对接 Week8 审计体系）
实际生产你会把 model.generate 替换为 vLLM/TensorRT-LLM/transformers generate。
"""

import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Week13 LLM Inference Service", version="0.1")

class InferReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/infer")
def infer(req: InferReq):
    if not req.prompt or len(req.prompt) > 20000:
        raise HTTPException(status_code=400, detail="Invalid prompt")

    trace_id = str(uuid.uuid4())

    # —— Prefill（示意）：真实情况这里会跑一次大前向
    t0 = time.time()
    time.sleep(0.01)  # placeholder
    prefill_ms = (time.time() - t0) * 1000

    # —— Decode（示意）：真实情况每 token 一次迭代，但推理引擎会做融合与 batching
    t1 = time.time()
    # 这里用 sleep 模拟：token 越多 decode 越久（且随上下文增长会更慢）
    time.sleep(0.001 * min(req.max_new_tokens, 200))
    decode_ms = (time.time() - t1) * 1000

    # 输出（示例）
    text = f"[trace_id={trace_id}] output placeholder"

    return {
        "trace_id": trace_id,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(prefill_ms + decode_ms, 2),
        "text": text
    }


你面试说法（20 秒）：
“我会把推理拆成 prefill 与 decode 两段分别监控，
因为优化手段不同：prefill 看 flash-attn/GEMM 吞吐，decode 看 KV cache、paged attention、连续批处理与 speculative decoding。”

4）量化上线方案（命令与选型清单）

保存为：quant/quant_plan.md

# Quantization Plan (Week13)

## Goal
Reduce memory & bandwidth to improve throughput while keeping quality.

## Common options
- INT8 weight-only: stable, small quality drop, good first production step.
- INT4 (GPTQ/AWQ): maximal memory saving, more sensitive.

## Calibration data
Use ~100-1000 representative prompts (same domain) to calibrate.
WHY: quantization error depends on activation distribution.

## Validation checklist
- Quality: task metrics + hallucination checks
- Safety: refusal rate, policy constraints
- Perf: tokens/s, p95 latency, GPU mem

## Deployment note
- vLLM supports high-throughput serving with continuous batching + paged KV.
- For strict latency, consider TensorRT-LLM (requires more engineering).

Week13 “30 秒面试总结”（建议背下来）

推理瓶颈主要在 decode 阶段的 KV cache 读写，KV cache 规模随层数/头数/序列长度/并发线性增长。为了提高吞吐，我会用 连续批处理保持 GPU 饱和，并用 PagedAttention分页管理 KV cache 减少碎片；在模型侧用 INT8/INT4 量化降低带宽与显存，必要时用 speculative decoding减少大模型 forward 次数。上线时我会把性能拆成 prefill/decode 指标监控，并在质量不下降前提下逐步优化。
