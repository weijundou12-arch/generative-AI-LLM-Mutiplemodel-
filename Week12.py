Week12 全覆盖工程：DeepSpeed ZeRO-2/ZeRO-3（可跑闭环）

下面给你一个“最小但工业味”的训练工程：

用 HuggingFace Transformers 训练一个小模型（你可换成更大）

一键切换 ZeRO-2 / ZeRO-3 / Offload

带 checkpointing、bf16、梯度累积、日志

训练脚本里保留关键函数（面试问到你能解释每个点的 why）

目录结构
week12_distributed_lab/
  train_ds.py
  ds_zero2_bf16.json
  ds_zero3_bf16.json
  ds_zero3_offload.json
  run_examples.sh

1）训练脚本：train_ds.py
"""
train_ds.py
DeepSpeed + Transformers 的最小训练闭环（可替换任意 causal LM）

核心覆盖点：
- torchrun 多卡启动
- DeepSpeed ZeRO 配置（外部 json）
- bf16 / gradient_accumulation / grad clipping
- activation checkpointing（可选）
- 保存/恢复 checkpoint（工程必备）
"""

import os
import argparse
from datasets import load_dataset

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="gpt2")  # 你可换 Qwen2.5/Llama 等
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--output_dir", type=str, default="saves/week12_ds")
    p.add_argument("--deepspeed", type=str, required=True, help="Path to deepspeed config json")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--use_ckpt", action="store_true", help="Enable activation checkpointing")
    return p.parse_args()

def tokenize_function(examples, tokenizer, seq_len):
    # WHY：LM 训练以 token 为基本单位；chunk 化控制序列长度 & 显存
    text = examples["text"]
    out = tokenizer(text, truncation=True, max_length=seq_len)
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Activation checkpointing：显存不够就开（用计算换显存）
    if args.use_ckpt:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        # 有些模型还需要关 cache，否则 ckpt 不生效/显存不稳定
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    ds = load_dataset(args.dataset, args.dataset_config)
    train_ds = ds["train"].map(lambda x: tokenize_function(x, tokenizer, args.seq_len), batched=True, remove_columns=["text"])
    eval_ds = ds["validation"].map(lambda x: tokenize_function(x, tokenizer, args.seq_len), batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 训练参数：把“有效 batch”交给 grad_accum，兼顾显存与稳定性
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=10,
        eval_steps=50,
        evaluation_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=torch.cuda.is_available(),   # 能 bf16 就 bf16，更稳
        deepspeed=args.deepspeed,         # 关键：接入 DeepSpeed 配置
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=args.use_ckpt,
        # 工业常用：clip 防止爆梯度
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Done. Saved to:", args.output_dir)

if __name__ == "__main__":
    main()

2）DeepSpeed 配置：ZeRO-2（推荐先用它跑通）

ds_zero2_bf16.json

{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}


**为什么先 ZeRO-2：**省掉 O+G 的冗余，显存收益大，且通信复杂度比 ZeRO-3 低，最适合“工业默认起点”。

3）DeepSpeed 配置：ZeRO-3（更省显存）

ds_zero3_bf16.json

{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,

    "reduce_scatter": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "reduce_bucket_size": 200000000,

    "stage3_param_persistence_threshold": 100000,
    "stage3_prefetch_bucket_size": 50000000
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 50
}


**为什么 ZeRO-3 更复杂：**参数也分片，训练中需要更频繁 all-gather 参数；用 bucket/prefetch/overlap 缓解。

4）DeepSpeed 配置：ZeRO-3 + CPU Offload（显存极限场景）

ds_zero3_offload.json

{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "bf16": { "enabled": true },

  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,

    "offload_param": { "device": "cpu", "pin_memory": true },
    "offload_optimizer": { "device": "cpu", "pin_memory": true },

    "reduce_scatter": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 100000000,
    "reduce_bucket_size": 100000000
  },

  "gradient_clipping": 1.0
}


**为什么 offload 可能很慢：**PCIe 带宽可能成为瓶颈；所以它是“装不下时的最后武器”。

5）运行命令（单机多卡）

run_examples.sh

#!/usr/bin/env bash
set -e

# 安装（一次）
# pip install deepspeed transformers datasets accelerate

# ZeRO-2（推荐先跑通）
torchrun --nproc_per_node=2 train_ds.py \
  --model gpt2 \
  --deepspeed ds_zero2_bf16.json \
  --output_dir saves/zero2 \
  --max_steps 100 \
  --seq_len 256 \
  --batch_size 1 \
  --grad_accum 8 \
  --use_ckpt

# ZeRO-3
torchrun --nproc_per_node=2 train_ds.py \
  --model gpt2 \
  --deepspeed ds_zero3_bf16.json \
  --output_dir saves/zero3 \
  --max_steps 100 \
  --seq_len 256 \
  --batch_size 1 \
  --grad_accum 8 \
  --use_ckpt

TP / PP（面试必须会讲 + 给你最小 Pipeline 示例）

现实里 TP 常用 Megatron-DeepSpeed 或 PyTorch 原生 TP（生态多样）。我这里给你 PP 的最小 DeepSpeed PipelineModule 示例，让你面试能“写出结构”。

DeepSpeed Pipeline Parallel 最小示例（可读、可改）

保存为：pp_minimal_pipeline.py（示例，不一定和上面 Trainer 直接拼接，属于“并行结构演示”）

"""
pp_minimal_pipeline.py
DeepSpeed Pipeline Parallel 的最小示意：
- 把模型切成多个 stage
- micro-batch 流水线传递
WHY：PP 解决“模型深度/参数太大单卡放不下”的问题
"""

import deepspeed
import torch
import torch.nn as nn
from deepspeed.pipe import PipelineModule, LayerSpec

class Embed(nn.Module):
    def __init__(self, vocab=50257, dim=768):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
    def forward(self, x):
        return self.emb(x)

class Block(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        return x + self.ff(x)

class Head(nn.Module):
    def __init__(self, dim=768, vocab=50257):
        super().__init__()
        self.lm = nn.Linear(dim, vocab)
    def forward(self, x):
        return self.lm(x)

def build_pipeline(num_blocks=8):
    layers = [LayerSpec(Embed)]
    for _ in range(num_blocks):
        layers.append(LayerSpec(Block))
    layers.append(LayerSpec(Head))
    return PipelineModule(layers=layers, num_stages=2, loss_fn=nn.CrossEntropyLoss())

def main():
    model = build_pipeline()
    # 注意：真实训练需要 deepspeed 初始化分布式与 pipeline engine
    # 这里重点是结构：layers 切 stage，micro-batch 减 bubble（靠 gradient_accum）
    print(model)

if __name__ == "__main__":
    main()


面试你怎么讲 PP：
“我会把模型按层切成多个 stage，用 micro-batch 走流水线；bubble 通过增加 micro-batch 数和 gradient accumulation 降低。”

Week12 你现在的复习/实战建议（最短路径）

先用 ZeRO-2 + bf16 + checkpointing 跑通（你能解释每个字段 why）

再切 ZeRO-3，看显存下降/吞吐变化（理解通信代价）

最后加 offload（理解带宽瓶颈）

面试准备：把 “显存四大块 + ZeRO 1/2/3 + checkpointing + TP/PP 选型逻辑” 背熟即可
