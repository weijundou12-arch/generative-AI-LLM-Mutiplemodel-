Week 1：Python 与深度学习底层（让你能“写得出、看得懂、调得动”）

1. 张量思维与形状推理（Tensor/Shape Calculus）

  张量的维度语义：(B, T, D) 分别是什么？（batch、序列长度、特征维）
  
  矩阵乘法规则：(…, m, k) @ (…, k, n) -> (…, m, n)
  
  Broadcasting：为什么能自动扩展？什么时候会 silently 出 bug？
  
  view/reshape/transpose/permute：为什么“只是换视图”会影响后续 contiguous？
  
  einsum 思维：用指标法理解 attention 的实现和优化
  
  Why（面试常问）
  
  Transformer 的 bug 80% 来自 shape、mask 广播、transpose 错维。
  
  你能稳定推 shape，说明你能写大模型底层模块，不止会调库。

2. 数值稳定性（训练能不能跑起来的生死线）

  Softmax 为什么会溢出？（exp 爆炸）
  
  LogSumExp trick 是什么？为什么等价？
  
  为什么 cross-entropy 通常从 logits 直接算，而不是先 softmax？
  
  必备代码：稳定 softmax / logsumexp
  import numpy as np
  
  def logsumexp(x, axis=-1, keepdims=True):
      # WHY：避免 exp(x) 溢出。先减去最大值，指数不会爆
      m = np.max(x, axis=axis, keepdims=True)
      return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
  
  def softmax(x, axis=-1):
      # WHY：稳定 softmax = exp(x - max) / sum(exp(x - max))
      x = x - np.max(x, axis=axis, keepdims=True)
      ex = np.exp(x)
      return ex / np.sum(ex, axis=axis, keepdims=True)
  
  Why（重要）
  
  attention 里 softmax 每一步都在用，不稳定就会 NaN；
  
  做研究/工程，NaN 排查能力是“入门线”。

3. 深度学习的“梯度与反向传播”直觉（不用你手推全网，但要懂机制）

  链式法则在计算图里怎么走
  
  为什么会出现梯度消失/爆炸（深层复合 + 激活饱和）
  
  为什么 residual 能救深网络（给梯度开高速路）
  
  autograd 在做什么（保存中间变量/构图/反传）
  
  Why（面试常问）
  
  “你为什么选择 Pre-LN？”、“为什么 residual 有效？”都来自这里。

4. 优化器与训练稳定性（Adam/weight decay/学习率）
你必须掌握

SGD vs Adam：为什么 Adam 更“快启动”

Momentum 的物理直觉：低通滤波减少震荡

Weight decay vs L2 正则：在 Adam 下为什么要 decouple（AdamW）

学习率：为什么 warmup 对 Transformer 特别重要（早期梯度噪声大）

Why（工作必用）

你训练模型最常调的不是网络结构，而是 LR / WD / warmup / batch。

5. PyTorch 必备工程点（面试+工作）

  nn.Module / Parameter / buffer 的区别
  
  model.train() vs model.eval()（dropout/layernorm 行为差异）
  
  torch.no_grad() 为什么能省显存
  
  dtype / mixed precision（fp16/bf16）基本概念：为什么能更快
  
  seed 与可复现：为什么“完全可复现”在 GPU 上很难

6. 表示学习基础：Embedding 与相似度（为 attention 做铺垫）
你必须掌握

embedding 是什么：离散 token -> 连续向量

dot product 相似度：为什么能表示匹配/相关

cosine vs dot：scale 的影响是什么

Week 2：Transformer 手撕核心（让你“能从零写一个 mini-Transformer”）
1. NLP 到 Transformer：为什么需要 Attention
你必须掌握

RNN 的瓶颈：长依赖 + 串行计算

CNN 的局限：局部感受野堆叠成本高

Attention 的核心：并行 + 任意位置交互

Why（关键）

Transformer 的胜利不是“更聪明”，而是更适合 GPU 并行 + 长程依赖。

2. Scaled Dot-Product Attention（核心中的核心）
你必须掌握（公式+直觉）

输入：Q, K, V

权重：A = softmax(QK^T / sqrt(dk) + mask)

输出：O = A V

Why：为什么要除以 sqrt(dk)

QK^T 的方差随维度增长变大 → softmax 更尖锐 → 梯度更不稳定

除 sqrt(dk) 相当于做了一个 方差归一化

必备代码：Scaled Dot-Product Attention（含 mask）
import numpy as np

def make_causal_mask(T):
    # WHY：decoder-only LLM 不能看未来 token
    # mask=True 表示“禁止关注”
    return np.triu(np.ones((T, T), dtype=bool), k=1)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, H, T, Dh)
    K: (B, H, T, Dh)
    V: (B, H, T, Dh)
    mask: (T, T) 或可广播到 (B, H, T, T) 的布尔mask（True=禁用）
    """
    Dh = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) / np.sqrt(Dh)  # (B,H,T,T)

    if mask is not None:
        # WHY：被mask的位置设为 -inf，softmax 后权重为0
        scores = np.where(mask, -1e9, scores)

    attn = softmax(scores, axis=-1)         # (B,H,T,T)
    out = np.matmul(attn, V)                # (B,H,T,Dh)
    return out, attn

面试必问点（你要能一句话答）

mask 为什么用 -inf：为了 softmax 后变成 0 且不影响未 mask 项归一化

padding mask 和 causal mask 区别：

padding mask：不看“空位置”

causal mask：不看“未来位置”

3. Multi-Head Attention（多头注意力）
你必须掌握

为什么要多头：

单头只能在一个子空间里做匹配

多头相当于多个“不同投影空间”的并行匹配（不同关系：语法、指代、位置等）

具体做法：

线性层把 D 拆成 H 个 Dh

每个 head 独立 attention

concat 后再投影回 D

必备代码：split/combine heads
def split_heads(x, H):
    # x: (B, T, D) -> (B, H, T, Dh)
    B, T, D = x.shape
    assert D % H == 0
    Dh = D // H
    x = x.reshape(B, T, H, Dh)
    return np.transpose(x, (0, 2, 1, 3))

def combine_heads(x):
    # x: (B, H, T, Dh) -> (B, T, D)
    B, H, T, Dh = x.shape
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(B, T, H * Dh)

4. Position：为什么必须有位置编码
你必须掌握

Attention 本身对 token 顺序不敏感（Permutation-invariant）

所以必须注入位置信息：

sinusoidal（经典）

learned（可学习）

旋转位置编码 RoPE（LLaMA/Qwen 常用，Week3 深挖）

Why（面试常问）

“没有位置编码会怎样？”→ 模型只会学“词袋式”的集合关系，顺序丢失。

5. FFN（前馈网络）与非线性：为什么不是只堆 Attention
你必须掌握

FFN 在每个 token 上做非线性变换：提升表达能力

经典 Transformer FFN：Linear -> GELU/ReLU -> Linear

现代 LLM 常见：SwiGLU（Week3 讲变体）

Why

Attention 更像“信息路由/加权汇聚”，FFN 更像“局部特征变换/提取”。

6. Residual + LayerNorm：Transformer 能训起来的关键
你必须掌握

residual：让信息与梯度有捷径

layernorm：对每个 token 的特征维做归一，稳定训练

Why：为什么是 LayerNorm 不是 BatchNorm

NLP 的序列长度变化大、batch size 也可能很小

BatchNorm 依赖 batch 统计量，容易不稳定

LayerNorm 对每个样本自身归一，更适合序列模型

必备代码：LayerNorm（numpy版）
def layer_norm(x, eps=1e-5):
    """
    x: (B, T, D)
    """
    mean = x.mean(axis=-1, keepdims=True)
    var  = ((x - mean)**2).mean(axis=-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + eps)
    return xhat


真正可训练版本还需要 learnable 的 gamma/beta（PyTorch 里 nn.LayerNorm 帮你做了）。

7. Transformer Block（你必须能写出来并解释每一行）
你必须掌握

结构：x -> (LN) -> MHA -> +res -> (LN) -> FFN -> +res

Pre-LN vs Post-LN：

Pre-LN：更稳定（现代 LLM 常用）

Post-LN：原始 Transformer，用深了更难训

Why（面试）

为什么 Pre-LN 更稳定：梯度更容易通过归一化后的路径传播，深层不容易崩。

8. Decoder-only LLM：为什么 GPT/LLaMA 都是 decoder-only
你必须掌握

decoder-only + causal mask → 做 next token prediction

为什么适合大规模生成任务：统一成一个自回归目标

9. 训练目标：next-token prediction + Cross-Entropy
你必须掌握

teacher forcing：输入是 x_0..x_{t-1}，预测 x_t

label 右移（shift）

perplexity 是什么：exp(loss)

Why

这是大语言模型的“最统一、最可规模化”的训练范式。

10. 推理基础：KV Cache（Week13会深讲，这里要先懂概念）
你必须掌握

为什么生成时每步重算 attention 代价高

KV cache：缓存历史 token 的 K/V，避免重复计算

Why（工程落地）

你做部署/加速时必须懂它，否则无法解释吞吐量瓶颈。
