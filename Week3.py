"""
week3_llama_variants_demo.py

目标：用一个“小型 LLaMA-style Decoder-only 模型”把 Week3 核心知识点全部跑通。
场景：小语料训练一个“临床/口腔病历风格”mini LLM，然后用 prefill+KV cache decode 生成文本。

覆盖的 Week3 知识点（面试/工作/研究）：
1) LLaMA-style Block：Pre-Norm（Pre-RMSNorm） + Residual；线性层常去 bias
2) RMSNorm：比 LayerNorm 更省算子/更快（不减均值，只按 RMS 缩放）
3) RoPE：把位置编码注入到 Q/K（更契合自回归注意力与相对位置信息）
4) SwiGLU FFN：门控前馈（更强的表达/更好的训练表现）
5) GQA/MQA：通过 n_kv_heads 控制 KV 头数，降低 KV cache 显存（吞吐更高）
6) KV Cache：prefill 一次算完 prompt，decode 逐 token 生成只算新 token
7) prefill vs decode：两种性能瓶颈不同（prefill 偏算力；decode 偏访存/KV）
8) Tokenizer：生产常用 SentencePiece；教学用 char-level 让你专注结构机制

依赖：pip install torch
运行：python week3_llama_variants_demo.py
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Week1/Week3 工程基础：可复现 & device
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_sync(device: str):
    # 计时在 GPU 上要同步，否则时间不准
    if device == "cuda":
        torch.cuda.synchronize()


# ----------------------------
# 场景：玩具语料（临床/口腔病历风格）
# 你可以替换为你自己的文本段落（越多越好）
# ----------------------------
TOY_CORPUS = """
Subjective: Patient reports pain in the lower right molar region, worse on chewing.
Objective: Mild swelling. Percussion positive on tooth 46. Gingiva slightly inflamed.
Assessment: Suspected apical periodontitis. Differential: cracked tooth syndrome.
Plan: Take periapical radiograph, cold test, discuss RCT if irreversible pulpitis.

Note: Informed consent obtained. Risks discussed: bleeding, infection, nerve injury.
Medication: Ibuprofen 400mg PRN, consider antibiotic only if systemic signs present.

Subjective: Post-op day 3 after ACL reconstruction, no fever, pain controlled.
Objective: Incision clean, mild effusion, ROM 0-90, neurovascular intact.
Assessment: Normal post-operative course.
Plan: Continue rehab protocol, monitor for infection signs, follow up in 1 week.
"""


# ----------------------------
# Tokenizer（教学版 char-level）
# WHY（面试答法）：
# - Week3 真正大模型常用 SentencePiece（BPE/Unigram）；
# - 这里用 char-level 是为了把注意力放在“架构变体与 KV cache/GQA/RoPE”上
# ----------------------------
def build_char_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text]


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join([itos[i] for i in ids])


# ----------------------------
# Week3：RMSNorm
# 面试满分简答：
# - LayerNorm：减均值+除方差；RMSNorm：只按 RMS 缩放（不减均值）
# - RMSNorm 更省算子、更快，实践中稳定性也很好（LLaMA/Qwen 常用）
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # rms = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight


# ----------------------------
# Week3：RoPE（Rotary Position Embedding）
# 面试满分简答：
# - RoPE 把位置注入 Q/K，通过旋转让注意力打分体现“相对距离”
# - 通常作用在 Q/K（而不是 token embedding），更贴合注意力匹配
# ----------------------------
def build_rope_cache(max_seq_len: int, head_dim: int, base: int = 10000, device: str = "cpu"):
    """
    生成 RoPE 用的 cos/sin cache
    shape: (max_seq_len, head_dim)
    """
    assert head_dim % 2 == 0, "RoPE 通常要求 head_dim 为偶数"
    # inv_freq: (head_dim/2,)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # positions: (max_seq_len,)
    t = torch.arange(max_seq_len, device=device).float()
    # freqs: (max_seq_len, head_dim/2)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # 为了对齐偶/奇维，拼成 (max_seq_len, head_dim)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """
    把 (x0,x1,x2,x3,...) 变成 (-x1,x0,-x3,x2,...)
    这是 RoPE 里的“旋转”操作
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    x: (B, H, T, Dh)
    cos/sin: (max_seq_len, Dh)
    pos: (T,) 或 (B,T) 位置索引（这里用 (T,) 足够）
    """
    # 取出当前位置的 cos/sin，并广播到 (B,H,T,Dh)
    cos_t = cos[pos].unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh)
    sin_t = sin[pos].unsqueeze(0).unsqueeze(0)  # (1,1,T,Dh)
    return (x * cos_t) + (rotate_every_two(x) * sin_t)


# ----------------------------
# Week3：SwiGLU FFN（LLaMA 常见）
# 面试满分简答：
# - FFN 不只是“增加参数”，它提供 token-wise 非线性变换
# - SwiGLU = gate * up，其中 gate 用 SiLU/Swish，表达效率常更高
# ----------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # LLaMA 风格通常无 bias
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)  # gate 分支
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)    # up 分支
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)  # 回投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # SiLU/Swish
        up = self.w_up(x)
        return self.w_down(gate * up)   # 门控相乘


# ----------------------------
# Week3：GQA/MQA/MHA Attention + KV cache
# 关键点：
# - n_heads：Query 头数
# - n_kv_heads：Key/Value 头数（GQA：n_kv_heads < n_heads；MQA：n_kv_heads=1；MHA：相等）
# - KV cache 存的是 K/V（通常按 n_kv_heads 存，显存更省）
# ----------------------------
def make_causal_mask(T_q: int, T_k: int, device: str) -> torch.Tensor:
    """
    生成 (T_q, T_k) 的 causal mask（True=禁止关注）
    这里考虑 qlen <= klen 的通用情况：
    - 当我们有 cache 时，key 长度更长（过去+现在）
    - query 只对应当前块（例如 decode 时 T_q=1）
    """
    # 假设 query 对应 key 的最后 T_q 个位置（当前块）
    # 让第 i 个 query 只能看见 key 的 <= (T_k - T_q + i) 位置
    # 构造方式：对每个 query i，mask 掉 key 中 “未来部分”
    mask = torch.zeros(T_q, T_k, dtype=torch.bool, device=device)
    for i in range(T_q):
        allowed = (T_k - T_q + i)
        mask[i, allowed + 1 :] = True
    return mask


class GQASelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int, rope_base: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0, "GQA 要求 n_heads 能被 n_kv_heads 整除"
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.group_size = n_heads // n_kv_heads
        self.dropout = dropout

        # Q 使用 n_heads；K/V 使用 n_kv_heads（GQA 的核心）
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # RoPE cache
        self.register_buffer("cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("sin_cache", torch.empty(0), persistent=False)
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

    def _maybe_init_rope(self, device: str):
        if self.cos_cache.numel() == 0 or self.cos_cache.device.type != device:
            cos, sin = build_rope_cache(self.max_seq_len, self.head_dim, base=self.rope_base, device=device)
            self.cos_cache = cos
            self.sin_cache = sin

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: (B, T, D)
        start_pos: 当前块在全序列中的起始位置（用于 RoPE 位置索引）
        past_kv:
            K_past: (B, n_kv_heads, T_past, head_dim)
            V_past: (B, n_kv_heads, T_past, head_dim)
        """
        B, T, D = x.shape
        device = x.device.type
        self._maybe_init_rope(device)

        # 线性投影
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)      # (B, n_heads, T, hd)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv_heads, T, hd)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, n_kv_heads, T, hd)

        # RoPE：作用在 Q/K（面试高频点）
        pos = torch.arange(start_pos, start_pos + T, device=x.device)
        q = apply_rope(q, self.cos_cache, self.sin_cache, pos)
        k = apply_rope(k, self.cos_cache, self.sin_cache, pos)

        # 拼接 cache
        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=2)  # (B, n_kv_heads, T_total, hd)
            v = torch.cat([v_past, v], dim=2)

        T_total = k.size(2)

        # GQA：把 kv 头扩展到 query 头数（repeat_interleave）
        # WHY（面试答法）：KV 头更少 -> cache 更省显存；用分组共享 KV -> 吞吐更高
        k_exp = k.repeat_interleave(self.group_size, dim=1)  # (B, n_heads, T_total, hd)
        v_exp = v.repeat_interleave(self.group_size, dim=1)

        # 手写 attention（便于讲清楚 cache + GQA 的 causal 逻辑）
        # scores: (B, n_heads, T, T_total)
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask：对每个 query，只允许看见 “过去 + 当前”
        mask = make_causal_mask(T_q=T, T_k=T_total, device=x.device.type)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v_exp)  # (B, n_heads, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.wo(out)

        present_kv = (k, v) if use_cache else None
        return out, present_kv


# ----------------------------
# Week3：LLaMA-style Block（Pre-RMSNorm）
# 面试满分简答：
# - Pre-Norm 更稳（深层训练）
# - Attention/FFN 都走 residual
# - 线性层常去 bias
# ----------------------------
class LlamaBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_hidden: int,
                 max_seq_len: int, rope_base: int, dropout: float):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GQASelfAttention(dim, n_heads, n_kv_heads, max_seq_len, rope_base, dropout)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention sub-layer (Pre-Norm)
        h = self.attn_norm(x)
        attn_out, present_kv = self.attn(h, start_pos=start_pos, past_kv=past_kv, use_cache=use_cache)
        x = x + F.dropout(attn_out, p=self.dropout, training=self.training)

        # FFN sub-layer (Pre-Norm)
        h = self.ffn_norm(x)
        ffn_out = self.ffn(h)
        x = x + F.dropout(ffn_out, p=self.dropout, training=self.training)

        return x, present_kv


# ----------------------------
# Decoder-only LLaMA-style LM
# ----------------------------
class MiniLlamaLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, n_kv_heads: int,
                 ffn_hidden: int, max_seq_len: int, rope_base: int, dropout: float, tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            LlamaBlock(dim, n_heads, n_kv_heads, ffn_hidden, max_seq_len, rope_base, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # weight tying：减少参数，常见且有效（面试点）
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,               # (B,T)
        start_pos: int = 0,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        B, T = idx.shape
        assert start_pos + T <= self.max_seq_len, "超出 max_seq_len（RoPE cache 范围）"

        x = self.tok_emb(idx)  # (B,T,dim)

        new_cache = [] if use_cache else None

        for i, blk in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache is not None else None
            x, present = blk(x, start_pos=start_pos, past_kv=past, use_cache=use_cache)
            if use_cache:
                new_cache.append(present)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B,T,V)
        return logits, new_cache


# ----------------------------
# 训练数据：next-token prediction（shift）
# ----------------------------
def get_batch(data_ids: torch.Tensor, batch_size: int, block_size: int, device: str):
    N = data_ids.size(0)
    ix = torch.randint(0, N - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data_ids[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_kv_cache_bytes(
    n_layers: int,
    batch_size: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype
) -> int:
    """
    粗略估算 KV cache 占用（bytes）
    K 与 V 各一份：2 *
    """
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return 2 * n_layers * batch_size * n_kv_heads * seq_len * head_dim * bytes_per


# ----------------------------
# 生成：prefill vs decode（含 KV cache）
# ----------------------------
@torch.no_grad()
def generate(
    model: MiniLlamaLM,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    device: str,
) -> List[int]:
    model.eval()

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T_prompt)

    # -------- Prefill：一次性跑完 prompt，建立 KV cache --------
    t0 = time.time()
    maybe_sync(device)
    logits, kv = model(idx, start_pos=0, kv_cache=None, use_cache=True)
    maybe_sync(device)
    prefill_ms = (time.time() - t0) * 1000.0

    out_ids = idx[0].tolist()
    start_pos = idx.size(1)  # cache 已有的长度

    # -------- Decode：逐 token 生成，只喂最后一个 token --------
    decode_times = []
    for _ in range(max_new_tokens):
        last = torch.tensor([[out_ids[-1]]], dtype=torch.long, device=device)  # (1,1)

        t1 = time.time()
        maybe_sync(device)
        logits, kv = model(last, start_pos=start_pos, kv_cache=kv, use_cache=True)  # 增量
        maybe_sync(device)
        decode_times.append((time.time() - t1) * 1000.0)

        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            v, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, float("-inf")), next_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        out_ids.append(next_id)

        start_pos += 1
        if start_pos >= model.max_seq_len:
            break

    avg_decode_ms = sum(decode_times) / max(len(decode_times), 1)
    print(f"[perf] prefill: {prefill_ms:.2f} ms | avg decode: {avg_decode_ms:.2f} ms/token | device={device}")
    return out_ids


# ----------------------------
# 配置：通过 n_kv_heads 控制 MHA/GQA/MQA
# - MHA: n_kv_heads = n_heads
# - GQA: n_kv_heads < n_heads（常见如 8Q/2KV）
# - MQA: n_kv_heads = 1
# ----------------------------
@dataclass
class Config:
    dim: int = 192
    n_layers: int = 2
    n_heads: int = 6
    n_kv_heads: int = 2          # 改成 6= MHA；改成 1= MQA；改成 2/3= GQA
    ffn_hidden: int = 512
    max_seq_len: int = 512
    rope_base: int = 10000
    dropout: float = 0.1

    batch_size: int = 32
    block_size: int = 128
    steps: int = 400
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 80
    grad_clip: float = 1.0

    eval_interval: int = 100


def lr_schedule(step: int, base_lr: float, warmup_steps: int) -> float:
    # 面试答法：warmup 让早期训练更稳，防止随机初始化时梯度噪声造成发散
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def main():
    set_seed(42)
    device = get_device()
    print(f"[INFO] device={device}")

    stoi, itos = build_char_vocab(TOY_CORPUS)
    vocab_size = len(stoi)
    print(f"[INFO] vocab_size={vocab_size}")

    data = torch.tensor(encode(TOY_CORPUS, stoi), dtype=torch.long)
    cfg = Config()

    assert cfg.dim % cfg.n_heads == 0
    head_dim = cfg.dim // cfg.n_heads

    # KV cache 显存对比（面试点：GQA/MQA 为何省显存）
    for kvh in [cfg.n_heads, cfg.n_kv_heads, 1]:
        bytes_ = estimate_kv_cache_bytes(cfg.n_layers, 1, kvh, 256, head_dim, torch.float16)
        print(f"[KV cache rough] n_kv_heads={kvh:<2d} | seq=256 | fp16 ~ {bytes_/1024/1024:.2f} MB")

    model = MiniLlamaLM(
        vocab_size=vocab_size,
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn_hidden=cfg.ffn_hidden,
        max_seq_len=cfg.max_seq_len,
        rope_base=cfg.rope_base,
        dropout=cfg.dropout,
        tie_weights=True,
    ).to(device)

    print(f"[INFO] params={count_params(model)/1e6:.2f}M | n_heads={cfg.n_heads} | n_kv_heads={cfg.n_kv_heads} (GQA/MQA/MHA)")

    # 优化器：AdamW（工程标准）
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()

    # --- 训练（toy） ---
    for step in range(cfg.steps):
        lr = lr_schedule(step, cfg.lr, cfg.warmup_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
        logits, _ = model(x, start_pos=0, kv_cache=None, use_cache=False)

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # grad clip：防梯度爆炸/NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        if (step + 1) % 50 == 0:
            ppl = math.exp(min(loss.item(), 20))
            print(f"[train] step={step+1:4d} lr={lr:.2e} loss={loss.item():.4f} ppl~{ppl:.2f}")

        if (step + 1) % cfg.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
                logits, _ = model(x)
                vloss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
            model.train()
            print(f"[eval ] step={step+1:4d} val_loss={vloss:.4f} val_ppl~{math.exp(min(vloss,20)):.2f}")

    # --- 生成：prefill + decode（KV cache） ---
    prompt = "Subjective: "
    prompt_ids = encode(prompt, stoi)

    out_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=260,
        temperature=0.9,
        top_k=50,
        device=device,
    )

    text = decode(out_ids, itos)
    print("\n[GENERATED]\n")
    print(text)


if __name__ == "__main__":
    main()
