"""
mini_gpt_week1_2.py

一个“从零到一”的 mini GPT（decoder-only Transformer）：
- 具体场景：用一小段“临床/口腔病历风格”文本做玩具训练，然后生成相似风格句子
- 覆盖 Week1–2 必备知识点（面试/工作/研究都要能讲清 why）

依赖：
    pip install torch

运行：
    python mini_gpt_week1_2.py

提示：
- 这是“复习用的教学代码”，所以注释非常多
- 模型很小、语料很小：目的不是追求效果，而是把所有核心机制跑通
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Week1：工程必备 - 可复现 & device
# -----------------------------
def set_seed(seed: int = 42):
    """
    WHY：
    - 训练可复现是科研/工程调参的底线
    - 但注意：GPU 上“完全可复现”不总是保证（cuDNN 算子、并行等）
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Week1：数值稳定性 - logsumexp / softmax（教学版）
# -----------------------------
def stable_logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    """
    WHY：
    - 直接 log(sum(exp(x))) 容易 overflow（exp(大数) -> inf）
    - trick：减去 max，保证 exp 的输入不爆
    """
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m), dim=dim, keepdim=keepdim))


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    WHY：
    - softmax 本质：exp(x)/sum(exp(x))
    - 数值稳定版：exp(x - max)/sum(exp(x - max))
    """
    x = x - torch.max(x, dim=dim, keepdim=True).values
    ex = torch.exp(x)
    return ex / torch.sum(ex, dim=dim, keepdim=True)


# -----------------------------
# 场景：玩具语料（你可以替换成自己的文本）
# -----------------------------
TOY_CORPUS = """
Subjective: Patient reports pain in the lower right molar region, worse on chewing.
Objective: Mild swelling, percussion positive on tooth 46, gingiva slightly inflamed.
Assessment: Suspected apical periodontitis. Differential: cracked tooth syndrome.
Plan: Take periapical radiograph, perform cold test, consider RCT referral if irreversible pulpitis.

Subjective: Post-op day 3 after ACL reconstruction, no fever, pain controlled.
Objective: Incision clean, mild effusion, ROM 0-90, neurovascular intact.
Assessment: Normal post-operative course.
Plan: Continue rehab protocol, monitor for infection signs, follow up in 1 week.

Note: Informed consent obtained. Discussed risks: bleeding, infection, nerve injury.
"""


# -----------------------------
# Week1：离散->连续：Tokenizer（最小实现，字符级）
# WHY：先用 char-level 最稳，能专注在 Transformer 机制
# -----------------------------
def build_char_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text]


def decode(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join([itos[i] for i in ids])


# -----------------------------
# Week2：位置编码（Sinusoidal）
# WHY：
# - Attention 本身对顺序不敏感（置换不变）
# - 必须注入位置信息，否则模型“像词袋”
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # div_term 控制不同频率：WHY：不同维度编码不同尺度的位置信息
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        self.register_buffer("pe", pe)  # buffer：不训练，但随模型迁移 device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        return: (B, T, D) 加上对应位置编码
        """
        B, T, D = x.shape
        return x + self.pe[:T].unsqueeze(0)  # (1,T,D) 广播到 (B,T,D)


# -----------------------------
# Week2：LayerNorm（可学习 gamma/beta）
# WHY：
# - NLP/序列中 batch size 可小、长度可变，BatchNorm 不稳定
# - LayerNorm 对每个 token 的特征维归一，更适合 Transformer
# -----------------------------
class MyLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * xhat + self.beta


# -----------------------------
# Week2：Mask（Causal + Padding）
# WHY：
# - causal mask：decoder-only 不许看未来 token
# - padding mask：不许关注 padding 的“空位置”
# -----------------------------
def make_causal_mask(T: int, device: str) -> torch.Tensor:
    """
    返回 (T,T) 的 bool mask：True 表示“禁止关注”
    上三角（k=1）为 True：未来位置被屏蔽
    """
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)


def make_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    token_ids: (B,T)
    返回 (B,1,1,T) 的 bool mask，便于广播到 (B,H,T,T)
    True 表示“key 位置是 padding，需要屏蔽”
    WHY 广播形状：
    - 在 attention scores: (B,H,T,T) 上加 mask
    - 让最后一维（被关注的 key 位置）屏蔽掉
    """
    return (token_ids == pad_id).unsqueeze(1).unsqueeze(1)  # (B,1,1,T)


# -----------------------------
# Week2：Scaled Dot-Product Attention（含 KV cache）
# WHY：
# - /sqrt(dk)：防止维度大时 dot-product 方差变大导致 softmax 过尖、梯度不稳
# - mask 用 -inf：softmax 后权重变 0 且不影响归一化
# - KV cache：推理生成时避免重复计算历史 K/V（工程部署基础）
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # WHY：Q/K/V 分开线性投影，而不是直接用 embedding 相似度
        # - 模型可以在不同投影空间里学习“不同类型的关系”
        # - 也是多头注意力有效的前提
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,D) -> (B,H,T,Dh)
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,T,Dh) -> (B,T,D)
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: (B,T,D)
        causal_mask: (T_total, T_total) 或 (T,T)（这里会按实际长度切）
        padding_mask: (B,1,1,T_total) 可选
        past_kv: (K_past, V_past)
            K_past: (B,H,T_past,Dh)
            V_past: (B,H,T_past,Dh)
        use_cache: 是否返回 present_kv

        返回：
            out: (B,T,D)
            present_kv: (K_total, V_total) 或 None
        """
        B, T, D = x.shape

        q = self._split_heads(self.wq(x))  # (B,H,T,Dh)
        k = self._split_heads(self.wk(x))  # (B,H,T,Dh)
        v = self._split_heads(self.wv(x))  # (B,H,T,Dh)

        # --- KV cache：把历史 K/V 拼接到当前 K/V 上 ---
        if past_kv is not None:
            k_past, v_past = past_kv
            k = torch.cat([k_past, k], dim=2)  # (B,H,T_past+T,Dh)
            v = torch.cat([v_past, v], dim=2)

        T_total = k.size(2)

        # scores: (B,H,T,T_total)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # causal mask：需要匹配 (T, T_total)
        # 取最后 T 行、前 T_total 列（生成时 T=1 也适配）
        cm = causal_mask[:T_total, :T_total].unsqueeze(0).unsqueeze(0)  # (1,1,T_total,T_total)
        cm = cm[:, :, -T:, :]  # (1,1,T,T_total)
        scores = scores.masked_fill(cm, float("-inf"))

        # padding mask：屏蔽 key 的 padding 位置
        if padding_mask is not None:
            # padding_mask: (B,1,1,T_total) -> broadcast to (B,H,T,T_total)
            scores = scores.masked_fill(padding_mask[:, :, :, :T_total], float("-inf"))

        # softmax：稳定性由 PyTorch 内部处理，但你要懂“减 max”的 why（见 stable_softmax）
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # (B,H,T,Dh)
        out = self._merge_heads(out)  # (B,T,D)
        out = self.wo(out)            # (B,T,D)

        present_kv = (k, v) if use_cache else None
        return out, present_kv


# -----------------------------
# Week2：FFN（前馈网络）
# WHY：
# - Attention 更像“路由/汇聚”，FFN 提供非线性“特征变换”
# -----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)     # WHY GELU：比 ReLU 更平滑，Transformer 常用
        x = self.drop(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Week2：Transformer Block（Pre-LN）
# WHY：
# - Pre-LN 比 Post-LN 更易训练深层：梯度更稳定（面试高频）
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = MyLayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2 = MyLayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Attention
        h = self.ln1(x)
        attn_out, present_kv = self.attn(
            h, causal_mask=causal_mask, padding_mask=padding_mask, past_kv=past_kv, use_cache=use_cache
        )
        x = x + self.drop(attn_out)  # residual：WHY 给梯度高速路

        # Pre-LN FFN
        h = self.ln2(x)
        ffn_out = self.ffn(h)
        x = x + self.drop(ffn_out)   # residual
        return x, present_kv


# -----------------------------
# Week2：Decoder-only Transformer LM（mini GPT）
# -----------------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int,
                 max_len: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln_f = MyLayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        pad_id: Optional[int] = None,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        idx: (B,T) token ids
        pad_id: 可选，用于 padding mask
        kv_cache: 每层的 (K,V)，用于增量生成
        use_cache: 是否返回新的 cache

        返回：
            logits: (B,T,V)
            new_cache: list[(K,V)] or None
        """
        device = idx.device
        B, T = idx.shape
        assert T <= self.max_len, "序列长度不能超过 max_len（位置编码范围）"

        x = self.tok_emb(idx)  # (B,T,D)
        x = self.pos_enc(x)    # 注入位置信息
        x = self.drop(x)

        causal_mask = make_causal_mask(T if kv_cache is None else (kv_cache[0][0].size(2) + T), device=device)

        padding_mask = None
        if pad_id is not None:
            padding_mask = make_padding_mask(idx, pad_id=pad_id)  # (B,1,1,T)（这里只对当前片段）
            # NOTE：教学示例：如果你真的做 padding + kv_cache，需要对 total 长度构造更完整的 mask

        new_cache = [] if use_cache else None

        for layer_i, block in enumerate(self.blocks):
            past_kv = None
            if kv_cache is not None:
                past_kv = kv_cache[layer_i]  # (K_past, V_past)
            x, present_kv = block(
                x, causal_mask=causal_mask, padding_mask=None, past_kv=past_kv, use_cache=use_cache
            )
            if use_cache:
                new_cache.append(present_kv)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)
        return logits, new_cache


# -----------------------------
# Week1/2：训练数据 - next token prediction（shift）
# WHY：
# - 输入 x[0..T-1] 预测 y[1..T]（右移一位）
# -----------------------------
def get_batch(data_ids: torch.Tensor, batch_size: int, block_size: int, device: str):
    """
    data_ids: (N,) 一整条长序列
    返回：
        x: (B,T)
        y: (B,T)
    """
    N = data_ids.size(0)
    ix = torch.randint(0, N - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data_ids[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


# -----------------------------
# Week1：优化器/学习率/训练稳定性
# - AdamW：decoupled weight decay（工程常用）
# - warmup：Transformer 常用（早期训练更稳）
# - grad clipping：防梯度爆炸
# -----------------------------
@dataclass
class TrainConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 256

    batch_size: int = 32
    block_size: int = 128
    steps: int = 600
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0

    eval_interval: int = 100


def lr_schedule(step: int, base_lr: float, warmup_steps: int):
    # WHY warmup：Transformer 早期梯度噪声大，直接大 lr 容易不稳
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


@torch.no_grad()
def estimate_loss(model: nn.Module, data_ids: torch.Tensor, cfg: TrainConfig, device: str, iters: int = 50):
    model.eval()  # WHY：eval 关闭 dropout
    losses = []
    for _ in range(iters):
        x, y = get_batch(data_ids, cfg.batch_size, cfg.block_size, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_text(
    model: MiniGPT,
    prompt: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    device: str = "cpu",
):
    """
    WHY：
    - generation 用 no_grad：省显存、加速推理
    - eval 模式：关闭 dropout，生成更稳定
    - KV cache：每步只算新 token，避免重复 attention（工程基础）
    """
    model.eval()

    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)  # (1,T)
    kv_cache = None

    for _ in range(max_new_tokens):
        # 增量生成：只喂最后一个 token（但第一次需要把 prompt 全喂进去建立 cache）
        if kv_cache is None:
            logits, kv_cache = model(idx, use_cache=True)  # (1,T,V)
            next_logits = logits[:, -1, :]
        else:
            last = idx[:, -1:]  # (1,1)
            logits, kv_cache = model(last, kv_cache=kv_cache, use_cache=True)  # (1,1,V)
            next_logits = logits[:, -1, :]

        # temperature：控制随机性（WHY：越大越“发散”，越小越“确定”）
        next_logits = next_logits / max(temperature, 1e-8)

        # top-k：简单截断采样（WHY：避免极小概率噪声 token）
        if top_k is not None:
            v, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, float("-inf")), next_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1,1)

        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist(), itos)


def main():
    set_seed(42)
    device = get_device()
    print(f"[INFO] device={device}")

    # ----- tokenizer -----
    stoi, itos = build_char_vocab(TOY_CORPUS)
    vocab_size = len(stoi)
    print(f"[INFO] vocab_size={vocab_size}")

    data = torch.tensor(encode(TOY_CORPUS, stoi), dtype=torch.long)
    cfg = TrainConfig()

    # ----- model -----
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    ).to(device)

    # ----- optimizer -----
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()

    for step in range(cfg.steps):
        lr = lr_schedule(step, cfg.lr, cfg.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(data, cfg.batch_size, cfg.block_size, device)
        logits, _ = model(x)

        # loss：next-token prediction（把 (B,T,V) 拉平到 (B*T,V)）
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # grad clipping：WHY 防止梯度爆炸，让训练更稳
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"[train] step={step+1:4d} lr={lr:.2e} loss={loss.item():.4f} ppl={math.exp(loss.item()):.2f}")

        if (step + 1) % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, data, cfg, device, iters=20)
            print(f"[eval ] step={step+1:4d} val_loss={val_loss:.4f} val_ppl={math.exp(val_loss):.2f}")

    # ----- generate -----
    prompt = "Subjective: "
    gen = generate_text(model, prompt, stoi, itos, max_new_tokens=260, temperature=0.9, top_k=50, device=device)
    print("\n[GENERATED]\n")
    print(gen)


if __name__ == "__main__":
    main()
