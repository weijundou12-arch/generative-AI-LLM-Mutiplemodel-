离线可跑（不依赖外网、不依赖向量库），可直接替换成真实文档

内置：

文档→chunk（重叠切分 + 元数据）

BM25 稀疏检索

向量检索（可插拔：默认 hashing embedding，若你本地有 sentence-transformers 可替换）

Hybrid + RRF 融合

MMR 多样化去冗余

简单 rerank（组合分数）

引用输出

FastAPI 服务：/index、/query

评测骨架：Recall@k / MRR（用你自己的 QA 对就能跑）

"""
week5_6_rag_finance_demo.py

一个“囊括 Week5-6 核心知识点”的最小工业级 RAG Demo（金融场景）：
- 文档：模拟年报/研报片段（你可换成真实txt/pdf解析后的文本）
- Pipeline：chunk -> BM25 + embedding -> Hybrid(RRF) -> MMR -> rerank -> answer with citations
- 提供 FastAPI：/index 重新建库，/query 询问并返回引用
- 提供 eval：Recall@k / MRR 的骨架

依赖：
    pip install fastapi uvicorn pydantic

运行：
    python week5_6_rag_finance_demo.py
然后访问：
    http://localhost:8001/docs

说明：
- 为了“离线可跑”，embedding 默认用 hashing（可替换成你自己的向量模型）
- 生产中可替换为：sentence-transformers + FAISS / Milvus / pgvector + cross-encoder rerank
"""

import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


# =========================================================
# 0) 场景数据：模拟金融文本（年报/研报）
#    你替换成真实数据时：把 docs 改为你加载后的文本列表即可
# =========================================================
RAW_DOCS = [
    {
        "doc_id": "AR_2024_ACME",
        "title": "ACME 2024 Annual Report - Management Discussion",
        "date": "2025-03-01",
        "text": """
Revenue grew by 18% year-over-year driven by the cloud segment. Gross margin improved from 41% to 44% due to better product mix.
Operating expenses increased primarily because of R&D investments. The company reported EBITDA margin of 22% in FY2024.
Risk factors include foreign exchange exposure and supplier concentration. Liquidity remains strong with cash balance of $1.2B.
"""
    },
    {
        "doc_id": "AR_2024_ACME_NOTE",
        "title": "ACME 2024 Annual Report - Notes",
        "date": "2025-03-01",
        "text": """
The company recognizes revenue when control of goods or services transfers to the customer. Subscription revenue is recognized over time.
For FY2024, the effective tax rate was 19%. Capital expenditures were $210M mainly for data centers.
Debt-to-equity ratio decreased to 0.35 compared to 0.42 in the prior year.
"""
    },
    {
        "doc_id": "RPT_Q4_2024_ACME",
        "title": "Broker Research Note - Q4 2024 ACME",
        "date": "2025-01-15",
        "text": """
We maintain a Buy rating. Q4 revenue beat consensus by 3%. Management guided FY2025 revenue growth of 12-15%.
Key upside: accelerating enterprise adoption. Key downside: pricing pressure and slower macro recovery.
We forecast FY2025 EBITDA margin at 23% assuming opex discipline.
"""
    },
]


# =========================================================
# 1) 文本清洗 & 分词（BM25/Hash Embedding 都会用）
# =========================================================
TOKEN_RE = re.compile(r"[A-Za-z0-9%$\.]+")

def normalize_text(s: str) -> str:
    # WHY：统一大小写、减少噪声，有助于稀疏检索稳定
    return re.sub(r"\s+", " ", s.strip().lower())

def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(s))


# =========================================================
# 2) Chunking（切块）
#    - chunk_size: 以“词”近似 token（demo 用）
#    - overlap: 防止信息跨块导致召回失败
# =========================================================
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    date: str
    text: str
    start_word: int
    end_word: int


def chunk_document(doc: Dict[str, str], chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    words = tokenize(doc["text"])
    chunks: List[Chunk] = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunk_words = words[i:j]
        chunk_text = " ".join(chunk_words)

        cid = f"{doc['doc_id']}::w{i}-{j}"
        chunks.append(Chunk(
            chunk_id=cid,
            doc_id=doc["doc_id"],
            title=doc["title"],
            date=doc["date"],
            text=chunk_text,
            start_word=i,
            end_word=j
        ))
        if j == len(words):
            break
        i = j - overlap  # overlap 回退
    return chunks


# =========================================================
# 3) BM25 稀疏检索（金融强关键词非常重要）
# =========================================================
class BM25:
    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(x) for x in corpus_tokens) / max(self.N, 1)

        # df / idf
        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            seen = set(doc)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

        # tf（倒排）
        self.tf: List[Dict[str, int]] = []
        for doc in corpus_tokens:
            d: Dict[str, int] = {}
            for t in doc:
                d[t] = d.get(t, 0) + 1
            self.tf.append(d)

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc_len = len(self.corpus_tokens[doc_idx])
        score = 0.0
        for t in query_tokens:
            if t not in self.tf[doc_idx]:
                continue
            tf = self.tf[doc_idx][t]
            idf = self.idf.get(t, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (tf * (self.k1 + 1) / denom)
        return score


# =========================================================
# 4) 向量检索（离线可跑：Hash Embedding）
#    WHY：
#    - 工业中应换 sentence embedding（e5/bge 等）+ 向量库
#    - 这里用 hashing 让你“跑通架构与流程”
# =========================================================
class HashEmbedding:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, tokens: List[str]) -> List[float]:
        vec = [0.0] * self.dim
        for t in tokens:
            h = hash(t) % self.dim
            vec[h] += 1.0
        # L2 normalize
        norm = math.sqrt(sum(v*v for v in vec)) + 1e-9
        return [v / norm for v in vec]

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


# =========================================================
# 5) Hybrid 融合：RRF（对分数尺度不敏感）
# =========================================================
def rrf_fusion(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """
    rank_lists: 多个检索器的“按相关性排序的 chunk_id 列表”
    返回：chunk_id -> 融合分数
    """
    score: Dict[str, float] = {}
    for rl in rank_lists:
        for r, cid in enumerate(rl):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r + 1)
    return score


# =========================================================
# 6) MMR：多样化，减少重复 chunk
# =========================================================
def mmr_select(
    query_vec: List[float],
    cands: List[str],
    vecs: Dict[str, List[float]],
    top_k: int = 5,
    lambda_: float = 0.7
) -> List[str]:
    """
    MMR = lambda*sim(query, cand) - (1-lambda)*max_sim(cand, selected)
    WHY：
    - 避免 top-k 都来自同一段/同一文档
    """
    selected: List[str] = []
    remaining = cands[:]
    while remaining and len(selected) < top_k:
        best = None
        best_score = -1e9
        for cid in remaining:
            rel = cosine(query_vec, vecs[cid])
            div = 0.0
            if selected:
                div = max(cosine(vecs[cid], vecs[sid]) for sid in selected)
            s = lambda_ * rel - (1 - lambda_) * div
            if s > best_score:
                best_score = s
                best = cid
        selected.append(best)
        remaining.remove(best)
    return selected


# =========================================================
# 7) 简易 rerank（工业中可换 cross-encoder）
# =========================================================
def heuristic_rerank(query_tokens: List[str], chunks: Dict[str, Chunk], cand_ids: List[str]) -> List[str]:
    """
    规则：query 词覆盖率更高的排前；同分按长度/日期略微偏好
    WHY：让你理解 rerank 的位置与作用
    """
    def score(cid: str) -> Tuple[float, float]:
        ctoks = set(tokenize(chunks[cid].text))
        qset = set(query_tokens)
        cover = len(qset & ctoks) / max(len(qset), 1)
        # 更近日期略微优先（demo 用）
        recency = float(chunks[cid].date.replace("-", ""))  # YYYYMMDD
        return (cover, recency)

    return sorted(cand_ids, key=lambda x: score(x), reverse=True)


# =========================================================
# 8) 生成：带引用的“可追溯回答”
#    - 离线 demo 不依赖 LLM：用 extractive summary + 引用
#    - 工业中可替换为 LLM：把 selected chunks 拼成 context 再生成
# =========================================================
def synthesize_answer(query: str, selected: List[str], chunks: Dict[str, Chunk]) -> str:
    """
    WHY：
    - 强制 grounded：答案只来自证据片段
    - 给引用：便于审计、降低幻觉
    """
    lines = []
    lines.append(f"Question: {query}")
    lines.append("Answer (grounded with citations):")

    # 简单策略：把最相关 chunk 的关键句（这里用原 chunk 文本）拼接
    for i, cid in enumerate(selected, 1):
        c = chunks[cid]
        lines.append(f"- Evidence[{i}] ({c.doc_id}, {c.date}): {c.text}")

    lines.append("\nCitations:")
    for i, cid in enumerate(selected, 1):
        c = chunks[cid]
        lines.append(f"[{i}] {c.title} | {c.doc_id} | {c.date} | chunk={c.chunk_id}")

    return "\n".join(lines)


# =========================================================
# 9) RAG Engine：把所有部件串起来
# =========================================================
class RAGEngine:
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}
        self.chunk_tokens: List[List[str]] = []
        self.chunk_ids: List[str] = []
        self.bm25: Optional[BM25] = None
        self.embedder = HashEmbedding(dim=256)
        self.vecs: Dict[str, List[float]] = {}

    def build(self, raw_docs: List[Dict[str, str]], chunk_size: int = 80, overlap: int = 20):
        # 1) chunk
        all_chunks: List[Chunk] = []
        for d in raw_docs:
            all_chunks.extend(chunk_document(d, chunk_size=chunk_size, overlap=overlap))

        # 2) index structures
        self.chunks = {c.chunk_id: c for c in all_chunks}
        self.chunk_ids = [c.chunk_id for c in all_chunks]
        self.chunk_tokens = [tokenize(self.chunks[cid].text) for cid in self.chunk_ids]

        # 3) BM25
        self.bm25 = BM25(self.chunk_tokens)

        # 4) Vector cache
        self.vecs = {cid: self.embedder.embed(tokenize(self.chunks[cid].text)) for cid in self.chunk_ids}

    def _vector_rank(self, query_tokens: List[str], top_n: int = 20) -> List[str]:
        qv = self.embedder.embed(query_tokens)
        scored = [(cid, cosine(qv, self.vecs[cid])) for cid in self.chunk_ids]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:top_n]]

    def _bm25_rank(self, query_tokens: List[str], top_n: int = 20) -> List[str]:
        assert self.bm25 is not None
        scored = [(cid, self.bm25.score(query_tokens, i)) for i, cid in enumerate(self.chunk_ids)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:top_n]]

    def query(
        self,
        query: str,
        top_k: int = 5,
        pretop: int = 30,
        use_mmr: bool = True
    ) -> Dict[str, Any]:
        t0 = time.time()
        q_tokens = tokenize(query)

        # 1) 两路检索
        bm25_rank = self._bm25_rank(q_tokens, top_n=pretop)
        vec_rank = self._vector_rank(q_tokens, top_n=pretop)

        # 2) 融合（RRF）
        fused = rrf_fusion([bm25_rank, vec_rank])
        fused_rank = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        cand_ids = [cid for cid, _ in fused_rank[:pretop]]

        # 3) 可选：MMR 多样化（减少重复）
        if use_mmr:
            qv = self.embedder.embed(q_tokens)
            cand_ids = mmr_select(qv, cand_ids, self.vecs, top_k=min(top_k*2, len(cand_ids)))

        # 4) rerank（规则版）
        reranked = heuristic_rerank(q_tokens, self.chunks, cand_ids)[:top_k]

        # 5) 生成（grounded + citations）
        answer = synthesize_answer(query, reranked, self.chunks)

        latency_ms = (time.time() - t0) * 1000.0
        return {
            "answer": answer,
            "selected": reranked,
            "latency_ms": round(latency_ms, 2),
            "debug": {
                "bm25_top": bm25_rank[:5],
                "vec_top": vec_rank[:5],
                "fused_top": [cid for cid, _ in fused_rank[:5]],
            }
        }


# =========================================================
# 10) 评测骨架：Recall@k / MRR（你用自己的 QA 对就能跑）
# =========================================================
def recall_at_k(ranked: List[str], gold: List[str], k: int) -> float:
    s = set(ranked[:k])
    g = set(gold)
    return 1.0 if len(s & g) > 0 else 0.0

def mrr(ranked: List[str], gold: List[str]) -> float:
    g = set(gold)
    for i, cid in enumerate(ranked):
        if cid in g:
            return 1.0 / (i + 1)
    return 0.0


# =========================================================
# 11) FastAPI：/index 与 /query
# =========================================================
app = FastAPI(title="Week5-6 Finance RAG Demo", version="0.1.0")
ENGINE = RAGEngine()
ENGINE.build(RAW_DOCS)

class IndexRequest(BaseModel):
    chunk_size: int = Field(default=80, ge=20, le=400)
    overlap: int = Field(default=20, ge=0, le=200)

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)
    use_mmr: bool = True

@app.post("/index")
def rebuild_index(req: IndexRequest):
    ENGINE.build(RAW_DOCS, chunk_size=req.chunk_size, overlap=req.overlap)
    return {"status": "ok", "chunks": len(ENGINE.chunk_ids), "chunk_size": req.chunk_size, "overlap": req.overlap}

@app.post("/query")
def rag_query(req: QueryRequest):
    return ENGINE.query(req.query, top_k=req.top_k, use_mmr=req.use_mmr)

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(ENGINE.chunk_ids)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("week5_6_rag_finance_demo:app", host="0.0.0.0", port=8001, reload=False)
我实现了一个金融 RAG：先对年报/研报做结构化切分（chunk+overlap+元数据），建立 BM25 与向量检索两套索引，再用 RRF 融合提高召回，
并用 MMR 减少重复证据，最后做轻量 rerank，把最相关的 chunk 送入生成模块并强制引用。
服务层用 FastAPI 提供 /index 与 /query，并保留离线评测接口（Recall@k、MRR），方便迭代优化与上线监控。
