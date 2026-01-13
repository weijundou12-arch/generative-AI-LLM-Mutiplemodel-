"""
week4_fastapi_llm_server.py

一个“生产思维”的 LLM 推理服务模板（FastAPI）：
- OpenAI-compatible: /v1/chat/completions
- 支持：stream(SSE)、API Key、限流、健康检查、ready、metrics、request_id 日志
- 推理后端：
    1) 优先 transformers 本地模型（MODEL_PATH 指向本地目录）
    2) 若 transformers 不可用/未配置模型，则 fallback 为 Dummy（方便联调）

运行：
    pip install fastapi uvicorn pydantic
    pip install torch transformers   # 可选：需要本地模型推理
    python week4_fastapi_llm_server.py

示例：
    export MODEL_PATH=/your/local/model
    export API_KEYS=devkey1,devkey2
    uvicorn week4_fastapi_llm_server:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Literal, Tuple

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

# -----------------------------
# 可选：transformers 后端
# WHY：生产通常用 vLLM 更强，但 Week4 先用 transformers 打通服务化
# -----------------------------
HF_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
    HF_AVAILABLE = True
except Exception:
    torch = None
    AutoTokenizer = AutoModelForCausalLM = TextIteratorStreamer = None


# -----------------------------
# 配置（环境变量驱动，部署友好）
# -----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()  # 本地模型目录（没有则 Dummy）
DEVICE = os.getenv("DEVICE", "cuda" if (HF_AVAILABLE and torch and torch.cuda.is_available()) else "cpu")
DTYPE = os.getenv("DTYPE", "auto")  # auto / float16 / bfloat16 / float32
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))  # 防止超大请求
MAX_NEW_TOKENS_LIMIT = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "512"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "256"))

API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]  # Header: Authorization: Bearer xxx
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))  # 每分钟每 key 限制（简化版）
SERVICE_NAME = os.getenv("SERVICE_NAME", "clinic-note-llm")

# -----------------------------
# 简易 metrics（生产建议用 Prometheus client；这里用最小可用）
# -----------------------------
METRICS = {
    "requests_total": 0,
    "requests_stream_total": 0,
    "errors_total": 0,
    "tokens_in_total": 0,
    "tokens_out_total": 0,
    "latency_ms_sum": 0.0,
}

# -----------------------------
# 简易限流：每 API key 记录最近一分钟的请求时间戳
# WHY：LLM 推理昂贵，必须防滥用，保证整体可用性
# -----------------------------
_REQUEST_LOG: Dict[str, List[float]] = {}


def _check_rate_limit(key: str):
    now = time.time()
    window_start = now - 60.0
    arr = _REQUEST_LOG.get(key, [])
    arr = [t for t in arr if t >= window_start]
    if len(arr) >= RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (RPM).")
    arr.append(now)
    _REQUEST_LOG[key] = arr


# -----------------------------
# 请求/响应：OpenAI-compatible（子集）
# -----------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    max_tokens: int = DEFAULT_MAX_NEW_TOKENS
    stream: bool = False
    stop: Optional[List[str]] = None


# -----------------------------
# Prompt 模板（简化 ChatML）
# WHY：生产中建议对齐你模型的官方 chat template（tokenizer.apply_chat_template）
# -----------------------------
def build_prompt(messages: List[ChatMessage]) -> str:
    sys_parts = [m.content for m in messages if m.role == "system"]
    user_parts = [m.content for m in messages if m.role == "user"]
    # 简化：system 合并为一段，user 合并为一段
    system = "\n".join(sys_parts).strip()
    user = "\n".join(user_parts).strip()

    # 一个明确的场景：生成“病历/随访记录”
    prompt = ""
    if system:
        prompt += f"[SYSTEM]\n{system}\n"
    prompt += f"[USER]\n{user}\n[ASSISTANT]\n"
    return prompt


# -----------------------------
# 后端：Dummy（无模型时也能跑通 API / 前端联调）
# -----------------------------
class DummyBackend:
    def __init__(self):
        self.name = "dummy"

    def generate(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        # 只是演示：真实部署换成 transformers/vLLM
        return (
            "Assessment: Based on the provided information, draft a structured clinical note.\n"
            "Plan: (1) Further examination. (2) Imaging if indicated. (3) Discuss risks/benefits.\n"
            "Note: This is a dummy response for integration testing.\n"
        )

    def stream_generate(self, prompt: str, max_new_tokens: int, **kwargs):
        text = self.generate(prompt, max_new_tokens, **kwargs)
        for ch in text:
            yield ch
            time.sleep(0.002)


# -----------------------------
# 后端：Transformers（本地模型）
# 支持：stream（TextIteratorStreamer）+ 非阻塞（thread）
# -----------------------------
class HFBackend:
    def __init__(self, model_path: str, device: str, dtype: str):
        assert HF_AVAILABLE, "transformers/torch not available"
        self.name = os.path.basename(model_path.rstrip("/")) or "local-model"
        self.device = device

        # dtype 选择：WHY：fp16/bf16 可显著省显存并提高吞吐
        torch_dtype = None
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None
        )
        if device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()

    def _count_tokens(self, text: str) -> int:
        # WHY：线上要统计输入/输出 token，便于限额与计费/容量规划
        return len(self.tokenizer.encode(text))

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
                 top_k: Optional[int], stop: Optional[List[str]]):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-6),
            top_p=top_p,
        )
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # 简化 stop：截断（生产建议更严格处理）
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text, self._count_tokens(prompt), self._count_tokens(text)

    def stream_generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
                        top_k: Optional[int], stop: Optional[List[str]]):
        """
        WHY：
        - 模型生成是阻塞计算，不能直接在 async 里跑
        - 用 TextIteratorStreamer + 后台线程，把 token 流式吐给 SSE
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            streamer=streamer,
        )
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        import threading

        def _run():
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        # 注意：这里 streamer 产出的是“解码后的文本片段”
        for piece in streamer:
            # stop 简化：如果出现 stop 词就结束
            if stop and any(s in piece for s in stop):
                break
            yield piece


# -----------------------------
# FastAPI app + 生命周期加载模型
# -----------------------------
app = FastAPI(title="Week4 LLM Service", version="0.1.0")

BACKEND: Any = None
READY = False


@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    """
    WHY：
    - request_id：排障/链路追踪基础
    - metrics：最小可观测性（生产建议 Prometheus + tracing）
    """
    rid = str(uuid.uuid4())
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        METRICS["errors_total"] += 1
        raise e
    finally:
        latency_ms = (time.time() - start) * 1000.0
        METRICS["latency_ms_sum"] += latency_ms
        METRICS["requests_total"] += 1

    response.headers["x-request-id"] = rid
    return response


@app.on_event("startup")
async def startup():
    """
    WHY：
    - 模型必须在启动时加载一次
    - readiness 需要在模型加载完成后才 True
    """
    global BACKEND, READY
    if HF_AVAILABLE and MODEL_PATH:
        BACKEND = HFBackend(MODEL_PATH, DEVICE, DTYPE)
    else:
        BACKEND = DummyBackend()
    READY = True


# -----------------------------
# 鉴权：Authorization: Bearer <key>
# -----------------------------
def require_api_key(authorization: Optional[str]) -> str:
    if not API_KEYS:
        return "anonymous"  # 未配置则不强制（开发模式）
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing API key (Authorization: Bearer ...)")
    key = authorization.split(" ", 1)[1].strip()
    if key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# -----------------------------
# 健康检查 / 就绪检查 / metrics
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/ready")
async def ready():
    if not READY:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "backend": getattr(BACKEND, "name", "unknown"), "device": DEVICE}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    # 简易 Prometheus exposition 风格（够用来理解概念）
    lines = []
    for k, v in METRICS.items():
        lines.append(f"# TYPE {k} counter")
        lines.append(f"{k} {v}")
    return "\n".join(lines) + "\n"


# -----------------------------
# OpenAI-compatible endpoints（子集）
# -----------------------------
@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": getattr(BACKEND, "name", "model"), "object": "model"}], "object": "list"}


def sse_event(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, authorization: Optional[str] = Header(default=None)):
    # 1) 鉴权 + 限流
    key = require_api_key(authorization)
    _check_rate_limit(key)

    # 2) 输入校验（防 OOM/滥用）
    prompt = build_prompt(req.messages)
    if len(prompt) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail="Input too large")
    max_new = min(max(req.max_tokens, 1), MAX_NEW_TOKENS_LIMIT)

    # 3) 生成（阻塞计算：stream / non-stream 分支）
    model_name = getattr(BACKEND, "name", "model")
    created = int(time.time())

    if not req.stream:
        t0 = time.time()
        # 放到线程里，避免阻塞 event loop（对 HFBackend 更重要）
        text, tin, tout = await asyncio.to_thread(
            lambda: _run_generate(prompt, max_new, req, BACKEND)
        )
        latency_ms = (time.time() - t0) * 1000.0

        METRICS["tokens_in_total"] += tin
        METRICS["tokens_out_total"] += tout

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout},
            "meta": {"latency_ms": round(latency_ms, 2), "backend": model_name},
        })

    # stream = True: SSE
    METRICS["requests_stream_total"] += 1

    async def stream_gen():
        # 先发一个 role chunk（OpenAI 风格）
        yield sse_event({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        })

        # 后台线程产出 token/piece，再 SSE 推送
        q: asyncio.Queue = asyncio.Queue()

        def producer():
            try:
                for piece in BACKEND.stream_generate(
                    prompt=prompt,
                    max_new_tokens=max_new,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    stop=req.stop,
                ):
                    q.put_nowait(("data", piece))
                q.put_nowait(("done", ""))
            except Exception as e:
                q.put_nowait(("err", str(e)))

        import threading
        threading.Thread(target=producer, daemon=True).start()

        out_text = []
        while True:
            kind, payload = await q.get()
            if kind == "data":
                out_text.append(payload)
                yield sse_event({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
                })
            elif kind == "err":
                METRICS["errors_total"] += 1
                yield sse_event({"error": payload})
                break
            else:
                # 结束标记
                yield sse_event({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                })
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


def _run_generate(prompt: str, max_new: int, req: ChatCompletionRequest, backend: Any) -> Tuple[str, int, int]:
    """
    统一 non-stream 的生成接口：
    - HFBackend 返回（全文, token_in, token_out）
    - DummyBackend 返回（全文, 估算 tokens）
    """
    if isinstance(backend, HFBackend):
        text, tin, tout = backend.generate(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop=req.stop,
        )
        # HFBackend decode 出来会包含 prompt（视模型/模板而定），这里做一个简单截断
        if "[ASSISTANT]" in text:
            text = text.split("[ASSISTANT]", 1)[-1].lstrip()
        return text, tin, tout

    # Dummy：简单 token 估算（字符数/4），仅用于联调
    text = backend.generate(prompt, max_new_tokens=max_new)
    tin = max(1, len(prompt) // 4)
    tout = max(1, len(text) // 4)
    return text, tin, tout


# -----------------------------
# 直接 python 运行（开发模式）
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("week4_fastapi_llm_server:app", host="0.0.0.0", port=8000, reload=False)


1）健康检查 / 就绪检查 / metrics
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics

2）非流式调用（OpenAI-compatible）
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer devkey1" \
  -d '{
    "messages":[
      {"role":"system","content":"You are a clinical assistant. Output in structured SOAP format."},
      {"role":"user","content":"Patient: tooth 46 pain on chewing, mild swelling, percussion positive. Draft the note."}
    ],
    "max_tokens":200,
    "temperature":0.7,
    "stream": false
  }'

3）流式输出（SSE）
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer devkey1" \
  -d '{
    "messages":[
      {"role":"system","content":"You are a clinical assistant. Output in structured SOAP format."},
      {"role":"user","content":"Post-op ACL reconstruction day 3, no fever, incision clean, ROM 0-90. Draft follow-up note."}
    ],
    "max_tokens":220,
    "temperature":0.8,
    "stream": true
  }'
