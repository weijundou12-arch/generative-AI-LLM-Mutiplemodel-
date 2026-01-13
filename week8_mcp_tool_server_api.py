✅ MCP tool server 做成 FastAPI 服务（工具可发现 + schema 校验 + call）
✅ 接入 Week4 LLM 服务（OpenAI-compatible /v1/chat/completions）
✅ 鉴权 / 限流 / 审计日志（audit）/ trace_id 全链路
✅ Swarm 多 Agent 协作（Retriever→Analyst→Writer）
✅ 离线可跑：即使你没启动 Week4/Week5-6 服务，也能 fallback（dummy 证据 + 模板报告）

运行方式（最快闭环）
1）启动 MCP 工具服务（8002）
pip install fastapi uvicorn pydantic requests
export MCP_API_KEYS=devkey1,devkey2
export MCP_RATE_LIMIT_RPM=60
export AUDIT_LOG_PATH=./audit.log
uvicorn week8_mcp_tool_server_api:app --host 0.0.0.0 --port 8002

2）（可选）启动 Week4 LLM 服务（8000）

你之前 Week4 的 week4_fastapi_llm_server.py 如果已经有，就：

export API_KEYS=llmkey1
uvicorn week4_fastapi_llm_server:app --host 0.0.0.0 --port 8000

3）（可选）启动 Week5-6 RAG 服务（8001）
python week5_6_rag_finance_demo.py
# 默认端口 8001

4）运行 Swarm 客户端（会自动调用 MCP + LLM）
export MCP_BASE_URL=http://localhost:8002
export MCP_API_KEY=devkey1

export LLM_BASE_URL=http://localhost:8000
export LLM_API_KEY=llmkey1   # 如果没启动 Week4，可以不设，会 fallback

python week8_swarm_client_orchestrator.py

"""
week8_mcp_tool_server_api.py

MCP Tool Server (FastAPI)
- /mcp/tools  : list tools + schema (discoverability)
- /mcp/call   : call tool with args (schema validation + allowlist)
- Auth        : API Key (Authorization: Bearer <key>)
- Rate limit  : per key per minute (RPM)
- Audit log   : JSONL (trace_id, tool, args_redacted, latency, ok/error)

可选外部依赖：
- 若你启动了 Week5-6 RAG 服务：http://localhost:8001/query
  rag_query 工具会优先调用它，否则 fallback dummy。
"""

import os
import json
import time
import uuid
import re
from typing import Any, Dict, Callable, List, Optional

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
MCP_API_KEYS = [k.strip() for k in os.getenv("MCP_API_KEYS", "").split(",") if k.strip()]
MCP_RATE_LIMIT_RPM = int(os.getenv("MCP_RATE_LIMIT_RPM", "60"))
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "./audit.log")

MAX_INPUT_CHARS = int(os.getenv("MCP_MAX_INPUT_CHARS", "20000"))
TOOL_TIMEOUT_SEC = float(os.getenv("MCP_TOOL_TIMEOUT_SEC", "4"))

# 简易限流：每 key 记录 60s 内请求时间戳
_REQ_TS: Dict[str, List[float]] = {}

app = FastAPI(title="Week8 MCP Tool Server", version="0.2.0")


# =========================
# Audit logging (JSONL)
# WHY：企业可观测性与审计必备
# =========================
SENSITIVE_KEYS = {"authorization", "api_key", "password", "token", "secret"}

def redact(obj: Any) -> Any:
    """递归脱敏，避免把敏感信息写入审计日志。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in SENSITIVE_KEYS:
                out[k] = "***REDACTED***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    if isinstance(obj, str) and len(obj) > 4000:
        return obj[:4000] + "...(truncated)"
    return obj

def write_audit(record: Dict[str, Any]) -> None:
    record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Auth + Rate limit
# =========================
def require_api_key(authorization: Optional[str]) -> str:
    if not MCP_API_KEYS:
        return "anonymous"  # 开发模式：未配置则不强制
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing API key (Authorization: Bearer ...)")
    key = authorization.split(" ", 1)[1].strip()
    if key not in MCP_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key

def check_rate_limit(key: str) -> None:
    now = time.time()
    window_start = now - 60.0
    arr = _REQ_TS.get(key, [])
    arr = [t for t in arr if t >= window_start]
    if len(arr) >= MCP_RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (RPM)")
    arr.append(now)
    _REQ_TS[key] = arr


# =========================
# MCP Tool Registry
# =========================
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

class MCPCallRequest(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

class MCPCallResponse(BaseModel):
    ok: bool
    trace_id: str
    tool: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    latency_ms: float

# 内置 dummy 证据（无 RAG 服务也可跑）
DUMMY_EVIDENCE = [
    {
        "doc_id": "AR_2024_ACME",
        "date": "2025-03-01",
        "chunk_id": "AR_2024_ACME::w0-80",
        "snippet": "The company reported EBITDA margin of 22% in FY2024. Revenue grew by 18% year-over-year..."
    },
    {
        "doc_id": "RPT_Q4_2024_ACME",
        "date": "2025-01-15",
        "chunk_id": "RPT_Q4_2024_ACME::w0-80",
        "snippet": "We forecast FY2025 EBITDA margin at 23% assuming opex discipline."
    },
]

def validate_schema(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    required = schema.get("required", [])
    props = schema.get("properties", {})
    type_map = {
        "string": str, "integer": int, "number": (int, float),
        "boolean": bool, "object": dict, "array": list
    }
    for r in required:
        if r not in args:
            raise ValueError(f"Missing required field: {r}")
    for k, v in args.items():
        if k not in props:
            continue
        t = props[k].get("type")
        if t and t in type_map and not isinstance(v, type_map[t]):
            raise ValueError(f"Field '{k}' must be type '{t}', got {type(v)}")

def safe_eval(expr: str) -> float:
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        raise ValueError("Unsafe expression")
    return float(eval(expr, {"__builtins__": {}}, {}))

NUM_RE = re.compile(r"(\d+(\.\d+)?)\s*%")

# Tool handlers
def tool_rag_query(args: Dict[str, Any]) -> Dict[str, Any]:
    q = args["query"]
    top_k = args["top_k"]
    if len(q) > MAX_INPUT_CHARS:
        raise ValueError("query too large")

    # 优先调用 Week5-6 RAG 服务
    try:
        r = requests.post(
            "http://localhost:8001/query",
            json={"query": q, "top_k": top_k, "use_mmr": True},
            timeout=TOOL_TIMEOUT_SEC
        )
        r.raise_for_status()
        data = r.json()
        # week5-6 demo 返回 answer + selected
        return {
            "source": "rag_service",
            "selected": data.get("selected", []),
            "answer_preview": data.get("answer", "")[:1200]
        }
    except Exception:
        return {
            "source": "dummy_kb",
            "evidence": DUMMY_EVIDENCE[:top_k]
        }

def tool_extract_numbers(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError("text too large")
    nums = [float(m.group(1)) for m in NUM_RE.finditer(text)]
    return {"numbers_percent": nums, "count": len(nums)}

def tool_calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = args["expression"]
    val = safe_eval(expr)
    return {"expression": expr, "result": val}

def tool_pack_citations(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 evidence 转成统一 citations 列表，供 Writer 使用（grounded 必备）
    """
    evidence = args["evidence"]
    citations = []
    for e in evidence:
        citations.append({"doc_id": e.get("doc_id"), "date": e.get("date"), "chunk_id": e.get("chunk_id")})
    return {"citations": citations}

TOOLS: Dict[str, Dict[str, Any]] = {
    "rag_query": {
        "spec": ToolSpec(
            name="rag_query",
            description="Retrieve evidence from finance documents (annual reports, broker notes).",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                "required": ["query", "top_k"],
            },
            output_schema={"type": "object"},
        ),
        "handler": tool_rag_query,
    },
    "extract_numbers": {
        "spec": ToolSpec(
            name="extract_numbers",
            description="Extract percentage numbers from evidence text.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            output_schema={"type": "object"},
        ),
        "handler": tool_extract_numbers,
    },
    "calculator": {
        "spec": ToolSpec(
            name="calculator",
            description="Compute a safe arithmetic expression.",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            output_schema={"type": "object"},
        ),
        "handler": tool_calculator,
    },
    "pack_citations": {
        "spec": ToolSpec(
            name="pack_citations",
            description="Normalize evidence list into citations list (doc_id/date/chunk_id).",
            input_schema={
                "type": "object",
                "properties": {"evidence": {"type": "array"}},
                "required": ["evidence"],
            },
            output_schema={"type": "object"},
        ),
        "handler": tool_pack_citations,
    },
}


# =========================
# Endpoints
# =========================
@app.middleware("http")
async def attach_trace_and_audit(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or str(uuid.uuid4())
    start = time.time()
    try:
        resp = await call_next(request)
        return resp
    finally:
        latency_ms = (time.time() - start) * 1000.0
        # 只做轻量访问日志（更详细的在 /mcp/call 里写）
        write_audit({
            "kind": "http_access",
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "latency_ms": round(latency_ms, 2),
        })

@app.get("/health")
def health():
    return {"status": "ok", "tools": list(TOOLS.keys())}

@app.get("/mcp/tools")
def list_tools(authorization: Optional[str] = Header(default=None)):
    key = require_api_key(authorization)
    check_rate_limit(key)
    return {"tools": [TOOLS[name]["spec"].model_dump() for name in TOOLS]}

@app.post("/mcp/call", response_model=MCPCallResponse)
def call_tool(req: MCPCallRequest, authorization: Optional[str] = Header(default=None)):
    key = require_api_key(authorization)
    check_rate_limit(key)

    trace_id = req.trace_id or str(uuid.uuid4())
    tool = req.tool
    args = req.args or {}

    # allowlist
    if tool not in TOOLS:
        record = {"kind": "tool_call", "trace_id": trace_id, "api_key": key, "tool": tool, "ok": False, "error": "TOOL_NOT_FOUND"}
        write_audit(redact(record))
        raise HTTPException(status_code=404, detail="Tool not found")

    spec: ToolSpec = TOOLS[tool]["spec"]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]] = TOOLS[tool]["handler"]

    # schema validation
    try:
        validate_schema(args, spec.input_schema)
    except Exception as e:
        write_audit(redact({
            "kind": "tool_call",
            "trace_id": trace_id,
            "api_key": key,
            "tool": tool,
            "args": args,
            "ok": False,
            "error": {"code": "SCHEMA_VALIDATION_ERROR", "message": str(e)},
        }))
        return MCPCallResponse(
            ok=False, trace_id=trace_id, tool=tool, latency_ms=0.0,
            error={"code": "SCHEMA_VALIDATION_ERROR", "message": str(e)}
        )

    # execute
    t0 = time.time()
    try:
        out = handler(args)
        latency_ms = (time.time() - t0) * 1000.0
        write_audit(redact({
            "kind": "tool_call",
            "trace_id": trace_id,
            "api_key": key,
            "tool": tool,
            "args": args,
            "ok": True,
            "latency_ms": round(latency_ms, 2),
            "result_preview": str(out)[:800],
        }))
        return MCPCallResponse(ok=True, trace_id=trace_id, tool=tool, result=out, latency_ms=round(latency_ms, 2))
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        write_audit(redact({
            "kind": "tool_call",
            "trace_id": trace_id,
            "api_key": key,
            "tool": tool,
            "args": args,
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "error": {"code": "TOOL_EXEC_ERROR", "message": str(e)},
        }))
        return MCPCallResponse(
            ok=False, trace_id=trace_id, tool=tool, latency_ms=round(latency_ms, 2),
            error={"code": "TOOL_EXEC_ERROR", "message": str(e)}
        )



