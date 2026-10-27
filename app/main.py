from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, rag
from .config import settings
from .ingest import build_index

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"
MAX_IMAGE_B64 = 3_000_000

app = FastAPI(title="拙政园智能导览 RAG")


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatReq(BaseModel):
    question: str = Field(default="", max_length=300)
    history: List[Msg] = []
    image: Optional[str] = None  # data URL


_hits: Dict[str, Deque[float]] = defaultdict(deque)


def check_rate(request: Request) -> None:
    """公开演示站点会消耗 API 额度，做一个简单的按 IP 滑动窗口限流。"""
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= settings.rate_limit_per_min:
        raise HTTPException(429, "请求太频繁，请稍后再试")
    q.append(now)


@app.on_event("startup")
async def startup() -> None:
    if not (settings.index_dir / "chunks.json").exists():
        await build_index()
    rag.reload_retriever()


@app.get("/api/health")
def health() -> dict:
    r = rag.get_retriever()
    return {"ok": True, "chunks": len(r.chunks), "dense": r.has_dense, "llm": settings.llm_model}


@app.post("/api/chat")
async def chat(req: ChatReq, request: Request) -> StreamingResponse:
    check_rate(request)
    if not req.question.strip() and not req.image:
        raise HTTPException(400, "问题不能为空")
    if req.image:
        if not req.image.startswith("data:image/") or len(req.image) > MAX_IMAGE_B64:
            raise HTTPException(400, "图片格式不支持或过大")
    history = [m.model_dump() for m in req.history]

    async def gen():
        async for ev in rag.answer_stream(req.question, history, req.image):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
