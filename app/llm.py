from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import numpy as np

from .config import settings

_client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _chat_body(messages: list[dict], stream: bool, **kw) -> dict:
    # deepseek-flash 带推理过程，推理 token 也算在 max_tokens 里；问答场景用 low 档降低首字延迟
    body = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": 2000,
        "reasoning_effort": "low",
        "stream": stream,
    }
    body.update(kw)
    return body


async def chat(messages: list[dict], **kw) -> str:
    url = f"{settings.llm_base_url}/chat/completions"
    for attempt in range(3):
        r = await _client.post(
            url,
            headers=_headers(settings.llm_api_key),
            json=_chat_body(messages, False, **kw),
        )
        if r.status_code in _RETRY_STATUS and attempt < 2:
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"].get("content") or ""
    return ""


async def chat_stream(messages: list[dict], **kw) -> AsyncIterator[str]:
    """只向外吐 content，reasoning_content（思考过程）丢弃。"""
    url = f"{settings.llm_base_url}/chat/completions"
    async with _client.stream(
        "POST",
        url,
        headers=_headers(settings.llm_api_key),
        json=_chat_body(messages, True, **kw),
    ) as r:
        if r.status_code >= 400:
            await r.aread()
            r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta.get("content"):
                yield delta["content"]


def embed_enabled() -> bool:
    return bool(settings.embed_api_key)


async def embed(texts: list[str], batch: int = 1) -> np.ndarray:
    """返回 L2 归一化后的向量矩阵，点积即余弦相似度。"""
    url = f"{settings.embed_base_url}/embeddings"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        part = texts[i : i + batch]
        for attempt in range(3):
            r = await _client.post(
                url,
                headers=_headers(settings.embed_api_key),
                json={"model": settings.embed_model, "input": part},
            )
            if r.status_code in _RETRY_STATUS and attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(d["embedding"] for d in data)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


def image_message(prompt: str, image_data_url: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
