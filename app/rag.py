"""RAG 主流程：[识图] -> 问题改写 -> 混合检索 -> 门限拒答 -> 带引用的流式生成。"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, List, Optional

import numpy as np

from . import llm
from .config import settings
from .retriever import ALIASES, Retriever

log = logging.getLogger("rag")

SPOTS = [k for k in ALIASES if len(k) <= 7 and k not in ("门票价格", "开放时间", "预约入园", "交通", "游览路线", "最佳季节", "王献臣", "名称由来", "文徵明")]

SYSTEM = """你是“拙政园智能导览”，面向游客回答关于苏州拙政园的问题。
规则：
1. 只能依据【资料】回答，不得使用资料之外的知识补充事实；资料没有提到的内容，直接说“资料中暂未涉及这一点”，并建议游客查看官方渠道或现场指示牌。
2. 每个关键事实后用 [1][2] 标注来源编号，编号对应【资料】中的序号。
3. 涉及门票、开放时间、预约等可能变动的信息，资料条目会标有“时效信息”，回答末尾必须提醒“以官方公众号‘苏州园林’公布为准”。
4. 回答口语化、简洁，适合游客在手机上阅读；问路线时给出清晰的先后顺序。
5. 如果不同资料互相矛盾，如实说明，不要自行挑选。"""

REWRITE_PROMPT = """根据对话历史，把用户最新的问题改写成一个不依赖上下文、可以独立检索的完整问句（补全指代，如“它”“那里”“这个亭子”）。
只输出改写后的问句，不要解释。

对话历史：
{history}

最新问题：{question}"""

VISION_PROMPT = """这是游客在苏州拙政园拍的照片。请判断照片拍的是下面哪个景点或景物，并简要描述画面。
候选景点：{spots}
只输出 JSON：{{"spot": "候选景点名，无法判断或不是拙政园景物则为 null", "confidence": "high|medium|low", "description": "20字以内的画面描述"}}"""

NO_ANSWER = "这个问题我在拙政园的资料库里没有找到相关内容。你可以换个问法，比如问景点介绍、门票开放时间、游览路线、最佳赏花季节；或者查看官方公众号“苏州园林”。"

_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reload_retriever() -> None:
    global _retriever
    _retriever = Retriever()


def _fmt_history(history: List[dict]) -> str:
    return "\n".join(f"{'游客' if h['role'] == 'user' else '助手'}：{h['content'][:200]}" for h in history[-6:])


async def rewrite_query(question: str, history: List[dict]) -> str:
    if not history:
        return question
    try:
        out = await llm.chat(
            [{"role": "user", "content": REWRITE_PROMPT.format(history=_fmt_history(history), question=question)}],
            max_tokens=400,
        )
        out = out.strip().splitlines()[0] if out.strip() else ""
        return out or question
    except Exception as e:  # 改写失败不应拖垮主链路
        log.warning("rewrite failed: %s", e)
        return question


async def identify_image(data_url: str) -> dict:
    prompt = VISION_PROMPT.format(spots="、".join(SPOTS))
    raw = await llm.chat(llm.image_message(prompt, data_url), max_tokens=600)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        info = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        info = {}
    spot = info.get("spot")
    if spot not in SPOTS:
        spot = None
    return {"spot": spot, "confidence": info.get("confidence", "low"), "description": info.get("description", "")}


def _context_block(hits) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        tag = f"（时效信息，资料更新于{c['updated']}）" if c["volatile"] else ""
        parts.append(f"[{i}]{tag}{c['path']}：{c['body']}")
    return "\n".join(parts)


def _public_source(i: int, h) -> dict:
    c = h.chunk
    return {
        "n": i,
        "path": c["path"],
        "text": c["body"],
        "source": c["source"],
        "updated": c["updated"],
        "volatile": c["volatile"],
        "score": round(h.score, 4),
        "bm25": round(h.bm25, 2),
        "dense": None if h.dense is None else round(h.dense, 3),
    }


async def answer_stream(question: str, history: List[dict], image: Optional[str] = None) -> AsyncIterator[dict]:
    query = question.strip()

    if image:
        vision = await identify_image(image)
        yield {"type": "vision", **vision}
        if vision["spot"]:
            query = f"{vision['spot']} {query}" if query else f"介绍一下{vision['spot']}"
        elif not query:
            yield {"type": "token", "text": "这张图片我没能识别出是拙政园的哪个景点，可以告诉我景点名称，或者换一张更清晰的照片试试。"}
            yield {"type": "done"}
            return

    standalone = await rewrite_query(query, history)
    if standalone != query:
        yield {"type": "rewrite", "text": standalone}

    r = get_retriever()
    qvec = None
    if r.has_dense:
        try:
            qvec = (await llm.embed([standalone]))[0]
        except Exception as e:  # embedding 服务不可用时退化为 BM25，保证可用性
            log.warning("embed failed, fallback to BM25: %s", e)

    hits = r.search(standalone, qvec, settings.top_k)
    yield {"type": "sources", "items": [_public_source(i, h) for i, h in enumerate(hits, 1)], "mode": "hybrid" if qvec is not None else "bm25"}

    if not hits:
        yield {"type": "token", "text": NO_ANSWER}
        yield {"type": "done"}
        return

    messages = [{"role": "system", "content": SYSTEM}]
    messages += [{"role": h["role"], "content": h["content"][:500]} for h in history[-6:]]
    messages.append({"role": "user", "content": f"【资料】\n{_context_block(hits)}\n\n【问题】{standalone}"})
    try:
        async for tok in llm.chat_stream(messages):
            yield {"type": "token", "text": tok}
    except Exception as e:
        log.exception("generation failed")
        yield {"type": "error", "text": f"生成回答时出错：{type(e).__name__}"}
    yield {"type": "done"}
