"""数据入库：清洗 -> 结构化切分 -> （可选）向量化 -> 落盘。

用法：python -m app.ingest
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import llm
from .config import settings

SENT_SPLIT = re.compile(r"(?<=[。！？；])|\n")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
MAX_LEN = 260  # 单块目标长度（字符），景点介绍一般在 100-250 字，基本一个小节一块
OVERLAP_SENTS = 1


def clean(text: str) -> str:
    """规则清洗：去 HTML、全角空白、多余空行，去掉常见爬取噪声行。"""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("　", " ").replace("\xa0", " ").replace("\r", "")
    noise = re.compile(r"^(责任编辑|来源[:：]|转载|版权所有|点击查看|相关阅读|广告).{0,30}$")
    lines = [ln.rstrip() for ln in text.split("\n") if not noise.match(ln.strip())]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_front_matter(raw: str) -> Tuple[Dict[str, str], str]:
    meta: Dict[str, str] = {}
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            raw = raw[end + 4 :]
    return meta, raw


def split_section(body: str) -> List[str]:
    """按句子累积切块，块间保留 1 句重叠，避免关键信息被切断在边界。"""
    sents = [s.strip() for s in SENT_SPLIT.split(body) if s and s.strip()]
    sents = [re.sub(r"^[-*]\s*", "", s) for s in sents]
    chunks: List[str] = []
    cur: List[str] = []
    for s in sents:
        if cur and sum(len(x) for x in cur) + len(s) > MAX_LEN:
            chunks.append("".join(cur))
            cur = cur[-OVERLAP_SENTS:]
        cur.append(s)
    if cur:
        chunks.append("".join(cur))
    return chunks


def chunk_document(path: Path) -> List[dict]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    body = clean(body)
    title = meta.get("title", path.stem)
    stack: List[str] = []  # 当前标题路径
    sections: List[Tuple[List[str], List[str]]] = []
    buf: List[str] = []
    for line in body.split("\n"):
        m = HEADING.match(line)
        if m:
            if buf:
                sections.append((list(stack), buf))
                buf = []
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2).strip()]
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((list(stack), buf))

    out: List[dict] = []
    for heading_path, lines in sections:
        for piece in split_section("\n".join(lines)):
            crumb = " / ".join(["拙政园"] + [h for h in heading_path if h != "拙政园概况"])
            out.append(
                {
                    "doc": title,
                    "path": crumb,
                    "body": piece,
                    # 入索引的文本带上标题路径：让“远香堂”这类实体名同时参与 BM25 与向量匹配，
                    # 解决“切块后丢失主语”的问题
                    "text": f"【{crumb}】{piece}",
                    "source": meta.get("source", ""),
                    "updated": meta.get("updated", ""),
                    "volatile": meta.get("volatile", "false").lower() == "true",
                }
            )
    return out


def build_chunks() -> List[dict]:
    chunks: List[dict] = []
    for p in sorted(settings.data_dir.rglob("*.md")):
        chunks.extend(chunk_document(p))
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


async def build_index() -> None:
    chunks = build_chunks()
    settings.index_dir.mkdir(exist_ok=True)
    (settings.index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    if llm.embed_enabled():
        vecs = await llm.embed([c["text"] for c in chunks])
        np.save(settings.index_dir / "vectors.npy", vecs)
        (settings.index_dir / "meta.json").write_text(json.dumps({"embed_model": settings.embed_model}), encoding="utf-8")
        print(f"indexed {len(chunks)} chunks, dense dim={vecs.shape[1]}")
    else:
        for f in ("vectors.npy", "meta.json"):
            (settings.index_dir / f).unlink(missing_ok=True)
        print(f"indexed {len(chunks)} chunks (BM25 only: EMBED_API_KEY not set)")


if __name__ == "__main__":
    asyncio.run(build_index())
