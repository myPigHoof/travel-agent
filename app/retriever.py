"""混合检索：BM25（jieba 分词 + 领域词典）+ 稠密向量，RRF 融合 + 实体加权 + 相关性门限。"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import jieba
import numpy as np

from .config import settings

STOP = set(
    "的 了 是 在 有 和 与 我 你 他 它 吗 呢 吧 啊 么 什么 怎么 怎么样 如何 请问 一下 一个 这个 那个 哪里 哪些 可以 能否 有没有 想要 帮 写 个 ， 。 ？ ！ 、 ； ： , . ? ! ; :".split()
)
RRF_K = 60
ENTITY_BOOST = 0.02  # 约等于 RRF 中一个名次的差距


def load_aliases() -> Dict[str, List[str]]:
    p = settings.data_dir / "aliases.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


ALIASES = load_aliases()
# 景点名、别名都加入分词词典，避免“卅六鸳鸯馆”“与谁同坐轩”被切碎
for _canon, _alts in ALIASES.items():
    for _w in [_canon] + _alts:
        if len(_w) >= 2:
            jieba.add_word(_w, freq=2000)


def tokenize(text: str) -> List[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip() and t not in STOP]


def expand_query(query: str) -> str:
    """同义词/别名扩展：用户说“廊桥”，补上标准名“小飞虹”再检索。"""
    extra: List[str] = []
    for canon, alts in ALIASES.items():
        if canon in query or any(a in query for a in alts):
            if canon not in query:
                extra.append(canon)
    return query + " " + " ".join(extra) if extra else query


def entities_in(query: str) -> List[str]:
    return [c for c, alts in ALIASES.items() if c in query or any(a in query for a in alts)]


class BM25:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if len(docs) else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: List[str]) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


@dataclass
class Hit:
    chunk: dict
    score: float
    bm25: float
    dense: Optional[float]


class Retriever:
    def __init__(self) -> None:
        d = settings.index_dir
        self.chunks: List[dict] = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])
        vp = d / "vectors.npy"
        self.vectors: Optional[np.ndarray] = np.load(vp) if vp.exists() else None

    @property
    def has_dense(self) -> bool:
        return self.vectors is not None

    def search(self, query: str, qvec: Optional[np.ndarray], k: int, pool: int = 20) -> List[Hit]:
        q_tokens = tokenize(expand_query(query))
        bm = self.bm25.scores(q_tokens)
        dense = (self.vectors @ qvec) if (qvec is not None and self.vectors is not None) else None

        fused = np.zeros(len(self.chunks), dtype=np.float32)
        for arr in [bm] + ([dense] if dense is not None else []):
            for rank, idx in enumerate(np.argsort(-arr)[:pool]):
                if arr[idx] > 0:
                    fused[idx] += 1.0 / (RRF_K + rank + 1)

        ents = entities_in(query)
        if ents:
            for i, c in enumerate(self.chunks):
                if any(e in c["path"] for e in ents):
                    fused[i] += ENTITY_BOOST

        top = np.argsort(-fused)[:k]
        hits = [
            Hit(self.chunks[i], float(fused[i]), float(bm[i]), float(dense[i]) if dense is not None else None)
            for i in top
            if fused[i] > 0
        ]
        # 相关性门限：BM25 与向量都没有像样的匹配时，宁可返回空也不拿无关资料去“编”
        if not hits:
            return []
        best_bm = max(h.bm25 for h in hits)
        best_dense = max((h.dense for h in hits if h.dense is not None), default=-1.0)
        # 有向量时以语义相似度为主，BM25 只在“强字面命中”时放行；无向量时用 BM25 门限
        bm_gate = settings.bm25_min if not self.has_dense or qvec is None else settings.bm25_strong
        if best_bm < bm_gate and best_dense < settings.dense_min:
            return []
        return hits
