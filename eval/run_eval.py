"""检索评估：hit@k（期望条目是否出现在前 k 条）、MRR；null 表示应当被拒答（空召回）。

用法：python -m eval.run_eval            # 用当前索引（有向量则为混合检索）
      python -m eval.run_eval --bm25     # 强制只用 BM25，用于对比消融
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app import llm
from app.config import settings
from app.retriever import Retriever


async def main() -> None:
    bm25_only = "--bm25" in sys.argv
    r = Retriever()
    rows = [json.loads(l) for l in (Path(__file__).parent / "qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    use_dense = r.has_dense and not bm25_only
    qvecs = (await llm.embed([x["q"] for x in rows])) if use_dense else [None] * len(rows)
    k = settings.top_k
    hit1 = hitk = rr = n_pos = refuse_ok = n_neg = 0
    for row, qv in zip(rows, qvecs):
        hits = r.search(row["q"], qv, k)
        if row["path"] is None:
            n_neg += 1
            refuse_ok += not hits
            if hits:
                print(f"[应拒答却命中] {row['q']} -> {hits[0].chunk['path']} (bm25={hits[0].bm25:.1f})")
            continue
        n_pos += 1
        rank = next((i for i, h in enumerate(hits, 1) if row["path"] in h.chunk["path"] or row["path"] in h.chunk["body"][:30]), None)
        if rank:
            hitk += 1
            hit1 += rank == 1
            rr += 1 / rank
        else:
            print(f"[未命中] {row['q']} -> {[h.chunk['path'] for h in hits[:3]]}")
    mode = "hybrid" if use_dense else "bm25"
    print(f"\nmode={mode}  hit@1={hit1/n_pos:.2f}  hit@{k}={hitk/n_pos:.2f}  MRR={rr/n_pos:.2f}  拒答正确率={refuse_ok}/{n_neg}")


if __name__ == "__main__":
    asyncio.run(main())
