from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


class Settings:
    # 对话 / 视觉模型：DeepSeek（OpenAI 兼容接口）
    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    llm_model = os.environ.get("LLM_MODEL", "deepseek-flash")

    # Embedding：DeepSeek 目前没有 embedding 接口，默认走 OpenAI 兼容的第三方（如硅基流动 bge-m3）。
    # EMBED_API_KEY 为空时自动退化为纯 BM25 检索，服务仍可用。
    embed_api_key = os.environ.get("EMBED_API_KEY", "")
    embed_base_url = os.environ.get("EMBED_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    embed_model = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

    data_dir = ROOT / "data"
    index_dir = ROOT / "index"

    top_k = int(os.environ.get("TOP_K", "5"))
    dense_min = float(os.environ.get("DENSE_MIN", "0.35"))  # 稠密相似度低于此值视为无关
    bm25_min = float(os.environ.get("BM25_MIN", "3.0"))  # BM25 原始分低于此值视为无关
    bm25_strong = float(os.environ.get("BM25_STRONG", "8.0"))  # 有向量检索时 BM25 单独放行的门限
    rate_limit_per_min = int(os.environ.get("RATE_LIMIT_PER_MIN", "15"))


settings = Settings()
