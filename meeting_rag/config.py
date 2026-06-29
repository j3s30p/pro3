from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


ENV_PATH = Path(".env")
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def load_env(path: Path = ENV_PATH) -> Dict[str, str]:
    """프로젝트 루트의 .env 파일을 읽어 key-value dict로 반환한다."""
    values: Dict[str, str] = {}

    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def get_llm_model() -> str:
    """환경 변수 또는 .env에서 사용할 LLM 모델명을 가져온다."""
    return os.environ.get("LLM_MODEL") or load_env().get("LLM_MODEL") or DEFAULT_LLM_MODEL


def get_openai_api_key() -> str:
    """환경 변수 또는 .env에서 OpenAI API 키를 가져온다."""
    api_key = os.environ.get("OPENAI_API_KEY") or load_env().get("OPENAI_API_KEY") or ""
    return api_key.strip()


def get_embedding_model() -> str:
    """환경 변수 또는 .env에서 사용할 임베딩 모델명을 가져온다."""
    return (
        os.environ.get("OPENAI_EMBEDDING_MODEL")
        or load_env().get("OPENAI_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
