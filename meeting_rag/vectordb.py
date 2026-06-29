from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from .chunking import chunk_utterances
from .config import get_embedding_model, get_openai_api_key
from .parsing import DEFAULT_INPUT_PATH, parse_clova_response


DEFAULT_MINUTES_PATH = Path("outputs/minutes.md")
DEFAULT_CHROMA_PATH = Path("data/chroma")
STT_COLLECTION_NAME = "meeting_stt_chunks"
MINUTES_COLLECTION_NAME = "meeting_minutes_chunks"
EMBEDDINGS_API_URL = "https://api.openai.com/v1/embeddings"


def embed_texts(texts: List[str], model: str, api_key: str) -> List[List[float]]:
    """OpenAI Embeddings API를 호출해 텍스트 목록을 벡터로 바꾼다."""
    request_body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        EMBEDDINGS_API_URL,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI 임베딩 API 호출 실패: HTTP {error.code}\n{body}") from error

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("OpenAI 임베딩 응답에 data 배열이 없습니다.")

    ordered = sorted(data, key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in ordered]


def split_minutes_sections(minutes_path: Path) -> List[Dict[str, Any]]:
    """회의록 Markdown을 H2 섹션 기준으로 나눈다."""
    text = minutes_path.read_text(encoding="utf-8")
    sections: List[Dict[str, Any]] = []
    current_title = "회의록"
    current_lines: List[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append(
                    {
                        "section_index": len(sections),
                        "section_title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
            current_title = line.removeprefix("## ").strip()
            current_lines = [line]
            continue

        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section_index": len(sections),
                "section_title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def index_stt_chunks(client: chromadb.ClientAPI, input_path: Path, model: str, api_key: str) -> int:
    """CLOVA STT 발화 청크를 Chroma collection에 저장한다."""
    rows = parse_clova_response(input_path)
    chunks = chunk_utterances(rows)
    documents = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(documents, model=model, api_key=api_key)
    collection = client.get_or_create_collection(STT_COLLECTION_NAME)

    collection.upsert(
        ids=[f"stt-{chunk['chunk_index']:04d}" for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "type": "stt_chunk",
                "source_file": str(input_path),
                "chunk_index": chunk["chunk_index"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "utterance_start_index": chunk["utterance_start_index"],
                "utterance_end_index": chunk["utterance_end_index"],
                "speakers": ", ".join(chunk["speakers"]),
            }
            for chunk in chunks
        ],
    )

    return len(chunks)


def index_minutes_sections(client: chromadb.ClientAPI, minutes_path: Path, model: str, api_key: str) -> int:
    """LLM 생성 회의록 섹션을 Chroma collection에 저장한다."""
    sections = split_minutes_sections(minutes_path)
    documents = [section["text"] for section in sections]
    embeddings = embed_texts(documents, model=model, api_key=api_key)
    collection = client.get_or_create_collection(MINUTES_COLLECTION_NAME)

    collection.upsert(
        ids=[f"minutes-{section['section_index']:04d}" for section in sections],
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "type": "minutes_section",
                "source_file": str(minutes_path),
                "section_index": section["section_index"],
                "section_title": section["section_title"],
            }
            for section in sections
        ],
    )

    return len(sections)


def index_vectordb(input_path: Path, minutes_path: Path, chroma_path: Path) -> Dict[str, Any]:
    """STT 청크와 회의록 섹션을 Chroma Vector DB에 저장한다."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정해야 Vector DB를 만들 수 있습니다.")

    model = get_embedding_model()
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    stt_count = index_stt_chunks(client, input_path=input_path, model=model, api_key=api_key)
    minutes_count = index_minutes_sections(client, minutes_path=minutes_path, model=model, api_key=api_key)

    return {
        "embedding_model": model,
        "chroma_path": str(chroma_path),
        "stt_chunks": stt_count,
        "minutes_sections": minutes_count,
    }


def main() -> int:
    """기본 입력 파일들을 Chroma Vector DB에 인덱싱한다."""
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    minutes_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MINUTES_PATH
    chroma_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CHROMA_PATH

    try:
        result = index_vectordb(input_path, minutes_path, chroma_path)
    except RuntimeError as error:
        print(f"error: {error}")
        return 1

    print(f"embedding_model: {result['embedding_model']}")
    print(f"chroma_path: {result['chroma_path']}")
    print(f"stt_chunks: {result['stt_chunks']}")
    print(f"minutes_sections: {result['minutes_sections']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
