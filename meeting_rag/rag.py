from __future__ import annotations

import json
import math
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import chromadb

from .config import get_embedding_model, get_llm_model, get_openai_api_key
from .vectordb import (
    DEFAULT_CHROMA_PATH,
    MINUTES_COLLECTION_NAME,
    STT_COLLECTION_NAME,
    embed_texts,
)


RESPONSES_API_URL = "https://api.openai.com/v1/responses"
SIMILARITY_TOP_K = 6
BM25_TOP_K = 8
RERANK_TOP_K = 5
RERANK_CANDIDATE_LIMIT = 12


def tokenize(text: str) -> List[str]:
    """BM25 계산을 위해 텍스트를 간단한 토큰 목록으로 나눈다."""
    return re.findall(r"[가-힣A-Za-z0-9_]+", text.lower())


def load_documents(client: chromadb.ClientAPI) -> List[Dict[str, Any]]:
    """Chroma의 두 collection에서 RAG 후보 문서를 모두 가져온다."""
    documents: List[Dict[str, Any]] = []

    for collection_name in [STT_COLLECTION_NAME, MINUTES_COLLECTION_NAME]:
        collection = client.get_collection(collection_name)
        result = collection.get(include=["documents", "metadatas"])

        for doc_id, document, metadata in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        ):
            documents.append(
                {
                    "key": f"{collection_name}:{doc_id}",
                    "id": doc_id,
                    "collection": collection_name,
                    "document": document,
                    "metadata": metadata or {},
                    "bm25_score": 0.0,
                    "similarity_score": 0.0,
                    "hybrid_score": 0.0,
                }
            )

    return documents


def bm25_search(query: str, documents: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """문서 전체에 대해 BM25 점수를 계산하고 상위 후보를 반환한다."""
    query_tokens = tokenize(query)
    doc_tokens = [tokenize(str(document["document"])) for document in documents]
    avg_doc_len = sum(len(tokens) for tokens in doc_tokens) / max(len(doc_tokens), 1)
    doc_freq: Dict[str, int] = {}

    for tokens in doc_tokens:
        for token in set(tokens):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    scored: List[Dict[str, Any]] = []
    for document, tokens in zip(documents, doc_tokens):
        token_counts = {token: tokens.count(token) for token in set(tokens)}
        score = 0.0

        for token in query_tokens:
            if token not in token_counts:
                continue

            df = doc_freq.get(token, 0)
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            tf = token_counts[token]
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(avg_doc_len, 1))
            score += idf * (tf * 2.5) / denominator

        candidate = dict(document)
        candidate["bm25_score"] = score
        scored.append(candidate)

    return sorted(scored, key=lambda item: item["bm25_score"], reverse=True)[:top_k]


def similarity_search(client: chromadb.ClientAPI, query: str, api_key: str) -> List[Dict[str, Any]]:
    """질문 임베딩으로 두 Chroma collection에서 유사도 검색을 수행한다."""
    query_embedding = embed_texts([query], model=get_embedding_model(), api_key=api_key)[0]
    candidates: List[Dict[str, Any]] = []

    for collection_name in [STT_COLLECTION_NAME, MINUTES_COLLECTION_NAME]:
        collection = client.get_collection(collection_name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=SIMILARITY_TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            candidates.append(
                {
                    "key": f"{collection_name}:{doc_id}",
                    "id": doc_id,
                    "collection": collection_name,
                    "document": document,
                    "metadata": metadata or {},
                    "bm25_score": 0.0,
                    "similarity_score": 1 / (1 + float(distance)),
                    "hybrid_score": 0.0,
                }
            )

    return candidates


def normalize(values: List[float]) -> List[float]:
    """점수 목록을 0~1 범위로 정규화한다."""
    if not values:
        return []

    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]

    return [(value - low) / (high - low) for value in values]


def merge_candidates(bm25: List[Dict[str, Any]], similarity: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """BM25 후보와 similarity 후보를 합치고 hybrid 점수를 계산한다."""
    by_key: Dict[str, Dict[str, Any]] = {}

    for candidate in bm25 + similarity:
        stored = by_key.setdefault(candidate["key"], dict(candidate))
        stored["bm25_score"] = max(stored["bm25_score"], candidate["bm25_score"])
        stored["similarity_score"] = max(stored["similarity_score"], candidate["similarity_score"])

    candidates = list(by_key.values())
    bm25_scores = normalize([candidate["bm25_score"] for candidate in candidates])
    similarity_scores = normalize([candidate["similarity_score"] for candidate in candidates])

    for candidate, bm25_score, similarity_score in zip(candidates, bm25_scores, similarity_scores):
        candidate["hybrid_score"] = 0.45 * bm25_score + 0.55 * similarity_score

    return sorted(candidates, key=lambda item: item["hybrid_score"], reverse=True)


def call_openai_text(prompt: str, model: str, api_key: str) -> str:
    """OpenAI Responses API를 호출하고 텍스트 응답만 반환한다."""
    body = json.dumps({"model": model, "input": prompt}).encode("utf-8")
    request = urllib.request.Request(
        RESPONSES_API_URL,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API 호출 실패: HTTP {error.code}\n{error_body}") from error

    texts: List[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    if not texts:
        raise RuntimeError("OpenAI 응답에서 텍스트를 찾지 못했습니다.")

    return "\n\n".join(texts)


def truncate(text: str, limit: int = 900) -> str:
    """프롬프트 길이를 줄이기 위해 긴 문서를 자른다."""
    return text if len(text) <= limit else text[:limit] + "\n...(생략)"


def rerank_candidates(query: str, candidates: List[Dict[str, Any]], api_key: str) -> List[Dict[str, Any]]:
    """LLM reranker로 후보 문서 순서를 재정렬한다."""
    rerank_pool = candidates[:RERANK_CANDIDATE_LIMIT]
    labels = {f"C{index + 1}": candidate for index, candidate in enumerate(rerank_pool)}
    candidate_text = "\n\n".join(
        f"{label}\n출처: {format_source(candidate)}\n내용:\n{truncate(str(candidate['document']))}"
        for label, candidate in labels.items()
    )

    prompt = f"""질문에 직접 답할 수 있는 근거 후보만 고르세요.
반드시 JSON 배열만 반환하세요. 예: ["C2", "C1", "C5"]
관련 후보가 없으면 []만 반환하세요.
- 회의록 섹션은 LLM이 생성한 요약이므로, 확정/결정/최종/담당 여부 판단에서는 STT 원문보다 낮은 신뢰도로 봅니다.
- 후보/샘플/검토 맥락과 최종 확정 맥락을 구분하세요.
- 질문이 확정/결정/최종/담당 여부를 묻는다면, 가능한 경우 STT 원문 후보를 우선 고르세요.

질문:
{query}

후보:
{candidate_text}
"""

    try:
        selected_labels = json.loads(call_openai_text(prompt, get_llm_model(), api_key))
    except (RuntimeError, json.JSONDecodeError):
        return candidates[:RERANK_TOP_K]

    if not isinstance(selected_labels, list):
        return candidates[:RERANK_TOP_K]

    reranked = [labels[label] for label in selected_labels if label in labels]
    return reranked[:RERANK_TOP_K]


def format_source(candidate: Dict[str, Any]) -> str:
    """후보 문서의 출처 정보를 사람이 읽을 수 있게 만든다."""
    metadata = candidate["metadata"]

    if candidate["collection"] == STT_COLLECTION_NAME:
        return (
            f"STT {metadata.get('start_time')}~{metadata.get('end_time')} "
            f"/ 화자: {metadata.get('speakers')}"
        )

    return f"회의록 섹션: {metadata.get('section_title')}"


def answer_question(query: str, candidates: List[Dict[str, Any]], api_key: str) -> str:
    """선택된 근거 후보만 사용해 최종 답변을 생성한다."""
    if not candidates:
        return "회의록에서 확인되지 않습니다."

    context = "\n\n".join(
        f"[S{index + 1}] {format_source(candidate)}\n{candidate['document']}"
        for index, candidate in enumerate(candidates)
    )

    prompt = f"""너는 회의록 RAG 어시스턴트다.
아래 근거만 사용해 한국어로 답하라.
근거가 부족하면 "회의록에서 확인되지 않습니다"라고 말하라.
답변에는 필요한 경우 [S1] 같은 출처 표시를 붙여라.
회의록 섹션은 LLM이 생성한 요약이고, STT 원문은 1차 근거다.
확정/결정/최종/담당 여부는 STT 원문 근거가 있을 때만 단정하라.
질문에 "최종", "확정", "결정"이 있으면 후보/샘플/검토 발언을 최종 확정으로 해석하지 말라.
서로 다른 근거가 있으면 단정하지 말고 "검토/후보로 언급됨", "확정 여부는 확인 필요"처럼 답하라.

질문:
{query}

근거:
{context}
"""
    return call_openai_text(prompt, get_llm_model(), api_key)


def run_rag(query: str, chroma_path: Path = DEFAULT_CHROMA_PATH) -> Dict[str, Any]:
    """BM25, similarity, LLM reranker를 거쳐 RAG 답변을 만든다."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정해야 RAG를 실행할 수 있습니다.")

    client = chromadb.PersistentClient(path=str(chroma_path))
    documents = load_documents(client)
    bm25_candidates = bm25_search(query, documents, BM25_TOP_K)
    similarity_candidates = similarity_search(client, query, api_key)
    hybrid_candidates = merge_candidates(bm25_candidates, similarity_candidates)
    reranked = rerank_candidates(query, hybrid_candidates, api_key)
    answer = answer_question(query, reranked, api_key)

    return {"answer": answer, "sources": reranked}


def main() -> int:
    """CLI 질문을 받아 RAG 답변과 출처를 출력한다."""
    if len(sys.argv) < 2:
        print('usage: uv run python -m meeting_rag.rag "질문"')
        return 1

    query = " ".join(sys.argv[1:])

    try:
        result = run_rag(query)
    except RuntimeError as error:
        print(f"error: {error}")
        return 1

    print("답변:")
    print(result["answer"])
    print()
    print("출처:")
    if not result["sources"]:
        print("관련 출처 없음")
    else:
        for index, candidate in enumerate(result["sources"], start=1):
            print(f"{index}. {format_source(candidate)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
