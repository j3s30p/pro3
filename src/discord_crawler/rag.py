from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypedDict

import chromadb
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_CHROMA_DIR = ROOT_DIR / "data" / "chroma"
MESSAGE_COLLECTION_NAME = "discord_message_windows"
ATTACHMENT_COLLECTION_NAME = "discord_attachment_chunks"
DEFAULT_ATTACHMENT_CHUNK_TOKENS = 600
DEFAULT_ATTACHMENT_CHUNK_OVERLAP_TOKENS = 80
KST = timezone(timedelta(hours=9), "KST")
AGENT_MAX_STEPS = 3
AGENT_MESSAGE_CANDIDATE_LIMIT = 80
AGENT_ATTACHMENT_CANDIDATE_LIMIT = 30
GENERIC_SEARCH_TERMS = {
    "뭐야",
    "뭐지",
    "뭔가",
    "뭔지",
    "어떤",
    "알려줘",
    "알려",
    "정리해줘",
    "있어",
    "있나",
    "했지",
    "했어",
}
TEMPORAL_QUERY_KEYWORDS = {
    "오늘",
    "내일",
    "이번",
    "이번주",
    "이번 주",
    "다음",
    "앞으로",
    "최근",
    "마지막",
    "마감",
    "일정",
    "언제",
    "주제",
}
KOREAN_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "부터",
    "까지",
    "에게",
    "하고",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "도",
    "만",
    "와",
    "과",
    "랑",
    "로",
)
DATE_MENTION_PATTERN = re.compile(r"(?:(20\d{2})[./-])?(\d{1,2})[./-](\d{1,2})")

ATTACHMENT_QUERY_KEYWORDS = {
    "첨부",
    "첨부파일",
    "파일",
    "자료",
    "문서",
    "pdf",
    "html",
    "이미지",
    "사진",
    "스크린샷",
    "캡처",
    "캡쳐",
    "코드",
    "노트북",
    "csv",
    "json",
    "txt",
    "md",
    "png",
    "jpg",
    "jpeg",
    "다운로드",
    "업로드",
    "올린 파일",
    "올린 자료",
}

TEXT_ATTACHMENT_EXTENSIONS = {
    ".csv",
    ".css",
    ".htm",
    ".html",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

SYSTEM_PROMPT = """당신은 Discord 대화 기록을 근거로 답하는 RAG Q&A 도우미입니다.
규칙:
- 반드시 제공된 검색 결과 안의 내용만 근거로 답합니다.
- 검색 결과에 근거가 부족하면 "대화 기록에서 확인되지 않습니다"라고 말합니다.
- 답변은 Markdown으로 간결하고 읽기 좋게 작성합니다.
- 필요한 경우 핵심 내용은 짧은 bullet list로 정리합니다.
- 누가, 언제, 어느 채널에서 말했는지 본문에 자연스럽게 요약합니다.
- 첨부파일 내용은 검색 결과에 추출된 텍스트나 파일명으로 확인되는 범위에서만 언급합니다.
- 본문에 raw source, local_path, URL을 길게 반복하지 않습니다.
- "참고 메시지", "참고 첨부파일", "출처" 섹션은 작성하지 않습니다. 출처 목록은 시스템이 자동으로 붙입니다.
"""

QUERY_PLANNER_PROMPT = """당신은 Discord 대화 RAG 시스템의 retrieval planner입니다.
역할:
- 사용자 질문을 보고 검색 전략만 결정합니다.
- 답변을 작성하지 않습니다.
- 검색 결과에 없는 사실을 만들지 않습니다.

반드시 JSON object 하나만 반환하세요.
스키마:
{
  "search_query": "vector search에 사용할 한국어 검색 질의",
  "include_attachments": true,
  "prefer_recent": false,
  "recency_weight": 0.0,
  "message_candidate_top_k": 20,
  "attachment_candidate_top_k": 6,
  "reason": "짧은 한국어 설명"
}

필드 규칙:
- search_query는 원 질문의 핵심 명사, 사람, 채널, 주제, 시간 표현을 보존하면서 검색에 잘 걸리게 확장합니다.
- include_attachments는 파일, 첨부, 자료, PDF, HTML, 이미지, 코드 등 첨부파일 확인이 필요한 질문일 때 true입니다.
- prefer_recent는 사용자가 오늘, 다음, 이번 주, 앞으로, 최근, 마지막, 마감, 일정처럼 시간 흐름이 중요한 질문을 할 때 true입니다.
- recency_weight는 최신성이 중요하지 않으면 0.0, 중요하면 0.3~1.0 사이로 둡니다.
- message_candidate_top_k는 일반 질문이면 10~20, 시간성/애매한 질문이면 30~60 사이로 넓힙니다.
- attachment_candidate_top_k는 include_attachments가 false면 0, true면 6~20 사이로 둡니다.
"""

AGENT_JUDGE_PROMPT = """당신은 Discord RAG agent의 retrieval judge입니다.
역할:
- 현재까지 검색된 후보가 사용자 질문에 답하기에 충분한지 판단합니다.
- 충분하면 답변 근거로 쓸 후보 key만 고릅니다.
- 부족하면 다음에 실행할 검색 도구를 하나 고릅니다.
- 답변을 작성하지 않습니다.
- 후보에 없는 사실을 만들지 않습니다.

반드시 JSON object 하나만 반환하세요.
스키마:
{
  "is_sufficient": true,
  "message_keys": ["message-key-1", "message-key-2"],
  "attachment_keys": ["attachment-key-1"],
  "next_action": {"tool": "none", "query": "", "k": 0},
  "reason": "짧은 한국어 설명"
}

도구:
- vector_search_messages: 의미적으로 비슷한 Discord 메시지 window를 검색합니다.
- recent_messages: 질문 키워드가 포함된 최근 메시지를 시간 역순으로 가져옵니다.
- vector_search_attachments: 첨부파일 chunk를 검색합니다.
- none: 추가 검색이 필요 없을 때만 사용합니다.

판단 규칙:
- 후보가 질문에 직접 답하면 is_sufficient=true로 두고, message_keys/attachment_keys를 최대 6개씩 선택합니다.
- 후보가 부족하면 is_sufficient=false로 두고, 다음 검색 도구 하나를 next_action에 넣습니다.
- "다음", "앞으로", "오늘", "이번 주" 같은 표현은 현재 시각 기준으로 해석합니다.
- 현재 시각 기준 이미 지난 일정/주제만 있으면 충분하지 않습니다. recent_messages로 더 찾아야 합니다.
- 파일, 자료, 첨부, HTML, PDF, 이미지, 코드 질문에서 첨부파일 후보가 부족하면 vector_search_attachments를 사용합니다.
- 이미 실행한 도구를 반복하기보다 query를 바꾸거나 다른 도구를 고릅니다.
- next_action.k는 메시지 검색 5~80, 첨부파일 검색 1~30 범위로 둡니다.
"""


class RagError(RuntimeError):
    pass


_KIWI: Kiwi | None = None


@dataclass(frozen=True)
class MessageRecord:
    guild_id: str | None
    guild_name: str | None
    category_id: str | None
    category_name: str | None
    channel_id: str
    channel_name: str
    message_id: str
    author_id: str
    author_name: str
    author_display_name: str
    created_at: str
    edited_at: str | None
    content: str
    clean_content: str
    jump_url: str
    attachments: list[dict[str, Any]]
    embeds: list[dict[str, Any]]
    source_path: str


@dataclass(frozen=True)
class ChunkDocument:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    document: str
    metadata: dict[str, Any]
    distance: float | None


@dataclass(frozen=True)
class RetrievalPlan:
    search_query: str
    include_attachments: bool
    prefer_recent: bool
    recency_weight: float
    message_candidate_top_k: int
    attachment_candidate_top_k: int
    final_top_k: int
    reason: str
    source: str


class AgenticRagState(TypedDict, total=False):
    question: str
    current_datetime: str
    plan: RetrievalPlan
    step: int
    max_steps: int
    actions: list[dict[str, Any]]
    next_action: dict[str, Any]
    is_sufficient: bool
    judge_reason: str
    message_candidates: list[SearchResult]
    attachment_candidates: list[SearchResult]
    message_results: list[SearchResult]
    attachment_results: list[SearchResult]


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def get_kiwi() -> Kiwi:
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...(truncated)"


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RagError("OPENAI_API_KEY is required. Put it in .env before building or asking.")
    return api_key


def is_attachment_query(question: str) -> bool:
    normalized = question.casefold()
    return any(keyword.casefold() in normalized for keyword in ATTACHMENT_QUERY_KEYWORDS)


def clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def fallback_retrieval_plan(question: str, final_top_k: int) -> RetrievalPlan:
    include_attachments = is_attachment_query(question)
    return RetrievalPlan(
        search_query=question,
        include_attachments=include_attachments,
        prefer_recent=False,
        recency_weight=0.0,
        message_candidate_top_k=final_top_k,
        attachment_candidate_top_k=final_top_k if include_attachments else 0,
        final_top_k=final_top_k,
        reason="planner fallback",
        source="fallback",
    )


def normalize_retrieval_plan(payload: dict[str, Any], question: str, final_top_k: int) -> RetrievalPlan:
    search_query = str(payload.get("search_query") or question).strip() or question
    include_attachments = parse_bool(payload.get("include_attachments"), default=is_attachment_query(question))
    prefer_recent = parse_bool(payload.get("prefer_recent"), default=False)
    recency_weight = clamp_float(payload.get("recency_weight"), 0.0, 1.0, 0.0)
    if prefer_recent and recency_weight < 0.3:
        recency_weight = 0.5

    message_candidate_top_k = clamp_int(
        payload.get("message_candidate_top_k"),
        minimum=final_top_k,
        maximum=80,
        default=max(final_top_k, 20),
    )
    attachment_candidate_top_k = (
        clamp_int(
            payload.get("attachment_candidate_top_k"),
            minimum=final_top_k,
            maximum=30,
            default=final_top_k,
        )
        if include_attachments
        else 0
    )
    return RetrievalPlan(
        search_query=search_query,
        include_attachments=include_attachments,
        prefer_recent=prefer_recent,
        recency_weight=recency_weight,
        message_candidate_top_k=message_candidate_top_k,
        attachment_candidate_top_k=attachment_candidate_top_k,
        final_top_k=final_top_k,
        reason=str(payload.get("reason") or "").strip(),
        source="llm",
    )


def build_retrieval_plan(
    question: str,
    current_datetime: str,
    openai_client: OpenAI,
    planner_model: str,
    final_top_k: int,
    use_planner: bool,
) -> RetrievalPlan:
    if not use_planner:
        return fallback_retrieval_plan(question, final_top_k)

    try:
        response = openai_client.chat.completions.create(
            model=planner_model,
            messages=[
                {"role": "system", "content": QUERY_PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"[현재 시각]\n{current_datetime}\n\n"
                        f"[사용자 질문]\n{question}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
    except Exception:
        return fallback_retrieval_plan(question, final_top_k)

    if not isinstance(payload, dict):
        return fallback_retrieval_plan(question, final_top_k)
    return normalize_retrieval_plan(payload, question, final_top_k)


def load_message_records(raw_dir: Path) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    for path in sorted(raw_dir.rglob("*.jsonl")):
        if path.name == "manifest.json":
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                records.append(
                    MessageRecord(
                        guild_id=payload.get("guild_id"),
                        guild_name=payload.get("guild_name"),
                        category_id=payload.get("category_id"),
                        category_name=payload.get("category_name"),
                        channel_id=str(payload["channel_id"]),
                        channel_name=str(payload.get("channel_name", "")),
                        message_id=str(payload["message_id"]),
                        author_id=str(payload.get("author_id", "")),
                        author_name=str(payload.get("author_name", "")),
                        author_display_name=str(
                            payload.get("author_display_name") or payload.get("author_name") or ""
                        ),
                        created_at=str(payload.get("created_at", "")),
                        edited_at=payload.get("edited_at"),
                        content=str(payload.get("content") or ""),
                        clean_content=str(payload.get("clean_content") or ""),
                        jump_url=str(payload.get("jump_url") or ""),
                        attachments=list(payload.get("attachments") or []),
                        embeds=list(payload.get("embeds") or []),
                        source_path=str(path),
                    )
                )
    return sorted(records, key=lambda record: (record.channel_id, record.created_at, record.message_id))


def group_by_channel(records: Iterable[MessageRecord]) -> dict[str, list[MessageRecord]]:
    grouped: dict[str, list[MessageRecord]] = {}
    for record in records:
        grouped.setdefault(record.channel_id, []).append(record)
    return grouped


def read_attachment_text(attachment: dict[str, Any], max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    local_path = attachment.get("local_path")
    if not local_path:
        return ""

    path = Path(str(local_path))
    suffix = path.suffix.casefold()
    content_type = str(attachment.get("content_type") or "").casefold()
    looks_textual = (
        suffix in TEXT_ATTACHMENT_EXTENSIONS
        or content_type.startswith("text/")
        or "json" in content_type
        or "csv" in content_type
        or "html" in content_type
        or "xml" in content_type
    )
    if not looks_textual or not path.exists() or path.stat().st_size > 1_000_000:
        return ""

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:max_chars].strip()


def split_text(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if max_tokens <= 0:
        return [stripped]

    tokens = [token for token in get_kiwi().tokenize(stripped) if token.form.strip()]
    if not tokens or len(tokens) <= max_tokens:
        return [stripped]

    chunks: list[str] = []
    start_index = 0
    while start_index < len(tokens):
        end_index = min(start_index + max_tokens, len(tokens))
        window = tokens[start_index:end_index]
        start_char = window[0].start
        end_char = max(token.end for token in window)
        chunk = stripped[start_char:end_char].strip()
        if chunk:
            chunks.append(chunk)

        if end_index >= len(tokens):
            break
        start_index = max(end_index - overlap_tokens, start_index + 1)

    return chunks


def render_message(record: MessageRecord, max_attachment_chars: int) -> str:
    author = record.author_display_name or record.author_name or record.author_id
    content = record.clean_content or record.content
    lines = [
        f"- 시간: {record.created_at}",
        f"  작성자: {author} ({record.author_name}, id={record.author_id})",
        f"  메시지: {content if content else '(본문 없음)'}",
    ]

    if record.embeds:
        lines.append("  임베드:")
        for index, embed in enumerate(record.embeds, start=1):
            embed_text = str(embed.get("text") or "").strip()
            title = str(embed.get("title") or "").strip()
            description = str(embed.get("description") or "").strip()
            url = str(embed.get("url") or "").strip()
            lines.append(f"    {index}. title={title or '(없음)'}")
            if description:
                lines.append(f"       description={description}")
            if url:
                lines.append(f"       url={url}")
            if embed_text and embed_text not in {title, description, url}:
                lines.append(f"       text={embed_text}")

    if record.attachments:
        lines.append("  첨부파일:")
        for index, attachment in enumerate(record.attachments, start=1):
            filename = attachment.get("filename") or "unknown"
            content_type = attachment.get("content_type") or "unknown"
            local_path = attachment.get("local_path") or ""
            lines.append(f"    {index}. {filename} ({content_type})")
            if local_path:
                lines.append(f"       local_path={local_path}")
            attachment_text = read_attachment_text(attachment, max_attachment_chars)
            if attachment_text:
                lines.append(f"       extracted_text={attachment_text}")

    if record.jump_url:
        lines.append(f"  링크: {record.jump_url}")

    return "\n".join(lines)


def render_chunk(
    center: MessageRecord,
    window_records: list[MessageRecord],
    max_attachment_chars: int,
) -> str:
    category_name = center.category_name or "no_category"
    header = [
        f"서버: {center.guild_name or center.guild_id}",
        f"카테고리: {category_name}",
        f"채널: #{center.channel_name}",
        f"중심 메시지 ID: {center.message_id}",
        "대화:",
    ]
    body = [render_message(record, max_attachment_chars) for record in window_records]
    return "\n".join(header + body)


def build_chunk_documents(
    records: list[MessageRecord],
    before: int,
    after: int,
    max_attachment_chars: int,
    max_chunk_chars: int,
) -> list[ChunkDocument]:
    documents: list[ChunkDocument] = []
    for channel_records in group_by_channel(records).values():
        for index, center in enumerate(channel_records):
            start = max(0, index - before)
            end = min(len(channel_records), index + after + 1)
            window_records = channel_records[start:end]
            text = truncate_text(render_chunk(center, window_records, max_attachment_chars), max_chunk_chars)
            author_names = sorted({record.author_display_name for record in window_records if record.author_display_name})
            message_ids = [record.message_id for record in window_records]
            attachment_count = sum(len(record.attachments) for record in window_records)
            embed_count = sum(len(record.embeds) for record in window_records)

            documents.append(
                ChunkDocument(
                    chunk_id=f"{center.channel_id}:{center.message_id}:w{before}-{after}",
                    text=text,
                    metadata={
                        "guild_id": center.guild_id or "",
                        "guild_name": center.guild_name or "",
                        "category_id": center.category_id or "",
                        "category_name": center.category_name or "no_category",
                        "channel_id": center.channel_id,
                        "channel_name": center.channel_name,
                        "center_message_id": center.message_id,
                        "center_author_id": center.author_id,
                        "center_author_name": center.author_name,
                        "center_author_display_name": center.author_display_name,
                        "center_created_at": center.created_at,
                        "center_jump_url": center.jump_url,
                        "start_message_id": message_ids[0],
                        "end_message_id": message_ids[-1],
                        "start_created_at": window_records[0].created_at,
                        "end_created_at": window_records[-1].created_at,
                        "message_ids": ",".join(message_ids),
                        "author_names": ",".join(author_names),
                        "attachment_count": attachment_count,
                        "embed_count": embed_count,
                        "source_path": center.source_path,
                    },
                )
            )
    return documents


def render_attachment_chunk(
    record: MessageRecord,
    attachment: dict[str, Any],
    extracted_text: str,
    chunk_index: int,
    chunk_count: int,
) -> str:
    author = record.author_display_name or record.author_name or record.author_id
    filename = attachment.get("filename") or "unknown"
    content_type = attachment.get("content_type") or "unknown"
    local_path = attachment.get("local_path") or ""
    attachment_url = attachment.get("url") or ""
    message_content = record.clean_content or record.content or "(본문 없음)"

    lines = [
        "첨부파일 검색 문서",
        f"서버: {record.guild_name or record.guild_id}",
        f"카테고리: {record.category_name or 'no_category'}",
        f"채널: #{record.channel_name}",
        f"작성자: {author} ({record.author_name}, id={record.author_id})",
        f"작성 시간: {record.created_at}",
        f"연결 메시지 ID: {record.message_id}",
        f"연결 메시지: {message_content}",
        f"첨부파일명: {filename}",
        f"첨부파일 타입: {content_type}",
        f"첨부파일 크기: {attachment.get('size')}",
        f"첨부파일 local_path: {local_path}",
        f"첨부파일 url: {attachment_url}",
        f"첨부파일 chunk: {chunk_index + 1}/{chunk_count}",
    ]
    if extracted_text:
        lines.extend(["첨부파일 내용:", extracted_text])
    else:
        lines.append("첨부파일 내용: (텍스트 추출 없음)")
    if record.jump_url:
        lines.append(f"Discord 메시지 링크: {record.jump_url}")
    return "\n".join(lines)


def build_attachment_documents(
    records: list[MessageRecord],
    max_attachment_text_chars: int,
    attachment_chunk_tokens: int,
    attachment_chunk_overlap_tokens: int,
    max_attachment_chunk_chars: int,
) -> list[ChunkDocument]:
    documents: list[ChunkDocument] = []
    for record in records:
        for attachment in record.attachments:
            attachment_id = str(attachment.get("attachment_id") or attachment.get("url") or attachment.get("filename"))
            extracted_text = read_attachment_text(attachment, max_attachment_text_chars)
            text_chunks = (
                split_text(
                    extracted_text,
                    max_tokens=attachment_chunk_tokens,
                    overlap_tokens=attachment_chunk_overlap_tokens,
                )
                or [""]
            )
            chunk_count = len(text_chunks)

            for chunk_index, text_chunk in enumerate(text_chunks):
                filename = str(attachment.get("filename") or "unknown")
                content_type = str(attachment.get("content_type") or "")
                local_path = str(attachment.get("local_path") or "")
                attachment_url = str(attachment.get("url") or "")
                document_text = render_attachment_chunk(
                    record=record,
                    attachment=attachment,
                        extracted_text=text_chunk,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                )
                documents.append(
                    ChunkDocument(
                        chunk_id=f"{record.message_id}:{attachment_id}:attachment:{chunk_index}",
                        text=truncate_text(document_text, max_attachment_chunk_chars),
                        metadata={
                            "document_kind": "attachment",
                            "guild_id": record.guild_id or "",
                            "guild_name": record.guild_name or "",
                            "category_id": record.category_id or "",
                            "category_name": record.category_name or "no_category",
                            "channel_id": record.channel_id,
                            "channel_name": record.channel_name,
                            "message_id": record.message_id,
                            "author_id": record.author_id,
                            "author_name": record.author_name,
                            "author_display_name": record.author_display_name,
                            "created_at": record.created_at,
                            "jump_url": record.jump_url,
                            "attachment_id": attachment_id,
                            "filename": filename,
                            "content_type": content_type,
                            "local_path": local_path,
                            "attachment_url": attachment_url,
                            "chunk_index": chunk_index,
                            "chunk_count": chunk_count,
                            "has_extracted_text": bool(text_chunk),
                            "source_path": record.source_path,
                        },
                    )
                )
    return documents


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_collection(chroma_path: Path, collection_name: str, reset: bool = False) -> Any:
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def build_embeddings(client: OpenAI, model: str, texts: Sequence[str]) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def index_documents(
    documents: list[ChunkDocument],
    chroma_path: Path,
    openai_client: OpenAI,
    embedding_model: str,
    batch_size: int,
    collection_name: str,
) -> None:
    collection = get_collection(chroma_path, collection_name, reset=True)
    for batch in batched(documents, batch_size):
        embeddings = build_embeddings(openai_client, embedding_model, [document.text for document in batch])
        collection.upsert(
            ids=[document.chunk_id for document in batch],
            documents=[document.text for document in batch],
            embeddings=embeddings,
            metadatas=[document.metadata for document in batch],
        )


def parse_datetime_value(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_display_datetime(value: Any) -> str:
    parsed = parse_datetime_value(value)
    if parsed is None:
        return str(value or "").strip()
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")


def format_current_datetime(now: datetime | None = None) -> str:
    current = now or datetime.now(KST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=KST)
    return current.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")


def escape_markdown_label(label: str) -> str:
    return label.replace("[", "\\[").replace("]", "\\]")


def markdown_url_link(label: str, url: str) -> str:
    return f"[{escape_markdown_label(label)}]({url})"


def source_context_label(
    metadata: dict[str, Any],
    created_at_key: str,
    author_display_key: str,
    author_name_key: str,
) -> str:
    category_name = metadata.get("category_name") or "no_category"
    channel_name = metadata.get("channel_name") or ""
    created_at = format_display_datetime(metadata.get(created_at_key))
    author = metadata.get(author_display_key) or metadata.get(author_name_key) or ""
    return " / ".join(str(part) for part in [category_name, f"#{channel_name}", created_at, author] if part)


def format_message_source(metadata: dict[str, Any]) -> str:
    label = source_context_label(
        metadata,
        created_at_key="center_created_at",
        author_display_key="center_author_display_name",
        author_name_key="center_author_name",
    )
    jump_url = str(metadata.get("center_jump_url") or "")
    return markdown_url_link(label, jump_url) if jump_url else label


def format_message_context_source(metadata: dict[str, Any]) -> str:
    return source_context_label(
        metadata,
        created_at_key="center_created_at",
        author_display_key="center_author_display_name",
        author_name_key="center_author_name",
    )


def absolute_local_path(local_path: str) -> str:
    path = Path(local_path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path.resolve(strict=False))


def markdown_file_link(label: str, local_path: str) -> str:
    absolute_path = absolute_local_path(local_path)
    return f"[{escape_markdown_label(label)}](<{absolute_path}>)"


def format_attachment_source(metadata: dict[str, Any]) -> str:
    filename = metadata.get("filename") or "unknown"
    local_path = metadata.get("local_path") or ""
    jump_url = metadata.get("jump_url") or ""
    file_label = markdown_file_link(str(filename), str(local_path)) if local_path else str(filename)
    context_label = source_context_label(
        metadata,
        created_at_key="created_at",
        author_display_key="author_display_name",
        author_name_key="author_name",
    )
    parts = [file_label, context_label]
    if jump_url:
        parts.append(markdown_url_link("Discord 원문", str(jump_url)))
    return " - ".join(parts)


def format_attachment_context_source(metadata: dict[str, Any]) -> str:
    filename = metadata.get("filename") or "unknown"
    context_label = source_context_label(
        metadata,
        created_at_key="created_at",
        author_display_key="author_display_name",
        author_name_key="author_name",
    )
    return f"{filename} - {context_label}"


def query_collection(
    chroma_path: Path,
    collection_name: str,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[str], list[dict[str, Any]], list[float | None]]:
    collection = get_collection(chroma_path, collection_name)
    if collection.count() == 0:
        return [], [], []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return documents, metadatas, distances


def make_search_results(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float | None],
) -> list[SearchResult]:
    results = []
    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        results.append(SearchResult(document=document, metadata=metadata, distance=distance))
    return results


def message_result_key(result: SearchResult) -> str:
    metadata = result.metadata
    center_message_id = metadata.get("center_message_id")
    if center_message_id:
        return str(center_message_id)
    start_message_id = metadata.get("start_message_id")
    end_message_id = metadata.get("end_message_id")
    if start_message_id or end_message_id:
        return f"{start_message_id}:{end_message_id}"
    return result.document


def attachment_result_key(result: SearchResult) -> str:
    metadata = result.metadata
    message_id = metadata.get("message_id")
    attachment_id = metadata.get("attachment_id")
    chunk_index = metadata.get("chunk_index")
    if message_id or attachment_id or chunk_index is not None:
        return f"{message_id}:{attachment_id}:{chunk_index}"
    return result.document


def vector_similarity(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return 1.0 - float(distance)


def dedupe_search_results(
    results: Iterable[SearchResult],
    key_func: Callable[[SearchResult], str] = message_result_key,
) -> list[SearchResult]:
    best_by_key: dict[str, SearchResult] = {}
    for result in results:
        key = key_func(result)
        current = best_by_key.get(key)
        if current is None or vector_similarity(result.distance) > vector_similarity(current.distance):
            best_by_key[key] = result
    return list(best_by_key.values())


def normalize_search_term(term: str) -> str:
    normalized = term.strip().casefold()
    for suffix in KOREAN_PARTICLE_SUFFIXES:
        if len(normalized) > len(suffix) + 1 and normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def extract_search_terms(text: str, max_terms: int = 12) -> list[str]:
    candidates = re.findall(r"[0-9A-Za-z가-힣_./+-]{2,}", text)
    terms: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_search_term(candidate)
        if normalized in GENERIC_SEARCH_TERMS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= max_terms:
            break
    return terms


def search_document_text(document: str) -> str:
    skipped_prefixes = (
        "서버:",
        "- 시간:",
        "링크:",
        "local_path=",
        "첨부파일 local_path:",
        "첨부파일 url:",
        "Discord 메시지 링크:",
    )
    lines = []
    for line in document.splitlines():
        stripped = line.strip()
        if stripped.startswith(skipped_prefixes):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def is_temporal_query(query_text: str) -> bool:
    normalized = query_text.casefold()
    return any(keyword in normalized for keyword in TEMPORAL_QUERY_KEYWORDS)


def parsed_mentioned_dates(text: str, now: datetime) -> list[datetime]:
    dates: list[datetime] = []
    for match in DATE_MENTION_PATTERN.finditer(text):
        year_text, month_text, day_text = match.groups()
        year = int(year_text) if year_text else now.astimezone(KST).year
        month = int(month_text)
        day = int(day_text)
        try:
            parsed = datetime(year, month, day, tzinfo=KST)
        except ValueError:
            continue
        if not year_text and parsed.date() < now.astimezone(KST).date() - timedelta(days=30):
            try:
                parsed = parsed.replace(year=parsed.year + 1)
            except ValueError:
                continue
        dates.append(parsed)
    return dates


def future_date_score(text: str, now: datetime) -> float:
    today = now.astimezone(KST).date()
    scores = []
    for mentioned in parsed_mentioned_dates(text, now):
        days_until = (mentioned.date() - today).days
        if days_until < 0:
            continue
        scores.append(1.0 / (1.0 + days_until / 30.0))
    return max(scores, default=0.0)


def term_match_score(text: str, terms: Sequence[str], weight: float, max_per_term: int = 3) -> float:
    normalized_text = text.casefold()
    score = 0.0
    for term in terms:
        if not term:
            continue
        count = normalized_text.count(term.casefold())
        if count:
            score += weight * min(count, max_per_term)
    return score


def relevance_score(result: SearchResult, query_text: str, terms: Sequence[str], now: datetime) -> float:
    metadata_text = "\n".join(
        [
            str(result.metadata.get("category_name") or ""),
            str(result.metadata.get("channel_name") or ""),
            str(result.metadata.get("center_author_display_name") or ""),
            str(result.metadata.get("center_author_name") or ""),
        ]
    )
    body_text = search_document_text(result.document)
    score = 0.0
    score += term_match_score(metadata_text, terms, weight=4.0, max_per_term=1)
    score += term_match_score(body_text, terms, weight=1.5, max_per_term=3)
    if is_temporal_query(query_text):
        score += future_date_score(body_text, now) * 8.0
        score += recency_score(result.metadata, now) * 1.5
    else:
        score += recency_score(result.metadata, now) * 0.3
    return score


def rank_recent_results_by_relevance(
    results: Sequence[SearchResult],
    query_text: str,
    limit: int,
    now: datetime,
) -> list[SearchResult]:
    terms = extract_search_terms(query_text)
    if not terms:
        ranked = sorted(
            results,
            key=lambda result: parse_datetime_value(result.metadata.get("center_created_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return ranked[:limit]

    scored = [
        (relevance_score(result, query_text, terms, now), result)
        for result in results
    ]
    filtered = [(score, result) for score, result in scored if score > 0]
    filtered.sort(
        key=lambda item: (
            item[0],
            parse_datetime_value(item[1].metadata.get("center_created_at"))
            or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return [result for _score, result in filtered[:limit]]


def result_matches_terms(result: SearchResult, terms: Sequence[str]) -> bool:
    if not terms:
        return True
    searchable = "\n".join(
        [
            str(result.metadata.get("category_name") or ""),
            str(result.metadata.get("channel_name") or ""),
            search_document_text(result.document),
        ]
    ).casefold()
    return any(term.casefold() in searchable for term in terms)


def get_recent_message_results(
    chroma_path: Path,
    query_text: str,
    limit: int,
    now: datetime | None = None,
) -> list[SearchResult]:
    collection = get_collection(chroma_path, MESSAGE_COLLECTION_NAME)
    if collection.count() == 0:
        return []

    payload = collection.get(include=["documents", "metadatas"])
    documents = list(payload.get("documents") or [])
    metadatas = list(payload.get("metadatas") or [])
    results = make_search_results(documents, metadatas, [None] * len(documents))
    return rank_recent_results_by_relevance(
        results,
        query_text=query_text,
        limit=limit,
        now=now or datetime.now(KST),
    )


def recency_score(metadata: dict[str, Any], now: datetime) -> float:
    created_at = parse_datetime_value(
        metadata.get("center_created_at") or metadata.get("created_at") or metadata.get("end_created_at")
    )
    if created_at is None:
        return 0.0
    age_days = max(0.0, (now.astimezone(KST) - created_at.astimezone(KST)).total_seconds() / 86400)
    return 1.0 / (1.0 + age_days / 14.0)


def rerank_message_results(
    results: list[SearchResult],
    plan: RetrievalPlan,
    now: datetime,
) -> list[SearchResult]:
    if not plan.prefer_recent or plan.recency_weight <= 0:
        return results[: plan.final_top_k]

    return sorted(
        results,
        key=lambda result: (
            vector_similarity(result.distance) + plan.recency_weight * recency_score(result.metadata, now),
            vector_similarity(result.distance),
        ),
        reverse=True,
    )[: plan.final_top_k]


def strip_generated_source_sections(answer: str) -> str:
    source_headings = [
        "참고 메시지:",
        "참고한 메시지:",
        "참고 첨부파일:",
        "참고한 첨부파일:",
        "출처:",
        "Sources:",
    ]
    lines = answer.strip().splitlines()
    kept_lines = []
    for line in lines:
        normalized = line.strip().lstrip("#").strip()
        if any(normalized.startswith(heading) for heading in source_headings):
            break
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def sanitize_context_document(document: str) -> str:
    hidden_prefixes = (
        "local_path=",
        "첨부파일 local_path:",
        "첨부파일 url:",
        "Discord 메시지 링크:",
    )
    lines = []
    for line in document.splitlines():
        if line.strip().startswith(hidden_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines)


def format_agent_candidates(
    message_candidates: list[SearchResult],
    attachment_candidates: list[SearchResult],
    max_message_candidates: int = 24,
    max_attachment_candidates: int = 30,
    max_chars: int = 700,
) -> tuple[str, dict[str, SearchResult], dict[str, SearchResult]]:
    lines: list[str] = []
    message_by_key: dict[str, SearchResult] = {}
    attachment_by_key: dict[str, SearchResult] = {}

    if message_candidates:
        lines.append("[메시지 후보]")
    for rank, result in enumerate(message_candidates, start=1):
        if len(message_by_key) >= max_message_candidates:
            break
        key = message_result_key(result)
        if key in message_by_key:
            continue
        message_by_key[key] = result
        preview = truncate_text(sanitize_context_document(result.document), max_chars)
        lines.extend(
            [
                f"- key: {key}",
                f"  rank: {rank}",
                f"  source: {format_message_context_source(result.metadata)}",
                f"  distance: {result.distance}",
                "  preview: |",
                *[f"    {line}" for line in preview.splitlines()],
            ]
        )

    if attachment_candidates:
        lines.append("[첨부파일 후보]")
    for rank, result in enumerate(attachment_candidates, start=1):
        if len(attachment_by_key) >= max_attachment_candidates:
            break
        key = attachment_result_key(result)
        if key in attachment_by_key:
            continue
        attachment_by_key[key] = result
        preview = truncate_text(sanitize_context_document(result.document), max_chars)
        lines.extend(
            [
                f"- key: {key}",
                f"  rank: {rank}",
                f"  source: {format_attachment_context_source(result.metadata)}",
                f"  distance: {result.distance}",
                "  preview: |",
                *[f"    {line}" for line in preview.splitlines()],
            ]
        )

    return "\n".join(lines), message_by_key, attachment_by_key


def normalize_agent_action(action: Any, default_query: str, default_k: int) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"tool": "none", "query": "", "k": 0}

    tool = str(action.get("tool") or "none").strip()
    allowed_tools = {
        "none",
        "vector_search_messages",
        "recent_messages",
        "vector_search_attachments",
    }
    if tool not in allowed_tools:
        tool = "none"

    query = str(action.get("query") or default_query).strip()
    if tool == "none":
        return {"tool": "none", "query": "", "k": 0}
    if not query:
        query = default_query

    max_k = AGENT_ATTACHMENT_CANDIDATE_LIMIT if tool == "vector_search_attachments" else AGENT_MESSAGE_CANDIDATE_LIMIT
    min_k = 1 if tool == "vector_search_attachments" else 5
    k = clamp_int(action.get("k"), minimum=min_k, maximum=max_k, default=default_k)
    return {"tool": tool, "query": query, "k": k}


def parse_agent_judge_response(
    content: str,
    default_query: str,
    default_k: int,
) -> tuple[bool, list[str], list[str], dict[str, Any], str]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False, [], [], {"tool": "none", "query": "", "k": 0}, ""
    if not isinstance(payload, dict):
        return False, [], [], {"tool": "none", "query": "", "k": 0}, ""

    is_sufficient = parse_bool(payload.get("is_sufficient"), default=False)
    message_keys = payload.get("message_keys")
    attachment_keys = payload.get("attachment_keys")
    next_action = normalize_agent_action(payload.get("next_action"), default_query, default_k)
    if is_sufficient:
        next_action = {"tool": "none", "query": "", "k": 0}
    return (
        is_sufficient,
        [str(key) for key in message_keys[:6]] if isinstance(message_keys, list) else [],
        [str(key) for key in attachment_keys[:6]] if isinstance(attachment_keys, list) else [],
        next_action,
        str(payload.get("reason") or "").strip(),
    )


def select_results_by_keys(
    message_keys: Sequence[str],
    attachment_keys: Sequence[str],
    message_by_key: dict[str, SearchResult],
    attachment_by_key: dict[str, SearchResult],
    plan: RetrievalPlan,
) -> tuple[list[SearchResult], list[SearchResult]]:
    selected_messages = [message_by_key[key] for key in message_keys if key in message_by_key]
    selected_attachments = [attachment_by_key[key] for key in attachment_keys if key in attachment_by_key]
    return selected_messages[: plan.final_top_k], selected_attachments[: plan.final_top_k]


def action_signature(action: dict[str, Any]) -> str:
    return f"{action.get('tool')}:{str(action.get('query') or '').casefold()}:{action.get('k')}"


def action_was_run(actions: Sequence[dict[str, Any]], action: dict[str, Any]) -> bool:
    signature = action_signature(action)
    return any(action_signature(existing) == signature for existing in actions)


def fallback_agent_action(state: AgenticRagState) -> dict[str, Any]:
    plan = state["plan"]
    actions = state.get("actions", [])
    executed_tools = {str(action.get("tool") or "") for action in actions}
    expanded_query = f"{state['question']} {plan.search_query}".strip()

    if plan.prefer_recent and "recent_messages" not in executed_tools:
        return {
            "tool": "recent_messages",
            "query": expanded_query,
            "k": max(plan.message_candidate_top_k, plan.final_top_k * 8, 40),
        }
    if plan.include_attachments and "vector_search_attachments" not in executed_tools:
        return {
            "tool": "vector_search_attachments",
            "query": plan.search_query,
            "k": max(plan.attachment_candidate_top_k, plan.final_top_k),
        }
    if "recent_messages" not in executed_tools:
        return {
            "tool": "recent_messages",
            "query": expanded_query,
            "k": max(plan.message_candidate_top_k, plan.final_top_k * 8, 40),
        }
    if "vector_search_attachments" not in executed_tools and is_attachment_query(state["question"]):
        return {
            "tool": "vector_search_attachments",
            "query": plan.search_query,
            "k": max(plan.attachment_candidate_top_k, plan.final_top_k),
        }
    return {"tool": "none", "query": "", "k": 0}


def initial_agent_action(plan: RetrievalPlan) -> dict[str, Any]:
    return {
        "tool": "vector_search_messages",
        "query": plan.search_query,
        "k": plan.message_candidate_top_k,
    }


def get_vector_search_results(
    chroma_path: Path,
    openai_client: OpenAI,
    embedding_model: str,
    collection_name: str,
    query: str,
    top_k: int,
) -> list[SearchResult]:
    if top_k <= 0 or not query.strip():
        return []
    query_embedding = build_embeddings(openai_client, embedding_model, [query])[0]
    documents, metadatas, distances = query_collection(
        chroma_path,
        collection_name,
        query_embedding,
        top_k,
    )
    return make_search_results(documents, metadatas, distances)


def merge_candidates(
    existing: list[SearchResult],
    new: list[SearchResult],
    limit: int,
    key_func: Callable[[SearchResult], str],
) -> list[SearchResult]:
    return dedupe_search_results(new + existing, key_func=key_func)[:limit]


def format_agent_actions(actions: Sequence[dict[str, Any]]) -> str:
    if not actions:
        return "없음"
    return "\n".join(
        f"{index}. tool={action.get('tool')} query={action.get('query')} k={action.get('k')}"
        for index, action in enumerate(actions, start=1)
    )


def run_agentic_retrieval(
    question: str,
    chroma_path: Path,
    openai_client: OpenAI,
    embedding_model: str,
    planner_model: str,
    selector_model: str,
    top_k: int,
    use_planner: bool,
) -> tuple[RetrievalPlan, list[SearchResult], list[SearchResult], str]:
    current_now = datetime.now(KST)
    current_datetime = format_current_datetime(current_now)

    def plan_node(state: AgenticRagState) -> AgenticRagState:
        plan = build_retrieval_plan(
            question=state["question"],
            current_datetime=state["current_datetime"],
            openai_client=openai_client,
            planner_model=planner_model,
            final_top_k=top_k,
            use_planner=use_planner,
        )
        return {
            "plan": plan,
            "next_action": initial_agent_action(plan),
        }

    def act_node(state: AgenticRagState) -> AgenticRagState:
        plan = state["plan"]
        action = normalize_agent_action(
            state.get("next_action"),
            default_query=plan.search_query,
            default_k=plan.message_candidate_top_k,
        )
        step = state.get("step", 0) + 1
        actions = [*state.get("actions", []), action]
        message_candidates = state.get("message_candidates", [])
        attachment_candidates = state.get("attachment_candidates", [])
        message_limit = max(plan.message_candidate_top_k, plan.final_top_k * 10, 40)
        attachment_limit = max(plan.attachment_candidate_top_k, plan.final_top_k * 5, 10)

        if action["tool"] == "vector_search_messages":
            new_messages = get_vector_search_results(
                chroma_path=chroma_path,
                openai_client=openai_client,
                embedding_model=embedding_model,
                collection_name=MESSAGE_COLLECTION_NAME,
                query=str(action["query"]),
                top_k=int(action["k"]),
            )
            message_candidates = merge_candidates(
                existing=message_candidates,
                new=new_messages,
                limit=message_limit,
                key_func=message_result_key,
            )
        elif action["tool"] == "recent_messages":
            new_messages = get_recent_message_results(
                chroma_path=chroma_path,
                query_text=str(action["query"]),
                limit=int(action["k"]),
                now=current_now,
            )
            message_candidates = merge_candidates(
                existing=message_candidates,
                new=new_messages,
                limit=message_limit,
                key_func=message_result_key,
            )
        elif action["tool"] == "vector_search_attachments":
            new_attachments = get_vector_search_results(
                chroma_path=chroma_path,
                openai_client=openai_client,
                embedding_model=embedding_model,
                collection_name=ATTACHMENT_COLLECTION_NAME,
                query=str(action["query"]),
                top_k=int(action["k"]),
            )
            attachment_candidates = merge_candidates(
                existing=attachment_candidates,
                new=new_attachments,
                limit=attachment_limit,
                key_func=attachment_result_key,
            )

        return {
            "step": step,
            "actions": actions,
            "message_candidates": message_candidates,
            "attachment_candidates": attachment_candidates,
        }

    def judge_node(state: AgenticRagState) -> AgenticRagState:
        plan = state["plan"]
        message_candidates = state.get("message_candidates", [])
        attachment_candidates = state.get("attachment_candidates", [])
        candidate_text, message_by_key, attachment_by_key = format_agent_candidates(
            message_candidates,
            attachment_candidates,
        )
        default_next_action = fallback_agent_action(state)

        if not candidate_text.strip():
            if state.get("step", 0) >= state.get("max_steps", AGENT_MAX_STEPS):
                return {
                    "is_sufficient": False,
                    "message_results": [],
                    "attachment_results": [],
                    "next_action": {"tool": "none", "query": "", "k": 0},
                    "judge_reason": "검색 후보가 없습니다.",
                }
            return {
                "is_sufficient": False,
                "next_action": default_next_action,
                "judge_reason": "검색 후보가 부족해 추가 검색합니다.",
            }

        try:
            response = openai_client.chat.completions.create(
                model=selector_model,
                messages=[
                    {"role": "system", "content": AGENT_JUDGE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"[현재 시각]\n{state['current_datetime']}\n\n"
                            f"[사용자 질문]\n{state['question']}\n\n"
                            f"[검색 계획]\n"
                            f"search_query={plan.search_query}\n"
                            f"include_attachments={plan.include_attachments}\n"
                            f"prefer_recent={plan.prefer_recent}\n"
                            f"planner_reason={plan.reason}\n\n"
                            f"[이미 실행한 검색]\n{format_agent_actions(state.get('actions', []))}\n\n"
                            f"[근거 후보]\n{candidate_text}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            is_sufficient, message_keys, attachment_keys, next_action, reason = parse_agent_judge_response(
                response.choices[0].message.content or "{}",
                default_query=plan.search_query,
                default_k=plan.message_candidate_top_k,
            )
        except Exception:
            is_sufficient, message_keys, attachment_keys, next_action, reason = (
                False,
                [],
                [],
                default_next_action,
                "judge 호출 실패로 fallback 검색을 사용합니다.",
            )

        if is_sufficient:
            selected_messages, selected_attachments = select_results_by_keys(
                message_keys,
                attachment_keys,
                message_by_key,
                attachment_by_key,
                plan,
            )
            if not selected_messages and not selected_attachments:
                selected_messages = message_candidates[: plan.final_top_k]
                selected_attachments = attachment_candidates[: plan.final_top_k]
            return {
                "is_sufficient": True,
                "message_results": selected_messages,
                "attachment_results": selected_attachments,
                "next_action": {"tool": "none", "query": "", "k": 0},
                "judge_reason": reason,
            }

        if state.get("step", 0) >= state.get("max_steps", AGENT_MAX_STEPS):
            return {
                "is_sufficient": False,
                "message_results": message_candidates[: plan.final_top_k],
                "attachment_results": attachment_candidates[: plan.final_top_k],
                "next_action": {"tool": "none", "query": "", "k": 0},
                "judge_reason": reason or "최대 검색 횟수 안에서 충분한 근거를 찾지 못했습니다.",
            }

        if next_action["tool"] == "none" or action_was_run(state.get("actions", []), next_action):
            next_action = default_next_action

        if next_action["tool"] == "none":
            return {
                "is_sufficient": False,
                "message_results": message_candidates[: plan.final_top_k],
                "attachment_results": attachment_candidates[: plan.final_top_k],
                "next_action": {"tool": "none", "query": "", "k": 0},
                "judge_reason": reason or "추가 검색 도구가 없어 정렬된 후보를 사용합니다.",
            }

        return {
            "is_sufficient": False,
            "next_action": next_action,
            "judge_reason": reason,
        }

    def route_after_judge(state: AgenticRagState) -> str:
        if state.get("is_sufficient"):
            return "end"
        if state.get("step", 0) >= state.get("max_steps", AGENT_MAX_STEPS):
            return "end"
        if state.get("next_action", {}).get("tool") == "none":
            return "end"
        return "act"

    graph = StateGraph(AgenticRagState)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("judge", judge_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "judge")
    graph.add_conditional_edges("judge", route_after_judge, {"act": "act", "end": END})

    final_state = graph.compile().invoke(
        {
            "question": question,
            "current_datetime": current_datetime,
            "step": 0,
            "max_steps": AGENT_MAX_STEPS,
            "actions": [],
            "message_candidates": [],
            "attachment_candidates": [],
        }
    )
    return (
        final_state["plan"],
        final_state.get("message_results", []),
        final_state.get("attachment_results", []),
        current_datetime,
    )


def ask_question(
    question: str,
    chroma_path: Path,
    openai_client: OpenAI,
    embedding_model: str,
    chat_model: str,
    top_k: int,
    planner_model: str | None = None,
    use_planner: bool = True,
) -> str:
    plan, message_results, attachment_results, current_datetime = run_agentic_retrieval(
        question=question,
        chroma_path=chroma_path,
        openai_client=openai_client,
        embedding_model=embedding_model,
        planner_model=planner_model or chat_model,
        selector_model=planner_model or chat_model,
        top_k=top_k,
        use_planner=use_planner,
    )
    message_documents = [result.document for result in message_results]
    message_metadatas = [result.metadata for result in message_results]
    message_distances = [result.distance for result in message_results]
    attachment_documents = [result.document for result in attachment_results]
    attachment_metadatas = [result.metadata for result in attachment_results]
    attachment_distances = [result.distance for result in attachment_results]

    if not message_documents and not attachment_documents:
        return "대화 기록에서 확인되지 않습니다.\n\n---\n\n### 참고 메시지\n- 없음\n\n### 참고 첨부파일\n- 없음"

    context_blocks = []
    for index, document in enumerate(message_documents, start=1):
        metadata = message_metadatas[index - 1]
        distance = message_distances[index - 1] if index - 1 < len(message_distances) else None
        context_blocks.append(
            "\n".join(
                [
                    f"[메시지 검색 결과 {index}]",
                    f"source={format_message_context_source(metadata)}",
                    f"distance={distance}",
                    sanitize_context_document(document),
                ]
            )
        )
    for index, document in enumerate(attachment_documents, start=1):
        metadata = attachment_metadatas[index - 1]
        distance = attachment_distances[index - 1] if index - 1 < len(attachment_distances) else None
        context_blocks.append(
            "\n".join(
                [
                    f"[첨부파일 검색 결과 {index}]",
                    f"source={format_attachment_context_source(metadata)}",
                    f"distance={distance}",
                    sanitize_context_document(document),
                ]
            )
        )
    context = "\n\n".join(context_blocks)

    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "아래 Discord 검색 결과만 근거로 사용자 질문에 답하세요.\n"
                    "오늘, 이번 주, 다음, 앞으로 같은 시간 표현은 현재 시각을 기준으로 해석하세요.\n"
                    "답변 본문만 작성하고 참고 메시지, 참고 첨부파일, 출처 섹션은 작성하지 마세요.\n\n"
                    f"[현재 시각]\n{current_datetime}\n\n"
                    f"[검색 계획]\n"
                    f"search_query={plan.search_query}\n"
                    f"prefer_recent={plan.prefer_recent}\n"
                    f"planner_reason={plan.reason}\n\n"
                    f"[검색 결과]\n{context}\n\n"
                    f"[사용자 질문]\n{question}"
                ),
            },
        ],
        temperature=0.1,
    )
    answer = strip_generated_source_sections(
        response.choices[0].message.content or "대화 기록에서 확인되지 않습니다."
    )

    source_lines = []
    seen: set[str] = set()
    for metadata in message_metadatas:
        message_id = str(metadata.get("center_message_id") or "")
        if message_id in seen:
            continue
        seen.add(message_id)
        source_lines.append(f"{len(source_lines) + 1}. {format_message_source(metadata)}")

    attachment_source_lines = []
    seen_attachments: set[str] = set()
    for metadata in attachment_metadatas:
        key = f"{metadata.get('message_id')}:{metadata.get('attachment_id')}"
        if key in seen_attachments:
            continue
        seen_attachments.add(key)
        attachment_source_lines.append(
            f"{len(attachment_source_lines) + 1}. {format_attachment_source(metadata)}"
        )

    message_sources = "\n".join(source_lines) if source_lines else "- 없음"
    attachment_section = ""
    if plan.include_attachments or attachment_source_lines:
        attachment_sources = "\n".join(attachment_source_lines) if attachment_source_lines else "- 없음"
        attachment_section = f"\n\n### 참고 첨부파일\n{attachment_sources}"

    return f"{answer.strip()}\n\n---\n\n### 참고 메시지\n{message_sources}{attachment_section}"


def build_index_main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Build a Chroma RAG index from Discord JSONL exports.")
    parser.add_argument("--raw-dir", default=os.getenv("DISCORD_RAW_DIR", str(DEFAULT_RAW_DIR)))
    parser.add_argument("--chroma-dir", default=os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_DIR)))
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--window-before", type=int, default=parse_int_env("RAG_WINDOW_BEFORE", 2))
    parser.add_argument("--window-after", type=int, default=parse_int_env("RAG_WINDOW_AFTER", 2))
    parser.add_argument("--batch-size", type=int, default=parse_int_env("RAG_EMBED_BATCH_SIZE", 64))
    parser.add_argument(
        "--max-attachment-chars",
        type=int,
        default=parse_int_env("RAG_MESSAGE_MAX_ATTACHMENT_CHARS", 0),
        help="Attachment text chars to inline into message window chunks. Keep 0 when using attachment chunks.",
    )
    parser.add_argument(
        "--max-attachment-text-chars",
        type=int,
        default=parse_int_env("RAG_MAX_ATTACHMENT_TEXT_CHARS", 50000),
    )
    parser.add_argument(
        "--attachment-chunk-tokens",
        type=int,
        default=parse_int_env("RAG_ATTACHMENT_CHUNK_TOKENS", DEFAULT_ATTACHMENT_CHUNK_TOKENS),
    )
    parser.add_argument(
        "--attachment-chunk-overlap-tokens",
        type=int,
        default=parse_int_env("RAG_ATTACHMENT_CHUNK_OVERLAP_TOKENS", DEFAULT_ATTACHMENT_CHUNK_OVERLAP_TOKENS),
    )
    parser.add_argument(
        "--max-attachment-chunk-chars",
        type=int,
        default=parse_int_env("RAG_MAX_ATTACHMENT_CHUNK_CHARS", 4000),
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=parse_int_env("RAG_MAX_CHUNK_CHARS", 4000),
    )
    args = parser.parse_args(argv)

    try:
        api_key = require_openai_api_key()
    except RagError as exc:
        raise SystemExit(str(exc)) from exc
    raw_dir = Path(args.raw_dir)
    records = load_message_records(raw_dir)
    if not records:
        raise SystemExit(f"No JSONL message records found under {raw_dir}")

    documents = build_chunk_documents(
        records,
        before=args.window_before,
        after=args.window_after,
        max_attachment_chars=args.max_attachment_chars,
        max_chunk_chars=args.max_chunk_chars,
    )
    attachment_documents = build_attachment_documents(
        records,
        max_attachment_text_chars=args.max_attachment_text_chars,
        attachment_chunk_tokens=args.attachment_chunk_tokens,
        attachment_chunk_overlap_tokens=args.attachment_chunk_overlap_tokens,
        max_attachment_chunk_chars=args.max_attachment_chunk_chars,
    )
    print(
        f"Loaded {len(records)} messages. "
        f"Built {len(documents)} message window chunks and {len(attachment_documents)} attachment chunks."
    )

    index_documents(
        documents=documents,
        chroma_path=Path(args.chroma_dir),
        openai_client=OpenAI(api_key=api_key),
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        collection_name=MESSAGE_COLLECTION_NAME,
    )
    index_documents(
        documents=attachment_documents,
        chroma_path=Path(args.chroma_dir),
        openai_client=OpenAI(api_key=api_key),
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        collection_name=ATTACHMENT_COLLECTION_NAME,
    )
    print(
        f"Indexed {len(documents)} message chunks and "
        f"{len(attachment_documents)} attachment chunks into {args.chroma_dir}."
    )


def ask_main(argv: Sequence[str] | None = None) -> None:
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(description="Ask a question against the local Discord RAG index.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--chroma-dir", default=os.getenv("CHROMA_PATH", str(DEFAULT_CHROMA_DIR)))
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--chat-model", default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--planner-model", default=os.getenv("OPENAI_PLANNER_MODEL") or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--top-k", type=int, default=parse_int_env("RAG_TOP_K", 6))
    parser.add_argument("--no-planner", action="store_true", help="Disable the LLM retrieval planner.")
    args = parser.parse_args(argv)

    try:
        api_key = require_openai_api_key()
    except RagError as exc:
        raise SystemExit(str(exc)) from exc
    question = " ".join(args.question).strip()
    answer = ask_question(
        question=question,
        chroma_path=Path(args.chroma_dir),
        openai_client=OpenAI(api_key=api_key),
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        top_k=args.top_k,
        planner_model=args.planner_model,
        use_planner=not args.no_planner,
    )
    print(answer)


if __name__ == "__main__":
    build_index_main()
