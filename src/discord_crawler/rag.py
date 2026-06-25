from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_CHROMA_DIR = ROOT_DIR / "data" / "chroma"
MESSAGE_COLLECTION_NAME = "discord_message_windows"
ATTACHMENT_COLLECTION_NAME = "discord_attachment_chunks"
DEFAULT_ATTACHMENT_CHUNK_TOKENS = 600
DEFAULT_ATTACHMENT_CHUNK_OVERLAP_TOKENS = 80

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
- 누가, 언제, 어느 채널에서 말했는지 가능한 한 구체적으로 정리합니다.
- 첨부파일 내용은 검색 결과에 추출된 텍스트나 파일명으로 확인되는 범위에서만 언급합니다.
- 답변 끝의 참고 메시지 목록은 유지합니다.
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


def format_message_source(metadata: dict[str, Any]) -> str:
    category_name = metadata.get("category_name") or "no_category"
    channel_name = metadata.get("channel_name") or ""
    created_at = str(metadata.get("center_created_at") or "")
    author = metadata.get("center_author_display_name") or metadata.get("center_author_name") or ""
    jump_url = metadata.get("center_jump_url") or ""
    suffix = f" - {jump_url}" if jump_url else ""
    return f"[{category_name} / #{channel_name} / {created_at} / {author}]{suffix}"


def format_attachment_source(metadata: dict[str, Any]) -> str:
    category_name = metadata.get("category_name") or "no_category"
    channel_name = metadata.get("channel_name") or ""
    created_at = str(metadata.get("created_at") or "")
    author = metadata.get("author_display_name") or metadata.get("author_name") or ""
    filename = metadata.get("filename") or "unknown"
    local_path = metadata.get("local_path") or ""
    jump_url = metadata.get("jump_url") or ""
    parts = [f"{filename}", f"[{category_name} / #{channel_name} / {created_at} / {author}]"]
    if local_path:
        parts.append(f"local_path={local_path}")
    if jump_url:
        parts.append(str(jump_url))
    return " - ".join(parts)


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


def ask_question(
    question: str,
    chroma_path: Path,
    openai_client: OpenAI,
    embedding_model: str,
    chat_model: str,
    top_k: int,
) -> str:
    query_embedding = build_embeddings(openai_client, embedding_model, [question])[0]
    include_attachment_results = is_attachment_query(question)
    message_documents, message_metadatas, message_distances = query_collection(
        chroma_path,
        MESSAGE_COLLECTION_NAME,
        query_embedding,
        top_k,
    )
    if include_attachment_results:
        attachment_documents, attachment_metadatas, attachment_distances = query_collection(
            chroma_path,
            ATTACHMENT_COLLECTION_NAME,
            query_embedding,
            top_k,
        )
    else:
        attachment_documents, attachment_metadatas, attachment_distances = [], [], []

    if not message_documents and not attachment_documents:
        return "대화 기록에서 확인되지 않습니다.\n\n참고 메시지:\n- 없음\n\n참고 첨부파일:\n- 없음"

    context_blocks = []
    for index, document in enumerate(message_documents, start=1):
        metadata = message_metadatas[index - 1]
        distance = message_distances[index - 1] if index - 1 < len(message_distances) else None
        context_blocks.append(
            "\n".join(
                [
                    f"[메시지 검색 결과 {index}]",
                    f"source={format_message_source(metadata)}",
                    f"distance={distance}",
                    document,
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
                    f"source={format_attachment_source(metadata)}",
                    f"distance={distance}",
                    document,
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
                    "아래 Discord 검색 결과만 근거로 사용자 질문에 답하세요.\n\n"
                    f"[검색 결과]\n{context}\n\n"
                    f"[사용자 질문]\n{question}"
                ),
            },
        ],
        temperature=0.1,
    )
    answer = response.choices[0].message.content or "대화 기록에서 확인되지 않습니다."

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
    if include_attachment_results:
        attachment_sources = "\n".join(attachment_source_lines) if attachment_source_lines else "- 없음"
        attachment_section = f"\n\n참고 첨부파일:\n{attachment_sources}"

    return f"{answer.strip()}\n\n참고 메시지:\n{message_sources}{attachment_section}"


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
    parser.add_argument("--top-k", type=int, default=parse_int_env("RAG_TOP_K", 6))
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
    )
    print(answer)


if __name__ == "__main__":
    build_index_main()
