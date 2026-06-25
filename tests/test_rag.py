from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path

from discord_crawler.rag import (
    MessageRecord,
    RetrievalPlan,
    SearchResult,
    build_attachment_documents,
    build_chunk_documents,
    fallback_retrieval_plan,
    format_attachment_source,
    format_current_datetime,
    format_message_source,
    is_attachment_query,
    normalize_retrieval_plan,
    read_attachment_text,
    render_message,
    rerank_message_results,
    sanitize_context_document,
    split_text,
    strip_generated_source_sections,
    truncate_text,
)


def make_record(message_id: str, content: str, author: str = "제섭이") -> MessageRecord:
    return MessageRecord(
        guild_id="1",
        guild_name="스터디방",
        category_id="10",
        category_name="광장",
        channel_id="100",
        channel_name="잡담방",
        message_id=message_id,
        author_id="500",
        author_name="jesepark",
        author_display_name=author,
        created_at=f"2026-06-25T00:00:0{message_id}+00:00",
        edited_at=None,
        content=content,
        clean_content=content,
        jump_url=f"https://discord.com/channels/1/100/{message_id}",
        attachments=[],
        embeds=[],
        source_path="data/raw/test.jsonl",
    )


def test_build_chunk_documents_uses_neighbor_window() -> None:
    records = [
        make_record("1", "처음"),
        make_record("2", "중심"),
        make_record("3", "마지막"),
    ]

    chunks = build_chunk_documents(records, before=1, after=1, max_attachment_chars=1000, max_chunk_chars=10000)

    center_chunk = chunks[1]
    assert "처음" in center_chunk.text
    assert "중심" in center_chunk.text
    assert "마지막" in center_chunk.text
    assert center_chunk.metadata["message_ids"] == "1,2,3"


def test_render_message_includes_embed_text() -> None:
    record = make_record("1", "링크 참고")
    record = replace(
        record,
        embeds=[{"title": "문서", "description": "설명", "url": "https://example.com", "text": "문서\n설명"}],
    )

    rendered = render_message(record, max_attachment_chars=1000)

    assert "임베드" in rendered
    assert "문서" in rendered
    assert "설명" in rendered


def test_read_attachment_text_for_text_file(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("첨부 내용", encoding="utf-8")

    text = read_attachment_text(
        {"local_path": str(path), "filename": "note.md", "content_type": "text/markdown"},
        max_chars=100,
    )

    assert text == "첨부 내용"


def test_truncate_text_adds_marker() -> None:
    assert truncate_text("abcdef", max_chars=3) == "abc\n...(truncated)"


def test_is_attachment_query_detects_file_intent() -> None:
    assert is_attachment_query("제섭이가 올린 CNN html 파일 찾아줘") is True
    assert is_attachment_query("첨부파일 중 pdf 있었어?") is True
    assert is_attachment_query("저번에 제섭이가 뭐라고 했지?") is False


def test_normalize_retrieval_plan_clamps_and_preserves_signals() -> None:
    plan = normalize_retrieval_plan(
        {
            "search_query": "기술면접 다음 주제 일정 발표",
            "include_attachments": False,
            "prefer_recent": True,
            "recency_weight": 0.1,
            "message_candidate_top_k": 500,
            "attachment_candidate_top_k": 20,
            "reason": "다음 주제를 묻는 시간성 질문",
        },
        question="기술면접 다음 주제가 뭐야?",
        final_top_k=6,
    )

    assert plan.search_query == "기술면접 다음 주제 일정 발표"
    assert plan.include_attachments is False
    assert plan.prefer_recent is True
    assert plan.recency_weight == 0.5
    assert plan.message_candidate_top_k == 80
    assert plan.attachment_candidate_top_k == 0
    assert plan.source == "llm"


def test_fallback_retrieval_plan_uses_attachment_keyword_rule() -> None:
    plan = fallback_retrieval_plan("CNN html 파일 찾아줘", final_top_k=6)

    assert plan.include_attachments is True
    assert plan.search_query == "CNN html 파일 찾아줘"
    assert plan.message_candidate_top_k == 6
    assert plan.attachment_candidate_top_k == 6


def test_rerank_message_results_prefers_recent_when_planned() -> None:
    plan = RetrievalPlan(
        search_query="기술면접 다음 주제",
        include_attachments=False,
        prefer_recent=True,
        recency_weight=0.8,
        message_candidate_top_k=40,
        attachment_candidate_top_k=0,
        final_top_k=2,
        reason="시간성 질문",
        source="llm",
    )
    older = SearchResult(
        document="오래된 개설 공지",
        metadata={"center_created_at": "2026-06-01T00:00:00+00:00"},
        distance=0.50,
    )
    recent = SearchResult(
        document="최근 다음 주제 공지",
        metadata={"center_created_at": "2026-06-24T00:00:00+00:00"},
        distance=0.62,
    )

    reranked = rerank_message_results(
        [older, recent],
        plan=plan,
        now=datetime(2026, 6, 25, 0, 0, tzinfo=timezone.utc),
    )

    assert reranked[0].document == "최근 다음 주제 공지"


def test_format_attachment_source_uses_clickable_local_file_link(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("첨부 내용", encoding="utf-8")

    source = format_attachment_source(
        {
            "category_name": "광장",
            "channel_name": "자료",
            "created_at": "2026-06-25T00:00:00+00:00",
            "author_display_name": "제섭이",
            "filename": "note.md",
            "local_path": str(path),
            "jump_url": "https://discord.com/channels/1/2/3",
        }
    )

    assert f"[note.md](<{path}>)" in source
    assert "광장 / #자료 / 2026-06-25 09:00 KST / 제섭이" in source
    assert "[Discord 원문](https://discord.com/channels/1/2/3)" in source
    assert "local_path=" not in source


def test_format_message_source_uses_clickable_discord_link() -> None:
    source = format_message_source(
        {
            "category_name": "기술면접",
            "channel_name": "잡답방",
            "center_created_at": "2026-06-24T10:24:26+00:00",
            "center_author_display_name": "해연",
            "center_jump_url": "https://discord.com/channels/1/2/3",
        }
    )

    assert source == "[기술면접 / #잡답방 / 2026-06-24 19:24 KST / 해연](https://discord.com/channels/1/2/3)"


def test_format_current_datetime_uses_kst() -> None:
    now = datetime(2026, 6, 25, 0, 30, tzinfo=timezone.utc)

    assert format_current_datetime(now) == "2026-06-25 09:30 KST"


def test_strip_generated_source_sections_removes_duplicate_llm_sources() -> None:
    answer = """CNN HTML 파일은 두 개입니다.

참고 메시지:
1. 중복 출처
"""

    assert strip_generated_source_sections(answer) == "CNN HTML 파일은 두 개입니다."


def test_sanitize_context_document_hides_paths_and_urls() -> None:
    document = """첨부파일명: CNN.html
첨부파일 local_path: /tmp/CNN.html
첨부파일 url: https://cdn.example/CNN.html
Discord 메시지 링크: https://discord.com/channels/1/2/3
첨부파일 내용:
CNN 설명
"""

    sanitized = sanitize_context_document(document)

    assert "첨부파일명: CNN.html" in sanitized
    assert "첨부파일 내용:" in sanitized
    assert "local_path" not in sanitized
    assert "https://" not in sanitized


def test_split_text_chunks_long_text() -> None:
    assert split_text("사과 바나나 포도", max_tokens=1) == ["사과", "바나나", "포도"]


def test_build_attachment_documents_indexes_text_attachment(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("첨부 내용 abcdef", encoding="utf-8")
    record = replace(
        make_record("1", "파일 올렸어요"),
        attachments=[
            {
                "attachment_id": "att-1",
                "filename": "note.md",
                "content_type": "text/markdown",
                "size": 100,
                "local_path": str(path),
                "url": "https://cdn.example/note.md",
            }
        ],
    )

    docs = build_attachment_documents(
        [record],
        max_attachment_text_chars=1000,
        attachment_chunk_tokens=2,
        attachment_chunk_overlap_tokens=0,
        max_attachment_chunk_chars=4000,
    )

    assert len(docs) == 2
    assert docs[0].metadata["filename"] == "note.md"
    assert docs[0].metadata["has_extracted_text"] is True
    assert "첨부파일 내용" in docs[0].text


def test_build_attachment_documents_indexes_binary_attachment_metadata() -> None:
    record = replace(
        make_record("1", "이미지 올렸어요"),
        attachments=[
            {
                "attachment_id": "att-2",
                "filename": "image.png",
                "content_type": "image/png",
                "size": 100,
                "local_path": "/tmp/image.png",
                "url": "https://cdn.example/image.png",
            }
        ],
    )

    docs = build_attachment_documents(
        [record],
        max_attachment_text_chars=1000,
        attachment_chunk_tokens=100,
        attachment_chunk_overlap_tokens=0,
        max_attachment_chunk_chars=4000,
    )

    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "image.png"
    assert docs[0].metadata["has_extracted_text"] is False
    assert "텍스트 추출 없음" in docs[0].text
