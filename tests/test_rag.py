from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from discord_crawler.rag import (
    MessageRecord,
    build_attachment_documents,
    build_chunk_documents,
    is_attachment_query,
    read_attachment_text,
    render_message,
    split_text,
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
