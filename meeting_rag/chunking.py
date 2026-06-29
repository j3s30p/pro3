from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from .parsing import DEFAULT_INPUT_PATH, parse_clova_response


MAX_CHARS = 1200


def format_utterance(row: Dict[str, Any]) -> str:
    """발화 한 개를 청크에 들어갈 텍스트 한 줄로 바꾼다."""
    text = str(row["text"]).replace("\n", " ").strip()
    return f"[{row['start_time']}] {row['speaker_name']}: {text}"


def build_chunk(chunk_index: int, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """발화 행 목록을 하나의 청크 dict로 만든다."""
    speakers = sorted({str(row["speaker_name"]) for row in rows})
    lines = [format_utterance(row) for row in rows]

    return {
        "chunk_index": chunk_index,
        "start_time": rows[0]["start_time"],
        "end_time": rows[-1]["end_time"],
        "start_ms": rows[0]["start_ms"],
        "end_ms": rows[-1]["end_ms"],
        "utterance_start_index": rows[0]["index"],
        "utterance_end_index": rows[-1]["index"],
        "speakers": speakers,
        "text": "\n".join(lines),
    }


def chunk_utterances(rows: List[Dict[str, Any]], max_chars: int = MAX_CHARS) -> List[Dict[str, Any]]:
    """발화 목록을 문자 수 기준으로 순서대로 묶는다."""
    chunks: List[Dict[str, Any]] = []
    current_rows: List[Dict[str, Any]] = []
    current_chars = 0

    for row in rows:
        line = format_utterance(row)
        next_chars = current_chars + len(line) + 1

        if current_rows and next_chars > max_chars:
            chunks.append(build_chunk(len(chunks), current_rows))
            current_rows = []
            current_chars = 0

        current_rows.append(row)
        current_chars += len(line) + 1

    if current_rows:
        chunks.append(build_chunk(len(chunks), current_rows))

    return chunks


def print_chunks(chunks: List[Dict[str, Any]]) -> None:
    """수동 확인을 위해 앞쪽 청크 일부를 출력한다."""
    print(f"chunks: {len(chunks)}")
    print()

    for chunk in chunks[:3]:
        preview = chunk["text"].replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:217] + "..."

        print(
            f"[chunk {chunk['chunk_index']:03d}] "
            f"{chunk['start_time']}~{chunk['end_time']} "
            f"utterances {chunk['utterance_start_index']}~{chunk['utterance_end_index']}"
        )
        print(preview)
        print()


def main() -> int:
    """CLOVA JSON 파일을 읽고 발화 기반 청크를 출력한다."""
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    rows = parse_clova_response(input_path)
    chunks = chunk_utterances(rows)

    print(f"input: {input_path}")
    print(f"utterances: {len(rows)}")
    print_chunks(chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
