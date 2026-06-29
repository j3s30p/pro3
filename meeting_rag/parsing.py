from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT_PATH = Path(
    "data/raw/clova_note/clova_note_project_direction_2026-06-29.converted-clova-response.json"
)


def format_ms(milliseconds: Optional[int]) -> str:
    """CLOVA의 밀리초 단위 시간을 화면 표시용 문자열로 바꾼다."""
    if milliseconds is None:
        return "-"

    total_seconds = max(milliseconds, 0) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def to_int(value: Any) -> Optional[int]:
    """JSON 값을 int로 바꿀 수 있으면 바꾼다."""
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def select_segment_text(segment: Dict[str, Any]) -> str:
    """수정된 텍스트가 있으면 우선 사용하고, 없으면 원본 텍스트를 사용한다."""
    text_edited = segment.get("textEdited")
    if isinstance(text_edited, str) and text_edited.strip():
        return text_edited.strip()

    text = segment.get("text")
    if isinstance(text, str):
        return text.strip()

    return ""


def parse_clova_response(path: Path) -> List[Dict[str, Any]]:
    """CLOVA Speech 응답 JSON 파일을 발화 행 목록으로 파싱한다."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments")

    if not isinstance(segments, list):
        raise ValueError("CLOVA 응답 JSON에 segments 배열이 없습니다.")

    rows: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{index}]가 object가 아닙니다.")

        start_ms = to_int(segment.get("start"))
        if start_ms is None:
            raise ValueError(f"segments[{index}]에 유효한 start 값이 없습니다.")

        speaker = segment.get("speaker")
        if not isinstance(speaker, dict):
            speaker = {}

        diarization = segment.get("diarization")
        if not isinstance(diarization, dict):
            diarization = {}

        speaker_label = str(speaker.get("label") or diarization.get("label") or "unknown")
        speaker_name = str(speaker.get("name") or f"참석자 {speaker_label}")
        end_ms = to_int(segment.get("end"))

        rows.append(
            {
                "index": index,
                "speaker_label": speaker_label,
                "speaker_name": speaker_name,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_time": format_ms(start_ms),
                "end_time": format_ms(end_ms),
                "text": select_segment_text(segment),
                "confidence": segment.get("confidence"),
            }
        )

    return rows


def print_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    """수동 확인을 위해 파싱 결과를 짧게 출력한다."""
    speakers = sorted({str(row["speaker_name"]) for row in rows})

    print(f"input: {path}")
    print(f"utterances: {len(rows)}")
    print(f"speakers: {len(speakers)}")
    print()

    for row in rows[:10]:
        text = str(row["text"]).replace("\n", " ")
        if len(text) > 90:
            text = text[:87] + "..."
        print(f"[{row['index']:03d}] {row['start_time']} {row['speaker_name']}: {text}")


def main() -> int:
    """입력된 CLOVA JSON 파일을 읽고 파싱된 발화를 출력한다."""
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    rows = parse_clova_response(input_path)
    print_summary(input_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
