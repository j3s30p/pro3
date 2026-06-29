from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from .chunking import chunk_utterances
from .config import get_llm_model, get_openai_api_key
from .parsing import DEFAULT_INPUT_PATH, parse_clova_response


DEFAULT_MINUTES_PATH = Path("outputs/minutes.md")
RESPONSES_API_URL = "https://api.openai.com/v1/responses"


def build_chunk_context(chunks: List[Dict[str, Any]]) -> str:
    """청크 목록을 LLM에 전달할 회의 원문 컨텍스트로 바꾼다."""
    blocks: List[str] = []

    for chunk in chunks:
        speakers = ", ".join(chunk["speakers"])
        blocks.append(
            "\n".join(
                [
                    f"### 청크 {chunk['chunk_index']:03d}",
                    f"- 시간: {chunk['start_time']} ~ {chunk['end_time']}",
                    f"- 발화: {chunk['utterance_start_index']} ~ {chunk['utterance_end_index']}",
                    f"- 화자: {speakers}",
                    "",
                    chunk["text"],
                ]
            )
        )

    return "\n\n".join(blocks)


def build_minutes_prompt(source_path: Path, chunks: List[Dict[str, Any]]) -> str:
    """회의록 생성을 위한 지시문과 회의 원문 컨텍스트를 만든다."""
    context = build_chunk_context(chunks)

    return f"""# 회의록 생성 프롬프트

너는 회의록 작성 도우미다. 아래 CLOVA STT 회의 원문만 근거로 회의록을 작성한다.

## 작성 규칙
- 원문에 없는 내용은 만들지 않는다.
- 확실하지 않은 담당자, 일정, 결정사항은 "확인 필요" 또는 "미정"으로 적는다.
- 결정사항은 원문에서 명확히 "정했다", "확정했다", "하기로 했다"는 흐름이 확인될 때만 작성한다.
- 단순 아이디어, 후보, 검토 의견, 가능성 논의는 결정사항에 넣지 말고 보류/이슈 또는 다음 확인 사항에 넣는다.
- 액션 아이템의 담당자는 원문에서 누가 하기로 했는지 명확할 때만 적고, 불명확하면 "확인 필요"로 적는다.
- 발언자를 담당자로 추정하지 않는다.
- 결정사항과 액션 아이템에는 가능한 한 근거 시간을 함께 적는다.
- 잡담은 제외하고 프로젝트 진행에 필요한 내용 중심으로 정리한다.
- 출력은 반드시 아래 "회의록 출력 양식"을 따른다.

## 회의록 출력 양식

# 회의록

## 회의 개요
- 회의명:
- 일시:
- 참석자:

## 핵심 요약

## 주요 안건

## 결정사항
| 결정사항 | 근거 시간 | 관련 화자 |
|---|---|---|

## 액션 아이템
| 담당자 | 할 일 | 마감일 | 상태 | 근거 시간 |
|---|---|---|---|---|

## 보류/이슈

## 다음 회의에서 확인할 사항

## 원문 근거

---

## 입력 파일
{source_path}

## 회의 원문

{context}
"""


def call_openai_minutes(prompt: str, model: str, api_key: str) -> str:
    """OpenAI Responses API를 호출해 회의록 Markdown을 생성한다."""
    request_body = json.dumps(
        {
            "model": model,
            "input": prompt,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        RESPONSES_API_URL,
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
        raise RuntimeError(f"OpenAI API 호출 실패: HTTP {error.code}\n{body}") from error

    return extract_response_text(payload)


def extract_response_text(payload: Dict[str, Any]) -> str:
    """Responses API 응답에서 텍스트 출력만 꺼낸다."""
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("OpenAI 응답에 output 배열이 없습니다.")

    texts = []
    for item in output:
        if not isinstance(item, dict):
            continue

        content = item.get("content")
        if not isinstance(content, list):
            continue

        for content_item in content:
            if not isinstance(content_item, dict):
                continue

            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

    if not texts:
        raise ValueError("OpenAI 응답에서 텍스트를 찾지 못했습니다.")

    return "\n\n".join(texts)


def generate_minutes(input_path: Path, minutes_path: Path) -> Dict[str, Any]:
    """CLOVA JSON에서 프롬프트를 만들고 LLM 응답 회의록을 저장한다."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정해야 회의록을 생성할 수 있습니다.")

    rows = parse_clova_response(input_path)
    chunks = chunk_utterances(rows)
    prompt = build_minutes_prompt(input_path, chunks)
    model = get_llm_model()
    minutes = call_openai_minutes(prompt=prompt, model=model, api_key=api_key)

    minutes_path.parent.mkdir(parents=True, exist_ok=True)
    minutes_path.write_text(minutes + "\n", encoding="utf-8")

    return {
        "model": model,
        "utterance_count": len(rows),
        "chunk_count": len(chunks),
        "prompt_chars": len(prompt),
        "minutes_chars": len(minutes),
    }


def main() -> int:
    """입력 CLOVA JSON을 LLM에 보내 회의록 결과 파일을 만든다."""
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    minutes_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MINUTES_PATH

    try:
        result = generate_minutes(input_path, minutes_path)
    except RuntimeError as error:
        print(f"error: {error}")
        return 1

    print(f"model: {result['model']}")
    print(f"input: {input_path}")
    print(f"output: {minutes_path}")
    print(f"utterances: {result['utterance_count']}")
    print(f"chunks: {result['chunk_count']}")
    print(f"prompt_chars: {result['prompt_chars']}")
    print(f"minutes_chars: {result['minutes_chars']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
