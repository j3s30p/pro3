# Meeting RAG Assistant

Clova Note/CLOVA Speech 회의 데이터를 파싱하고, 회의록 생성과 RAG 검색으로 확장하기 위한 프로젝트입니다.

현재 구현 범위는 CLOVA Speech 응답형 JSON 파싱, 발화 청킹, LLM 회의록 생성입니다.

Clova Note txt export는 샘플 데이터 변환용으로만 사용하고, 기본 입력은 `segments[]`를 포함한 JSON입니다.

파싱 확인:

```bash
uv run python -m meeting_rag.parsing
```

청킹 확인:

```bash
uv run python -m meeting_rag.chunking
```

회의록 생성:

```bash
uv run python -m meeting_rag.minutes
```

Vector DB 인덱싱:

```bash
uv run python -m meeting_rag.vectordb
```

전체 파이프라인 실행:

```bash
uv run python -m meeting_rag.pipeline
```

RAG 질의응답:

```bash
uv run python -m meeting_rag.rag "STT는 어떻게 하기로 했어?"
```
