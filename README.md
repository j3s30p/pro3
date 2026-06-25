# Discord Message Crawler

Discord 서버의 텍스트 채널 메시지를 채널별 파일로 수집하는 1차 검증용 도구입니다.

현재 구현은 1차 검증용으로, Discord 메시지 수집과 로컬 RAG 질의까지 제공합니다.
Discord 챗봇 실행 코드는 아직 포함하지 않습니다.

## Setup

```bash
uv sync
cp .env.example .env
```

`.env`에 Discord bot token과 대상 서버 ID를 입력합니다.

```text
DISCORD_TOKEN=...
DISCORD_GUILD_ID=...
DISCORD_CHANNEL_IDS=
CRAWL_LIMIT=0
DOWNLOAD_ATTACHMENTS=true
```

- `DISCORD_CHANNEL_IDS`가 비어 있으면 봇이 볼 수 있는 모든 텍스트 채널을 수집합니다.
- `CRAWL_LIMIT=0`이면 가능한 전체 히스토리를 수집합니다.
- `DOWNLOAD_ATTACHMENTS=true`이면 첨부파일을 로컬에 저장합니다.

## Run

```bash
uv run crawl-discord-messages
```

특정 채널만 수집:

```bash
uv run crawl-discord-messages --channel-id 123 --channel-id 456
```

Markdown 검수 파일 없이 JSONL만 수집:

```bash
uv run crawl-discord-messages --no-markdown
```

첨부파일은 metadata만 남기고 다운로드하지 않기:

```bash
uv run crawl-discord-messages --no-download-attachments
```

## Output

```text
data/raw/{guild_id}/{category_position}_{category_id}_{category_name}/{channel_position}_{channel_id}_{channel_name}.jsonl
data/raw/{guild_id}/manifest.json
data/markdown/{guild_id}/{category_position}_{category_id}_{category_name}/{channel_position}_{channel_id}_{channel_name}.md
data/attachments/{guild_id}/{category_position}_{category_id}_{category_name}/{channel_position}_{channel_id}_{channel_name}/
```

카테고리가 없는 채널은 `000_no_category` 폴더에 저장됩니다.

RAG 인덱싱 원본은 `JSONL`입니다. `MD` 파일은 사람이 대화 흐름을 확인하기 위한 검수용입니다.
첨부파일은 `data/attachments`에 저장되고, JSONL에는 로컬 경로와 다운로드 상태가 함께 기록됩니다.
임베드는 Discord raw embed와 검색용 텍스트가 JSONL에 함께 기록됩니다.

## RAG

`.env`에 OpenAI API key를 추가합니다.

```text
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
CHROMA_PATH=data/chroma
RAG_WINDOW_BEFORE=2
RAG_WINDOW_AFTER=2
RAG_TOP_K=6
RAG_MAX_CHUNK_CHARS=4000
RAG_MESSAGE_MAX_ATTACHMENT_CHARS=0
RAG_MAX_ATTACHMENT_TEXT_CHARS=50000
RAG_ATTACHMENT_CHUNK_TOKENS=600
RAG_ATTACHMENT_CHUNK_OVERLAP_TOKENS=80
RAG_MAX_ATTACHMENT_CHUNK_CHARS=4000
```

인덱스 생성:

```bash
uv run build-rag-index
```

질문:

```bash
uv run ask-discord-rag "저번에 제섭이가 뭐라고 했지?"
```

첨부파일을 찾는 질문:

```bash
uv run ask-discord-rag "첨부파일 중 CNN html 파일 찾아줘"
```

기본 RAG는 두 Chroma collection을 만듭니다.

```text
discord_message_windows
discord_attachment_chunks
```

`discord_message_windows`는 메시지 1개를 중심으로 앞뒤 메시지 2개씩 묶은 window chunk입니다.
첨부파일은 파일명, 타입, 경로, URL만 포함하고 본문은 직접 넣지 않습니다.

`discord_attachment_chunks`는 첨부파일 전용 chunk입니다.
텍스트형 첨부파일은 `kiwipiepy` 한국어 토크나이저 기준으로 나누어 색인하고, 이미지 같은 비텍스트 첨부파일도 파일명/타입/경로/연결 메시지 기준으로 색인합니다.

임베드 텍스트는 메시지 window chunk에 포함됩니다.
질문에 `첨부`, `파일`, `html`, `pdf`, `이미지`, `자료`, `코드` 같은 힌트가 있을 때만 첨부파일 collection을 함께 검색하고 `참고 첨부파일`을 출력합니다.
이미지, PDF 같은 파일의 OCR/파싱은 아직 포함하지 않습니다.

## Discord Permissions

Developer Portal에서 `Message Content Intent`를 켜야 합니다.

봇에는 최소 권한이 필요합니다.

```text
View Channels
Read Message History
```

수집 대상 비공개 채널이 있다면 해당 채널에서도 봇이 볼 수 있어야 합니다.
