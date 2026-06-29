from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from .chunking import chunk_utterances
from .config import get_llm_model, get_openai_api_key
from .minutes import DEFAULT_MINUTES_PATH, build_minutes_prompt, call_openai_minutes
from .parsing import DEFAULT_INPUT_PATH, parse_clova_response
from .vectordb import DEFAULT_CHROMA_PATH, index_vectordb


class PipelineState(TypedDict, total=False):
    """LangGraph 노드 사이에서 공유하는 파이프라인 상태."""

    input_path: str
    minutes_path: str
    chroma_path: str
    rows: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    parse_result: Dict[str, Any]
    chunk_result: Dict[str, Any]
    minutes_result: Dict[str, Any]
    vectordb_result: Dict[str, Any]


def parse_node(state: PipelineState) -> PipelineState:
    """CLOVA JSON 파일을 발화 목록으로 파싱한다."""
    input_path = Path(state["input_path"])
    rows = parse_clova_response(input_path)
    speakers = sorted({str(row["speaker_name"]) for row in rows})

    return {
        "rows": rows,
        "parse_result": {
            "input_path": str(input_path),
            "utterances": len(rows),
            "speakers": len(speakers),
        },
    }


def chunk_node(state: PipelineState) -> PipelineState:
    """파싱된 발화 목록을 RAG용 청크로 묶는다."""
    chunks = chunk_utterances(state["rows"])

    return {
        "chunks": chunks,
        "chunk_result": {
            "chunks": len(chunks),
        },
    }


def minutes_node(state: PipelineState) -> PipelineState:
    """청크를 LLM에 전달해 회의록 Markdown 파일을 생성한다."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(".env에 OPENAI_API_KEY를 설정해야 회의록을 생성할 수 있습니다.")

    input_path = Path(state["input_path"])
    minutes_path = Path(state["minutes_path"])
    prompt = build_minutes_prompt(input_path, state["chunks"])
    model = get_llm_model()
    minutes = call_openai_minutes(prompt=prompt, model=model, api_key=api_key)

    minutes_path.parent.mkdir(parents=True, exist_ok=True)
    minutes_path.write_text(minutes + "\n", encoding="utf-8")

    return {
        "minutes_result": {
            "model": model,
            "output_path": str(minutes_path),
            "prompt_chars": len(prompt),
            "minutes_chars": len(minutes),
        },
    }


def vectordb_node(state: PipelineState) -> PipelineState:
    """STT 청크와 생성된 회의록을 Vector DB에 저장한다."""
    result = index_vectordb(
        input_path=Path(state["input_path"]),
        minutes_path=Path(state["minutes_path"]),
        chroma_path=Path(state["chroma_path"]),
    )

    return {"vectordb_result": result}


def build_pipeline_graph():
    """파싱부터 Vector DB 저장까지 이어지는 LangGraph 그래프를 만든다."""
    graph = StateGraph(PipelineState)
    graph.add_node("parse", RunnableLambda(parse_node))
    graph.add_node("chunk", RunnableLambda(chunk_node))
    graph.add_node("generate_minutes", RunnableLambda(minutes_node))
    graph.add_node("index_vectordb", RunnableLambda(vectordb_node))

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "chunk")
    graph.add_edge("chunk", "generate_minutes")
    graph.add_edge("generate_minutes", "index_vectordb")
    graph.add_edge("index_vectordb", END)

    return graph.compile()


def run_pipeline(
    input_path: Path = DEFAULT_INPUT_PATH,
    minutes_path: Path = DEFAULT_MINUTES_PATH,
    chroma_path: Path = DEFAULT_CHROMA_PATH,
) -> PipelineState:
    """입력 파일 하나를 처리해 회의록과 Vector DB를 갱신한다."""
    graph = build_pipeline_graph()
    return graph.invoke(
        {
            "input_path": str(input_path),
            "minutes_path": str(minutes_path),
            "chroma_path": str(chroma_path),
        }
    )


def print_result(result: PipelineState) -> None:
    """파이프라인 실행 결과를 CLI에서 확인하기 쉽게 출력한다."""
    parse_result = result["parse_result"]
    chunk_result = result["chunk_result"]
    minutes_result = result["minutes_result"]
    vectordb_result = result["vectordb_result"]

    print(f"input: {parse_result['input_path']}")
    print(f"utterances: {parse_result['utterances']}")
    print(f"speakers: {parse_result['speakers']}")
    print(f"chunks: {chunk_result['chunks']}")
    print(f"minutes_model: {minutes_result['model']}")
    print(f"minutes_output: {minutes_result['output_path']}")
    print(f"minutes_chars: {minutes_result['minutes_chars']}")
    print(f"embedding_model: {vectordb_result['embedding_model']}")
    print(f"chroma_path: {vectordb_result['chroma_path']}")
    print(f"stt_chunks: {vectordb_result['stt_chunks']}")
    print(f"minutes_sections: {vectordb_result['minutes_sections']}")


def main() -> int:
    """CLI 인자를 받아 전체 파이프라인을 실행한다."""
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    minutes_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MINUTES_PATH
    chroma_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CHROMA_PATH

    try:
        result = run_pipeline(input_path, minutes_path, chroma_path)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}")
        return 1

    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
