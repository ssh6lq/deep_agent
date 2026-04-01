"""
LangChain Deep Agent — deepagents SDK 기반 v9

파이프라인:
  입력 (텍스트 / 파일)
    ↓
  DocumentLoader + RAG Pipeline
    ↓
  create_deep_agent [deepagents SDK]
    내장: Planning / File Tools / Sub Agents / Summarization
    커스텀: Web Search / Code Exec / Memory / File Tasks / MCP / RAG
    ↓
  Streaming Output → 최종 답변

모델: openai:gpt-4o (OpenAI)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
load_dotenv(_APP_DIR / ".env")

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

# ── 내부 모듈 ──────────────────────────────────────────────────────
from tools import custom_tools
from file_tools import file_tools
from subagents import build_subagents
from rag_pipeline import RAGPipeline


# ══════════════════════════════════════════════════════════════════
#  환경 검증
# ══════════════════════════════════════════════════════════════════

def require_api_keys() -> None:
    missing = []
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        missing.append("OPENAI_API_KEY")
    if missing:
        print(
            f"오류: 다음 API 키가 설정되지 않았습니다: {', '.join(missing)}\n"
            f"  • {_APP_DIR / '.env'} 파일에 키를 추가하세요.",
            file=sys.stderr,
        )
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  시스템 프롬프트
#  deepagents 기본 프롬프트(Planning, File, SubAgent)에 추가되는 부분
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """당신은 복잡한 멀티스텝 태스크를 수행하는 Deep Agent입니다.

## 파일 작업 도구 선택 기준

| 요청 유형            | 사용 도구           |
|----------------------|---------------------|
| 파일 요약            | summarize_file      |
| 파일 번역            | translate_file      |
| 핵심 포인트 추출     | extract_key_points  |
| 커스텀 분석·검토     | analyze_file        |
| 파일 내용 직접 열람  | load_file_content   |
| 첨부 파일 질의응답   | rag_search          |
| 결과를 파일로 저장   | save_report         |

## 보고서 저장 규칙
- "저장", "파일로 만들어줘", "보고서 형태로" 등의 요청이 있으면 반드시 save_report 사용
- 저장 후 파일 경로를 사용자에게 알려줄 것
- 기본 저장 형식: md (마크다운), 요청 시 txt 사용 가능
- 작업 흐름 예시: summarize_file → 결과 → save_report

## 메모리 관리 도구

| 작업              | 사용 도구       |
|-------------------|-----------------|
| 정보 저장         | memory_save     |
| 정보 조회         | memory_recall   |
| 전체 목록 확인    | memory_list     |
| 키워드 검색       | memory_search   |
| 특정 항목 삭제    | memory_delete   |
| 전체 초기화       | memory_clear    |

## 메모리 활용 원칙
- 사용자가 이름·날짜·설정 등 중요 정보를 말하면 memory_save로 저장
- 이전에 저장된 정보가 필요할 때 memory_recall / memory_search 먼저 확인
- 메모리는 세션 종료 후에도 유지됨 (JSON 파일 영구 보존)

## MCP Registry 활용
내부 문서 검색 시:
  list_collections(query) → 관련 컬렉션 탐색
  search_collection(id, query) → 청크 검색
  adaptive_search(mode='sequential'|'parallel') → 연쇄/병렬 분석

## Sub Agent 위임
메인 에이전트가 task() 도구로 서브에이전트에 위임:
  research-agent  → 웹 조사
  document-agent  → 내부 문서 분석 (MCP 활용)
  code-agent      → 데이터 처리 / 코드 실행

## 금지 사항
- 확인되지 않은 정보를 사실로 단정하지 말 것
- 파일 경로가 제공되면 반드시 해당 파일 도구를 사용할 것
- 보고서 저장 요청 시 save_report 없이 텍스트만 출력하지 말 것
"""


# ══════════════════════════════════════════════════════════════════
#  MCP 클라이언트
# ══════════════════════════════════════════════════════════════════

async def get_mcp_tools() -> list:
    try:
        client = MultiServerMCPClient(
            {
                "mcp_registry": {
                    "command": "python",
                    "args": [str(_APP_DIR / "mcp_registry_server.py")],
                    "transport": "stdio",
                },
            }
        )
        tools = await client.get_tools()
        print(f"[MCP] {len(tools)}개 도구 로드: {[t.name for t in tools]}")
        return tools
    except Exception as exc:
        print(f"[MCP] 연결 실패 (MCP 없이 계속): {exc}", file=sys.stderr)
        return []


# ══════════════════════════════════════════════════════════════════
#  RAG Pipeline 초기화 (파일 첨부 시)
# ══════════════════════════════════════════════════════════════════

def build_rag_tools(file_paths: list[str]) -> list:
    if not file_paths:
        return []
    pipeline = RAGPipeline(k=4)
    count = pipeline.add_files(file_paths)
    if count == 0:
        return []
    print(f"[RAG] {len(file_paths)}개 파일, {count}개 청크 인덱싱 완료")
    return [pipeline.as_retriever_tool()]


# ══════════════════════════════════════════════════════════════════
#  스트리밍 출력 핸들러
# ══════════════════════════════════════════════════════════════════

def _box(label: str, text: str, width: int = 80) -> None:
    """텍스트를 박스 형태로 출력한다."""
    border = "─" * width
    print(f"\n┌{border}┐", flush=True)
    print(f"│ {label:<{width - 1}}│", flush=True)
    print(f"├{border}┤", flush=True)
    for line in (text.splitlines() or ["-"]):
        for i in range(0, max(len(line), 1), width - 3):
            seg = line[i : i + width - 3]
            print(f"│ {seg:<{width - 2}}│", flush=True)
    print(f"└{border}┘", flush=True)


def _fmt(data, limit: int = 800) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, indent=2)[:limit]


async def stream_agent(agent, query: str, verbose: int = 0) -> None:
    """
    deepagents CompiledStateGraph를 스트리밍으로 실행하고 출력한다.

    verbose 레벨:
      0 — 기본 (도구 입출력만)
      1 — 미들웨어 입출력 + tool_calls 결정 내역
      2 — 모든 LangGraph 체인 이벤트 + LLM 프롬프트 메시지 전체
    """
    state = {"messages": [{"role": "user", "content": query}]}

    print(f"\n{'━' * 64}")
    print(f"  Deep Agent  질의: {query}")
    print(f"{'━' * 64}\n")

    tool_step = 0

    async for event in agent.astream_events(state, version="v2"):
        kind = event["event"]
        name = event.get("name", "-")

        # ── verbose=2: 모든 체인 이벤트 ─────────────────────────────
        if verbose >= 2:
            if kind == "on_chain_start":
                data = event["data"].get("input")
                _box(f"⬇  chain_start  {name}", _fmt(data))
            elif kind == "on_chain_end":
                data = event["data"].get("output")
                _box(f"⬆  chain_end    {name}", _fmt(data))
            elif kind == "on_retriever_start":
                q = event["data"].get("input", {}).get("query", "")
                print(f"\n  [RAG 검색 시작] query={q!r}", flush=True)
            elif kind == "on_retriever_end":
                docs = event["data"].get("output", {}).get("documents", [])
                print(f"  [RAG 검색 완료] {len(docs)}개 청크 반환", flush=True)

        # ── verbose=1: 미들웨어만 ────────────────────────────────────
        elif verbose >= 1 and "Middleware" in name:
            if kind == "on_chain_start":
                _box(f"▶ {name}  [input]", _fmt(event["data"].get("input")))
            elif kind == "on_chain_end":
                _box(f"◀ {name}  [output]", _fmt(event["data"].get("output")))

        # ── LLM 프롬프트 (verbose=2) ─────────────────────────────────
        if verbose >= 2 and kind == "on_chat_model_start":
            messages = event["data"].get("input", {}).get("messages", [])
            for turn in messages:
                role = getattr(turn, "type", "?") if hasattr(turn, "type") else "?"
                content = str(getattr(turn, "content", turn))[:400]
                print(f"\n  [LLM 입력 | {role}] {content}", flush=True)

        # ── LLM 스트리밍 출력 ────────────────────────────────────────
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                content = chunk.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block["text"], end="", flush=True)
                elif isinstance(content, str):
                    print(content, end="", flush=True)

        # ── tool_calls 결정 (verbose>=1) ─────────────────────────────
        elif kind == "on_chat_model_end" and verbose >= 1:
            msg = event["data"].get("output")
            if msg and hasattr(msg, "tool_calls") and msg.tool_calls:
                _box("  LLM → tool_calls 결정", _fmt(msg.tool_calls))

        # ── 도구 실행 ────────────────────────────────────────────────
        elif kind == "on_tool_start":
            tool_step += 1
            args = json.dumps(event["data"].get("input", {}), ensure_ascii=False, indent=2)
            limit = 600 if verbose >= 1 else 240
            print(f"\n{'·' * 64}", flush=True)
            print(f"  Step {tool_step}  {name}", flush=True)
            print(f"{'·' * 64}", flush=True)
            print(f"  입력: {args[:limit]}", flush=True)

        elif kind == "on_tool_end":
            limit = 600 if verbose >= 1 else 200
            output = str(event["data"].get("output", ""))[:limit].replace("\n", " ")
            print(f"  결과: {output}", flush=True)
            print(f"{'·' * 64}", flush=True)

    print(f"\n{'━' * 64}")
    print(f"  Deep Agent  완료  (도구 {tool_step}회 실행)")
    print(f"{'━' * 64}\n")


# ══════════════════════════════════════════════════════════════════
#  에이전트 빌더
# ══════════════════════════════════════════════════════════════════

async def build_agent(file_paths: list[str] | None = None):
    """도구·서브에이전트·모델을 조합해 deepagents 에이전트를 생성한다."""
    mcp_tools = await get_mcp_tools()
    rag_tools = build_rag_tools(file_paths or [])
    subagents_list = build_subagents(mcp_tools)

    all_tools = [
        *custom_tools,   # 웹 검색, 코드 실행, 메모리
        *file_tools,     # 파일 요약·번역·분석
        *mcp_tools,      # MCP Registry
        *rag_tools,      # RAG (첨부 파일 있을 때)
    ]

    print(f"[Deep Agent] 도구 {len(all_tools)}개, 서브에이전트 {len(subagents_list)}개 로드")

    # model = init_chat_model(
    #     model="openai:/models/Qwen3.5-35B-A3B-FP8",
    #     base_url="http://192.168.1.51:8001/v1",  # vLLM 서버 주소
    #     api_key="EMPTY",                          # vLLM은 키 불필요, 빈 값으로
    #     max_retries=10,
    #     timeout=120,
    #     temperature=1.0,
    #     top_p=0.95,
    #     max_tokens=8192,
    #     # model_kwargs={
    #     #     "chat_template_kwargs": {"enable_thinking": False},
    #     # },
    # )


    model = init_chat_model(
        model="openai:gpt-5.3-chat-latest",
        max_retries=10,   # 불안정한 네트워크 대비
        timeout=120,      # 느린 연결 대비
    )

    backend = FilesystemBackend(root_dir=_APP_DIR)

    return create_deep_agent(
        model=model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        subagents=subagents_list,
        backend=backend,
        skills=["/skills/"],
    )


# ══════════════════════════════════════════════════════════════════
#  CLI 인수 파싱
# ══════════════════════════════════════════════════════════════════

def parse_args(argv: list[str]) -> tuple[str, list[str]]:
    """
    CLI 인수를 파싱해 (query, file_paths)를 반환한다.

    사용법:
      # 자연어 질의
      python main.py "이 파일을 요약해줘" report.pdf

      # 파일 작업 단축 모드 (파일 도구 직접 지정)
      python main.py summarize  report.pdf [--lang English]
      python main.py translate  contract.pdf --lang English
      python main.py keypoints  paper.pdf
      python main.py analyze    contract.pdf "불리한 조항 찾기"
      python main.py 요약 / 번역 / 핵심 / 분석  (한국어 단축어도 지원)
    """
    FILE_TASK_ALIASES: dict[str, str] = {
        "summarize": "summarize_file",  "요약": "summarize_file",
        "translate": "translate_file",  "번역": "translate_file",
        "keypoints": "extract_key_points", "핵심": "extract_key_points",
        "analyze":   "analyze_file",    "분석": "analyze_file",
        "load":      "load_file_content", "열람": "load_file_content",
    }

    # --lang / -l 옵션 추출
    lang = "Korean"
    filtered: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--lang", "-l") and i + 1 < len(argv):
            lang = argv[i + 1]
            i += 2
        else:
            filtered.append(argv[i])
            i += 1

    # 단축 모드: <task> <file> [extra_instruction]
    if (
        filtered
        and filtered[0].lower() in FILE_TASK_ALIASES
        and len(filtered) >= 2
        and Path(filtered[1]).exists()
    ):
        task_key = filtered[0].lower()
        file_path = filtered[1]
        extra = filtered[2] if len(filtered) > 2 else ""
        tool_name = FILE_TASK_ALIASES[task_key]

        task_descriptions = {
            "summarize_file":     f'"{file_path}" 파일을 {lang}로 요약하라. summarize_file 도구 사용.',
            "translate_file":     f'"{file_path}" 파일을 {lang}로 번역하라. translate_file 도구 사용.',
            "extract_key_points": f'"{file_path}" 파일의 핵심 포인트를 {lang}로 추출하라. extract_key_points 도구 사용.',
            "analyze_file":       f'"{file_path}" 파일에 대해 다음 작업을 수행하라: {extra or "전체 분석"}. analyze_file 도구 사용.',
            "load_file_content":  f'"{file_path}" 파일의 내용을 열람하라. load_file_content 도구 사용.',
        }
        return task_descriptions[tool_name], [file_path]

    # 일반 모드
    if not filtered:
        default_query = "hr 규정을 기준으로 영업팀 계약서에 문제가 있는지 검토해줘."
        return default_query, []

    query = filtered[0]
    file_paths = [a for a in filtered[1:] if Path(a).exists()]
    return query, file_paths


# ══════════════════════════════════════════════════════════════════
#  메인 실행
# ══════════════════════════════════════════════════════════════════

async def main() -> None:
    require_api_keys()

    # query, file_paths = parse_args(sys.argv[1:])
    query, file_paths = "먼저 파일별로 요약을 해주고, 통합적으로 비교 및 분석을 해줘. 그리고 이 내용을 바탕으로 보고서 형태의 docx 파일로 저장해줘.", ["/Users/sso/Downloads/workspace/files/aidoc_인수인계문서.docx", "/Users/sso/Downloads/workspace/수강신청_장바구니_메뉴얼.pdf"]
    agent = await build_agent(file_paths)
    # 파일 경로가 있으면 쿼리에 포함시켜 에이전트가 파일을 인식하도록 함
    effective_query = query
    if file_paths:
        paths_str = "\n".join(f"  - {p}" for p in file_paths)
        effective_query = f"{query}\n\n[첨부 파일]\n{paths_str}"

    await stream_agent(agent, effective_query, verbose=2)


# ── invoke 방식 (비스트리밍 / 프로그래매틱 호출용) ─────────────────

async def invoke_agent(query: str, file_paths: list[str] | None = None) -> str:
    """에이전트를 비스트리밍으로 실행하고 최종 답변 문자열을 반환한다."""
    require_api_keys()

    agent = await build_agent(file_paths)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    asyncio.run(main())
