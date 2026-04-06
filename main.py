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
from langchain.agents.middleware import PIIMiddleware
import asyncio
import json
import os
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_APP_DIR = Path(__file__).resolve().parent
load_dotenv(_APP_DIR / ".env")

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import AsyncPostgresStore

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from deepagents.backends.filesystem import FilesystemBackend


# ── 내부 모듈 ──────────────────────────────────────────────────────
from tools import custom_tools, _current_user_id
from file_tools import file_tools
from subagents import build_subagents
from rag_pipeline import RAGPipeline
from document_loader import load_document, image_to_vision_content


# ══════════════════════════════════════════════════════════════════
#  PII Middleware
# ══════════════════════════════════════════════════════════════════

class LoggingPIIMiddleware(PIIMiddleware):
    """PIIMiddleware를 래핑해 마스킹된 내용을 콘솔에 출력한다."""

    def _log_masked(self, orig_messages: list, new_messages: list) -> None:
        for i, (orig, new) in enumerate(zip(orig_messages, new_messages)):
            orig_content = str(getattr(orig, "content", ""))
            new_content = str(getattr(new, "content", ""))
            if orig_content != new_content:
                msg_type = type(orig).__name__
                print("=" * 60)
                print(f"  [PIIMiddleware] pii_type={self.pii_type!r}  [{msg_type}] msg[{i}] 마스킹 적용")
                print("-" * 60)
                print(f"  [BEFORE] {orig_content[:300]}")
                print(f"  [AFTER ] {new_content[:300]}")
                print("=" * 60)

    def before_model(self, state, runtime):
        messages = state.get("messages", [])
        last_human = next(
            (m for m in reversed(messages) if hasattr(m, "type") and m.type == "human"),
            None,
        )
        print(f"  [PIIMiddleware][{self.pii_type}] before_model 호출됨 | 마지막 human 메시지: {str(getattr(last_human, 'content', '(없음)'))[:80]!r}")
        result = super().before_model(state, runtime)
        print(f"  [PIIMiddleware][{self.pii_type}] 감지 결과: {'마스킹 적용' if result else 'PII 없음'}")
        if result and "messages" in result:
            self._log_masked(state["messages"], result["messages"])
        return result

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)


def _detect_banned_words(content: str) -> list[dict]:
    banned = ["비밀번호", "주민번호", "계좌번호"]
    matches = []
    for word in banned:
        for m in re.finditer(re.escape(word), content):
            matches.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    return matches


# _PII_RULES: list[dict] = [
#     {
#         "name": "api_key",
#         "detector": r"sk-[a-zA-Z0-9]{32,}",
#         "strategy": "block",
#     },
#     {
#         "name": "banned_word",
#         "detector": _detect_banned_words,
#         "strategy": "redact",
#     },
# ]


# class PIIMiddleware(AgentMiddleware):
#     """LLM 호출 전 메시지에서 PII/민감 정보를 탐지해 block 또는 redact한다."""

#     def _apply_rules(self, text: str) -> tuple[str, list[str], list[dict]]:
#         """규칙을 적용해 (처리된 텍스트, block된 규칙명 목록, redact 로그 목록) 반환."""
#         blocked: list[str] = []
#         redact_log: list[dict] = []
#         for rule in _PII_RULES:
#             detector = rule["detector"]
#             strategy = rule["strategy"]
#             name = rule["name"]

#             if isinstance(detector, str):
#                 matches = [
#                     {"text": m.group(0), "start": m.start(), "end": m.end()}
#                     for m in re.finditer(detector, text)
#                 ]
#             else:
#                 matches = detector(text)

#             if not matches:
#                 continue

#             if strategy == "block":
#                 blocked.append(name)
#                 for m in matches:
#                     # API 키 등 민감값은 앞 4자만 노출
#                     preview = m["text"][:4] + "..." if len(m["text"]) > 4 else m["text"]
#                     print(f"  [PIIMiddleware] block 규칙={name!r}  감지={preview!r}  위치={m['start']}-{m['end']}")
#             elif strategy == "redact":
#                 for m in sorted(matches, key=lambda x: x["start"], reverse=True):
#                     preview = m["text"][:4] + "..." if len(m["text"]) > 4 else m["text"]
#                     redact_log.append({"rule": name, "preview": preview})
#                     text = text[: m["start"]] + "[REDACTED]" + text[m["end"]:]

#         return text, blocked, redact_log

    # def _process_content(self, content):
    #     """
    #     content가 문자열이면 그대로 규칙 적용.
    #     content가 리스트(multimodal)이면 text block만 규칙 적용하고
    #     image_url 등 나머지 block은 그대로 유지.
    #     반환: (새 content, blocked 목록)
    #     """
    #     if isinstance(content, str):
    #         cleaned, blocked, redact_log = self._apply_rules(content)
    #         return cleaned, blocked, redact_log

    #     if isinstance(content, list):
    #         all_blocked: list[str] = []
    #         all_redact_log: list[dict] = []
    #         new_blocks = []
    #         for block in content:
    #             if isinstance(block, dict) and block.get("type") == "text":
    #                 cleaned, blocked, redact_log = self._apply_rules(block["text"])
    #                 all_blocked.extend(blocked)
    #                 all_redact_log.extend(redact_log)
    #                 new_blocks.append({**block, "text": cleaned})
    #             else:
    #                 new_blocks.append(block)
    #         return new_blocks, all_blocked, all_redact_log

    #     return content, [], []

    # def _process_request(self, request: ModelRequest) -> ModelRequest:
    #     processed_messages = []
    #     for msg in request.messages:
    #         content = getattr(msg, "content", "")
    #         new_content, blocked, redact_log = self._process_content(content)

    #         if blocked:
    #             raise ValueError(
    #                 f"[PIIMiddleware] 민감 정보 감지로 요청 차단: {', '.join(blocked)}"
    #             )

    #         if redact_log:
    #             for entry in redact_log:
    #                 print(f"  [PIIMiddleware] redact 규칙={entry['rule']!r}  감지={entry['preview']!r}  → [REDACTED]")
    #         if new_content != content:
    #             msg = msg.model_copy(update={"content": new_content})

    #         processed_messages.append(msg)

    #     return ModelRequest(
    #         model=request.model,
    #         messages=processed_messages,
    #         system_message=request.system_message,
    #         tool_choice=request.tool_choice,
    #         tools=request.tools,
    #         response_format=request.response_format,
    #         state=request.state,
    #         runtime=request.runtime,
    #         model_settings=request.model_settings,
    #     )

    # async def awrap_model_call(self, request: ModelRequest, handler):
    #     return await handler(self._process_request(request))

    # def wrap_model_call(self, request: ModelRequest, handler):
    #     return handler(self._process_request(request))


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


# ══════════════════════════════════════════════════════════1════════
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

## 외부 서비스 MCP 활용

| 서비스 | 가능한 작업 | 위임할 서브에이전트 |
|--------|------------|-------------------|
| Notion | 페이지 검색·생성·수정, DB 쿼리 | notion-agent |
| Google Calendar | 일정 조회·생성·수정·삭제 | calendar-agent |
| Gmail | 메일 검색·읽기·전송·초안 작성 | gmail-agent |
| Jira | 이슈 조회·생성·수정, 댓글 추가, JQL 검색 | jira-agent |

## Sub Agent 위임
메인 에이전트가 task() 도구로 서브에이전트에 위임:
  research-agent  → 웹 조사
  document-agent  → 내부 문서 분석 (MCP Registry)
  code-agent      → 데이터 처리 / 코드 실행
  notion-agent    → Notion 페이지·DB 작업
  calendar-agent  → Google Calendar 일정 관리
  gmail-agent     → Gmail 메일 처리
  jira-agent      → Jira 이슈 관리

## 세션 오프로드

| 상황                          | 사용 도구          |
|-------------------------------|--------------------|
| 긴 중간 결과를 컨텍스트 밖으로 저장 | save_turn_result |

- thread_id: 현재 세션 식별자 (없으면 "default" 사용)
- turn_num: 현재 턴 번호 (1부터 순서대로)
- 저장 후 파일 경로를 메모해 두고 이후 턴에서 참조 가능

## 금지 사항
- 확인되지 않은 정보를 사실로 단정하지 말 것
- 파일 경로가 제공되면 반드시 해당 파일 도구를 사용할 것
- 보고서 저장 요청 시 save_report 없이 텍스트만 출력하지 말 것
"""


# ══════════════════════════════════════════════════════════════════
#  MCP 클라이언트
# ══════════════════════════════════════════════════════════════════

async def _connect_mcp_server(name: str, config: dict, tool_names: set[str]) -> list:
    """단일 MCP 서버에 연결하고 도구 목록을 반환한다. 실패 시 빈 리스트."""
    try:
        client = MultiServerMCPClient({name: config})
        all_tools = await client.get_tools()
        tools = [t for t in all_tools if t.name in tool_names] if tool_names else list(all_tools)
        print(f"[MCP] {name}: {len(tools)}개 도구 로드 {[t.name for t in tools]}")
        return tools
    except Exception as exc:
        print(f"[MCP] {name} 연결 실패: {exc}", file=sys.stderr)
        return []


async def get_mcp_tools() -> dict[str, list]:
    """MCP 서버별 도구 목록을 dict로 반환한다.

    서버마다 독립적으로 연결하므로 한 서버 실패가 다른 서버에 영향 없음.
    필수 환경변수가 없는 서버는 자동으로 스킵한다.

    Returns:
        {"registry": [...], "notion": [...], "google_calendar": [...], "gmail": [...], "jira": [...]}
    """
    SERVERS: dict[str, dict] = {
        "registry": {
            "config": {
                "command": "python",
                "args": [str(_APP_DIR / "mcp_registry_server.py")],
                "transport": "stdio",
            },
            "tools": {"list_collections", "search_collection", "adaptive_search"},
            "required_env": [],
        },
        "notion": {
            "config": {
                "command": "npx",
                "args": ["-y", "@notionhq/notion-mcp-server"],
                "transport": "stdio",
                "env": {"NOTION_API_KEY": os.environ.get("NOTION_API_KEY", "")},
            },
            "tools": set(),  # 빈 set = 서버의 전체 도구 수집
            "required_env": ["NOTION_API_KEY"],
        },
        "google_calendar": {
            "config": {
                "command": "npx",
                "args": ["-y", "@gongrzhe/server-calendar-mcp"],
                "transport": "stdio",
                "env": {
                    "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID", ""),
                    "GOOGLE_CLIENT_SECRET": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
                    "GOOGLE_REFRESH_TOKEN": os.environ.get("GOOGLE_REFRESH_TOKEN", ""),
                },
            },
            "tools": set(),
            "required_env": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
        },
        "gmail": {
            "config": {
                "command": "npx",
                "args": ["-y", "@gongrzhe/server-gmail-autoauth-mcp"],
                "transport": "stdio",
                "env": {
                    "GOOGLE_CLIENT_ID": os.environ.get("GOOGLE_CLIENT_ID", ""),
                    "GOOGLE_CLIENT_SECRET": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
                    "GOOGLE_REFRESH_TOKEN": os.environ.get("GOOGLE_REFRESH_TOKEN", ""),
                },
            },
            "tools": set(),
            "required_env": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
        },
        "jira": {
            "config": {
                "command": "uvx",
                "args": ["mcp-atlassian"],
                "transport": "stdio",
                "env": {
                    "JIRA_SERVER_URL": os.environ.get("JIRA_SERVER_URL", ""),
                    "JIRA_USERNAME": os.environ.get("JIRA_USERNAME", ""),
                    "JIRA_API_TOKEN": os.environ.get("JIRA_API_TOKEN", ""),
                },
            },
            "tools": {"jira_get_issue", "jira_search_issues", "jira_create_issue",
                      "jira_update_issue", "jira_add_comment"},
            "required_env": ["JIRA_SERVER_URL", "JIRA_USERNAME", "JIRA_API_TOKEN"],
        },
    }

    result: dict[str, list] = {}
    pending_names: list[str] = []
    pending_tasks = []

    for name, cfg in SERVERS.items():
        missing = [k for k in cfg["required_env"] if not os.environ.get(k, "").strip()]
        if missing:
            print(f"[MCP] {name}: 환경변수 미설정 ({', '.join(missing)}), 스킵")
            result[name] = []
            continue
        pending_names.append(name)
        pending_tasks.append(
            _connect_mcp_server(f"mcp_{name}", cfg["config"], cfg["tools"])
        )

    gathered = await asyncio.gather(*pending_tasks)
    for name, tools in zip(pending_names, gathered):
        result[name] = tools

    return result


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


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def build_user_message(query: str, current_files: list[str]) -> dict:
    """
    텍스트 쿼리와 현재 턴 파일 목록으로 user 메시지를 구성한다.
    이미지 파일은 Vision content block으로 변환하고,
    그 외 파일은 기존처럼 [첨부 파일] 텍스트로 명시한다.
    """
    image_blocks: list[dict] = []
    text_files: list[str] = []

    for fp in current_files:
        if Path(fp).suffix.lower() in _IMAGE_EXTS:
            result = load_document(fp)
            if result.documents and result.documents[0].image_data:
                doc = result.documents[0]
                blocks = image_to_vision_content(doc)
                # 기본 분석 텍스트 제거 — 사용자 쿼리로 대체
                image_blocks.extend(b for b in blocks if b.get("type") == "image_url")
                print(f"[Vision] 이미지 로드: {Path(fp).name} ({doc.image_mime})")
            else:
                text_files.append(fp)
        else:
            text_files.append(fp)

    if not image_blocks:
        # 이미지 없음 — 기존 텍스트 방식
        if text_files:
            paths_str = "\n".join(f"  - {p}" for p in text_files)
            return {"role": "user", "content": f"{query}\n\n[첨부 파일]\n{paths_str}"}
        return {"role": "user", "content": query}

    # 이미지 있음 — multimodal content block
    content: list[dict] = [{"type": "text", "text": query}]
    if text_files:
        paths_str = "\n".join(f"  - {p}" for p in text_files)
        content[0]["text"] += f"\n\n[첨부 파일]\n{paths_str}"
    content.extend(image_blocks)
    return {"role": "user", "content": content}


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
    """데이터를 읽기 쉬운 형태로 포맷한다."""
    if data is None:
        return "-"
    # LangChain 메시지 리스트 처리
    if isinstance(data, list):
        lines = []
        for item in data:
            if hasattr(item, "type") and hasattr(item, "content"):
                role = getattr(item, "type", "?")
                content = str(getattr(item, "content", ""))
                # tool_calls 요약
                tool_calls = getattr(item, "tool_calls", None) or getattr(item, "additional_kwargs", {}).get("tool_calls")
                if tool_calls:
                    tc_names = [
                        (tc.get("function", {}).get("name") or tc.get("name", "?"))
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "?")
                        for tc in tool_calls
                    ]
                    lines.append(f"[{role}] tool_calls: {', '.join(tc_names)}")
                else:
                    lines.append(f"[{role}] {content[:200]}")
            else:
                lines.append(str(item)[:200])
        return "\n".join(lines)[:limit]
    # dict 처리 — messages 키가 있으면 재귀
    if isinstance(data, dict) and "messages" in data:
        return _fmt(data["messages"], limit)
    try:
        return json.dumps(data, ensure_ascii=False, default=str, indent=2)[:limit]
    except Exception:
        return str(data)[:limit]


def _get_sessions_dir(user_id: str) -> Path:
    """사용자별 세션 저장 디렉터리를 반환한다."""
    path = _APP_DIR / "users" / user_id / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _msg_to_dict(turn) -> dict:
    """LangChain 메시지 객체를 직렬화 가능한 dict로 변환한다."""
    role = getattr(turn, "type", "unknown")
    content = getattr(turn, "content", str(turn))
    tool_calls = getattr(turn, "tool_calls", None) or getattr(turn, "additional_kwargs", {}).get("tool_calls")
    entry: dict = {"role": role, "content": content}
    if tool_calls:
        try:
            entry["tool_calls"] = json.loads(json.dumps(tool_calls, default=str))
        except Exception:
            entry["tool_calls"] = str(tool_calls)
    return entry


def _save_session_snapshot(user_id: str, session_id: str, turn_index: int, messages: list) -> None:
    """LLM 호출 시점의 메시지 스냅샷을 사용자별 세션 마크다운 파일에 추가한다."""
    sessions_dir = _get_sessions_dir(user_id)
    path = sessions_dir / f"{session_id}.md"
    ts = datetime.now().strftime("%H:%M:%S")

    lines = [f"\n## Turn {turn_index}  `{ts}`\n"]
    for msg in messages:
        d = _msg_to_dict(msg)
        role = d["role"]
        content = d.get("content", "")
        tool_calls = d.get("tool_calls")

        if tool_calls:
            # tool_calls 목록을 테이블로
            lines.append(f"**[{role}]** tool_calls\n")
            lines.append("| # | 함수명 | 인자 요약 |")
            lines.append("|---|--------|-----------|")
            for i, tc in enumerate(tool_calls, 1):
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    name = fn.get("name") or tc.get("name", "?")
                    args_raw = fn.get("arguments") or tc.get("arguments", "")
                    try:
                        args_dict = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                        args_str = ", ".join(f"{k}={str(v)[:60]}" for k, v in args_dict.items())
                    except Exception:
                        args_str = str(args_raw)[:120]
                else:
                    name = getattr(tc, "name", "?")
                    args_str = str(getattr(tc, "args", ""))[:120]
                lines.append(f"| {i} | `{name}` | {args_str} |")
            lines.append("")
        elif content:
            # content가 리스트(멀티파트)인 경우 텍스트만 추출
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append("\n-----\ntype: text\n-----")
                            parts.append(item.get("text", ""))
                        elif item.get("type") == "tool_result":
                            parts.append("\n-----\ntype: tool_result\n-----")

                            tc_content = item.get("content", "")
                            if isinstance(tc_content, list):
                                tc_content = "\n".join(
                                    c.get("text", str(c)) if isinstance(c, dict) else str(c)
                                    for c in tc_content
                                )
                            parts.append(f"[tool_result] {tc_content}")
                    else:
                        parts.append(str(item))
                text = "\n".join(p for p in parts if p)
            else:
                text = str(content)
            # 내용이 길면 접을 수 있게 <details> 처리
            if len(text) > 400:
                short = text[:200].replace("\n", " ")
                lines.append(f"**[{role}]** {short}…")
                lines.append(f"<details><summary>전체 보기</summary>\n\n```\n{text}\n```\n</details>\n")
            else:
                lines.append(f"**[{role}]** {text}\n")

    lines.append("\n---")

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_multi_turn_messages(
    messages: list[dict],
) -> tuple[str, list[str], list[str], list[dict]]:
    """
    멀티턴 메시지 배열을 파싱하여
    (최신 쿼리, 현재 턴 파일 경로, 전체 파일 경로, 히스토리 메시지) 반환.

    메시지 포맷:
      {
        "role": "user" | "assistant",
        "content": "...",
        "messagesId": "...",          # 선택
        "metadata": {                  # user 메시지에만 존재 (선택)
          "files": [
            {"filepath": "/path/to/file.pdf", "filename": "...", ...}
          ]
        }
      }

    반환:
      - query            : 마지막 user 메시지의 content
      - current_files    : 마지막 user 메시지에 첨부된 파일 경로 목록
                           → effective_query 의 [첨부 파일] 블록에 사용
      - all_file_paths   : 모든 메시지의 파일 경로 (중복 제거)
                           → RAG 인덱싱에 사용 (이전 턴 파일도 검색 가능하도록)
      - history          : 마지막 메시지를 제외한 이전 대화 [{role, content}, ...]
    """
    if not messages:
        return "", [], [], []

    # 마지막 user 메시지
    last = messages[-1]
    query = last.get("content", "")

    # 현재 턴(마지막 메시지)에 첨부된 파일만 추출
    current_files: list[str] = [
        fi.get("filepath", "").strip()
        for fi in (last.get("metadata") or {}).get("files", [])
        if fi.get("filepath", "").strip()
    ]

    # 전체 메시지에서 파일 경로 수집 (중복 제거, 순서 유지) — RAG용
    seen: set[str] = set()
    all_file_paths: list[str] = []
    for msg in messages:
        for file_info in (msg.get("metadata") or {}).get("files", []):
            fp = file_info.get("filepath", "").strip()
            if fp and fp not in seen:
                seen.add(fp)
                all_file_paths.append(fp)

    # 히스토리: 마지막 메시지를 제외한 이전 대화 (파일 메타데이터 포함)
    history: list[dict] = []
    for msg in messages[:-1]:
        entry: dict = {"role": msg["role"], "content": msg.get("content", "")}
        files = (msg.get("metadata") or {}).get("files", [])
        if files:
            entry["files"] = [
                {"filepath": fi.get("filepath", ""), "filename": fi.get("filename", "")}
                for fi in files
                if fi.get("filepath", "").strip()
            ]
        history.append(entry)

    return query, current_files, all_file_paths, history


async def stream_agent(
    agent,
    query: str | dict,
    user_id: str,
    verbose: int = 0,
) -> None:
    """
    deepagents CompiledStateGraph를 스트리밍으로 실행하고 출력한다.

    query: 문자열 또는 {"role": "user", "content": str | list} 형태의 메시지 dict.
           이미지가 포함된 경우 content는 multimodal content block 리스트.

    verbose 레벨:
      0 — 기본 (도구 입출력만)
      1 — 미들웨어 입출력 + tool_calls 결정 내역
      2 — 모든 LangGraph 체인 이벤트 + LLM 프롬프트 메시지 전체

    히스토리는 state["messages"]에 직접 주입하지 않는다.
    대신 _run_main()에서 /memories/AGENTS.md 에 요약본을 저장하고,
    MemoryMiddleware가 시스템 프롬프트에 주입한다.
    """
    # ContextVar에 user_id 주입 → tools.py의 경로 헬퍼가 자동으로 사용자별 경로 반환
    _current_user_id.set(user_id)

    if isinstance(query, dict):
        user_msg = query
        # 세션 로그용 텍스트 추출
        _query_text = (
            next((b["text"] for b in user_msg["content"] if isinstance(b, dict) and b.get("type") == "text"), "")
            if isinstance(user_msg.get("content"), list)
            else str(user_msg.get("content", ""))
        )
    else:
        user_msg = {"role": "user", "content": query}
        _query_text = query

    state = {"messages": [user_msg]}

    # 세션 ID: user_id + 타임스탬프 + uuid suffix
    session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    llm_turn = 0

    # 세션 파일 헤더 작성
    sessions_dir = _get_sessions_dir(user_id)
    session_path = sessions_dir / f"{session_id}.md"
    with session_path.open("w", encoding="utf-8") as f:
        f.write(f"# 세션 {session_id}\n\n")
        f.write(f"**질의:** {_query_text}\n\n")
        f.write(f"**시작:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n")

    print(f"\n{'━' * 64}")
    print(f"  Deep Agent  사용자: {user_id}  질의: {_query_text}")
    print(f"  세션 ID: {session_id}")
    print(f"{'━' * 64}\n")

    tool_step = 0
    total_input_tokens = 0
    total_output_tokens = 0

    async for event in agent.astream_events(state, version="v2",config={"recursion_limit": 50}):
        kind = event["event"]
        name = event.get("name", "-")

        # ── verbose=2: 모든 체인 이벤트 ─────────────────────────────
        if verbose >= 2:
            if kind == "on_chain_start":
                data = event["data"].get("input")
                # 단순 문자열이나 노이즈 많은 이벤트는 한 줄로 축약
                if isinstance(data, str):
                    print(f"\n  ⬇ chain_start [{name}] {data[:120]}", flush=True)
                elif data:
                    _box(f"⬇  chain_start  {name}", _fmt(data))
            elif kind == "on_chain_end":
                data = event["data"].get("output")
                if isinstance(data, str):
                    print(f"  ⬆ chain_end   [{name}] {data[:120]}", flush=True)
                elif data:
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
                output = event["data"].get("output")
                # SkillsMiddleware: 활성화된 스킬 이름 추출해서 별도 출력
                if "Skills" in name and isinstance(output, dict):
                    active = output.get("active_skills") or output.get("skills") or []
                    if not active:
                        # system_prompt 안에 주입된 스킬 이름을 파싱
                        sp = ""
                        msgs = output.get("messages", [])
                        for m in (msgs if isinstance(msgs, list) else []):
                            if getattr(m, "type", "") == "system" or (isinstance(m, dict) and m.get("role") == "system"):
                                sp = str(getattr(m, "content", m.get("content", "")))
                                break
                        import re
                        active = re.findall(r"^#\s+(.+?\s+Skill)", sp, re.MULTILINE)
                    if active:
                        print(f"\n  [Skills] 활성 스킬: {', '.join(active)}", flush=True)
                    else:
                        print(f"\n  [Skills] 활성 스킬 없음 (기본 동작)", flush=True)
                _box(f"◀ {name}  [output]", _fmt(output))

        # ── LLM 프롬프트 — 세션 저장 + verbose=2 출력 ───────────────
        if kind == "on_chat_model_start":
            messages = event["data"].get("input", {}).get("messages", [])
            # 중첩 리스트 플랫하게 처리
            flat = []
            for m in messages:
                if isinstance(m, list):
                    flat.extend(m)
                else:
                    flat.append(m)

            # 항상 세션 파일에 저장
            llm_turn += 1
            _save_session_snapshot(user_id, session_id, llm_turn, flat)

            if verbose >= 2:
                print(f"\n  {'─'*60}", flush=True)
                print(f"  🤖 LLM 호출 #{llm_turn} — {len(flat)}개 메시지", flush=True)
                for turn in flat:
                    role = getattr(turn, "type", "?") if hasattr(turn, "type") else "?"
                    content = getattr(turn, "content", turn)
                    tool_calls = getattr(turn, "tool_calls", None) or getattr(turn, "additional_kwargs", {}).get("tool_calls")
                    if tool_calls:
                        tc_names = [
                            (tc.get("function", {}).get("name") or tc.get("name", "?"))
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "?")
                            for tc in tool_calls
                        ]
                        print(f"    [{role}] tool_calls → {', '.join(tc_names)}", flush=True)
                    elif content:
                        snippet = str(content)[:200].replace("\n", " ")
                        print(f"    [{role}] {snippet}", flush=True)
                print(f"  {'─'*60}", flush=True)

        elif kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                content = chunk.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            print(block["text"], end="", flush=True)
                elif isinstance(content, str):
                    print(content, end="", flush=True)

        elif kind == "on_chat_model_end":
            msg = event["data"].get("output")
            if msg:
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    inp = usage.get("input_tokens", 0)
                    out = usage.get("output_tokens", 0)
                    details = usage.get("input_token_details", {}) or {}
                    cached = details.get("cache_read", 0) or details.get("cached", 0)
                    uncached = inp - cached
                    total_input_tokens += inp
                    total_output_tokens += out
                    print(
                        f"\n  [토큰] LLM #{llm_turn}"
                        f"  입력={inp:,} (캐시={cached:,} / 신규={uncached:,})"
                        f"  출력={out:,}"
                        f"  (누계 입력={total_input_tokens:,} 출력={total_output_tokens:,})",
                        flush=True,
                    )
                else:
                    print(f"\n  [토큰] LLM #{llm_turn}  토큰 정보 없음", flush=True)
                if verbose >= 1 and hasattr(msg, "tool_calls") and msg.tool_calls:
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
    print(f"  총 토큰  입력={total_input_tokens:,}  출력={total_output_tokens:,}  합계={total_input_tokens + total_output_tokens:,}")
    print(f"{'━' * 64}\n")


# ══════════════════════════════════════════════════════════════════
#  에이전트 빌더
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
#  CompositeBackend 빌더 (사용자별 격리)
# ══════════════════════════════════════════════════════════════════

def build_backend(user_id: str):
    """사용자별 격리된 CompositeBackend 팩토리를 반환한다."""
    user_dir = _APP_DIR / "users" / user_id
    global_skills_dir = _APP_DIR / "skills"

    mem_backend = FilesystemBackend(
        root_dir=str(user_dir / "memories"), virtual_mode=True
    )
    user_skills_backend = FilesystemBackend(
        root_dir=str(user_dir / "skills"), virtual_mode=True
    )
    # team_skills_backend = FilesystemBackend(
    #     root_dir=str(user_dir / "team" / "skills"), virtual_mode=True
    # )
    global_skills_backend = FilesystemBackend(
        root_dir=str(global_skills_dir), virtual_mode=True
    )

    def composite(rt):
        return CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/skills/":       global_skills_backend,   # 전역 공용 스킬
                "/user-skills/":  user_skills_backend,     # 사용자 커스텀 스킬
                "/memories/":     mem_backend,             # 사용자별 단기 메모리
            },
        )

    return composite


async def build_agent(
    user_id: str,
    file_paths: list[str] | None = None,
    store=None,
):
    """사용자 ID를 받아 격리된 deepagents 에이전트를 생성한다."""
    mcp_tools = await get_mcp_tools()                                   # dict[str, list]
    all_mcp = [t for tools in mcp_tools.values() for t in tools]        # 메인 에이전트용 flat
    rag_tools = build_rag_tools(file_paths or [])
    subagents_list = build_subagents(mcp_tools)

    all_tools = [
        *custom_tools,   # 웹 검색, 코드 실행, 메모리
        *file_tools,     # 파일 요약·번역·분석
        *all_mcp,        # 모든 MCP 도구 (서버 무관 flat)
        *rag_tools,      # RAG (첨부 파일 있을 때)
    ]

    print(f"[Deep Agent] user={user_id} | 도구 {len(all_tools)}개, 서브에이전트 {len(subagents_list)}개 로드")

    # model = init_chat_model(
    #     model="openai:/models/Qwen3.5-35B-A3B-FP8",
    #     base_url="http://192.168.1.51:8001/v1",  # vLLM 서버 주소
    #     api_key="EMPTY",                          # vLLM은 키 불필요, 빈 값으로
    #     max_retries=10,
    #     timeout=120,
    #     temperature=0.1,
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

    backend = build_backend(user_id)

    return create_deep_agent(
        model=model,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        subagents=subagents_list,
        middleware=[
            LoggingPIIMiddleware("email",       strategy="mask", apply_to_input=True, apply_to_tool_results=True),
            LoggingPIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_tool_results=True),
            LoggingPIIMiddleware("ip",          strategy="mask", apply_to_input=True, apply_to_tool_results=True),
        ],
        backend=backend,
        store=store,
        skills=["/skills/", "/user-skills/"],   # 전역 + 개인 스킬
        memory=["/memories/AGENTS.md"],          # 사용자별 단기 메모리
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

    # 사용자 ID: 환경변수 > 기본값 "default"
    user_id = os.environ.get("AGENT_USER_ID", "default").strip() or "default"

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url:
        async with AsyncPostgresStore.from_conn_string(db_url) as store:
            await store.setup()
            await _run_main(store, user_id)
    else:
        print(f"[Store] DATABASE_URL 미설정 — InMemoryStore 사용 | user={user_id}", flush=True)
        store = InMemoryStore()
        await _restore_short_term_memory(user_id)
        await _run_main(store, user_id)


def _summarize_history(history: list[dict]) -> str:
    """
    이전 대화 히스토리를 단기 메모리용 마크다운 요약으로 변환한다.

    반환값은 /memories/AGENTS.md 에 저장되어 MemoryMiddleware를 통해
    에이전트 시스템 프롬프트에 주입된다. 에이전트는 이를 바탕으로
    이전 대화 맥락을 이해하고, 필요시 memory_recall/memory_search로
    세부 내용을 조회할 수 있다.
    """
    if not history:
        return ""

    lines = ["## 단기 메모리: 이전 대화 요약\n"]
    lines.append(
        "아래는 현재 세션의 직전 대화 내역 요약입니다. "
        "현재 질의에 이 맥락이 관련될 경우 활용하세요. "
        "세부 내용이 필요하면 memory_recall / memory_search 도구를 사용하세요.\n"
    )

    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = str(msg.get("content", "")).strip()
        preview = content[:300] + ("…" if len(content) > 300 else "")
        role_label = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"**[{i}] {role_label}:** {preview}")

        # 첨부 파일 정보가 있으면 명시적으로 기록
        files = msg.get("files", [])
        if files:
            for fi in files:
                fname = fi.get("filename") or fi.get("filepath", "")
                fpath = fi.get("filepath", "")
                lines.append(f"   - 첨부파일: `{fname}` ({fpath})")
        lines.append("")

    return "\n".join(lines)


def _get_user_memories_dir(user_id: str) -> Path:
    path = _APP_DIR / "users" / user_id / "memories"
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _write_short_term_memory(user_id: str, history: list[dict]) -> None:
    """
    단기 메모리 요약을 users/{user_id}/memories/AGENTS.md 파일에 저장한다.
    FilesystemBackend가 이 파일을 직접 읽어 MemoryMiddleware가 시스템 프롬프트에 주입한다.
    """
    summary = _summarize_history(history)
    text = summary if summary else "(이전 대화 없음)"

    agents_md = _get_user_memories_dir(user_id) / "AGENTS.md"
    agents_md.write_text(text, encoding="utf-8")

    print(
        f"[ShortTermMemory] {len(history)}개 턴 요약 → "
        f"users/{user_id}/memories/AGENTS.md 저장",
        flush=True,
    )


async def _restore_short_term_memory(user_id: str) -> None:
    """FilesystemBackend 방식에서는 파일이 그대로 남아 있으므로 별도 복원 불필요."""
    agents_md = _get_user_memories_dir(user_id) / "AGENTS.md"
    if agents_md.exists():
        print(
            f"[ShortTermMemory] 이전 세션 메모리 유지 → "
            f"users/{user_id}/memories/AGENTS.md",
            flush=True,
        )


async def _run_main(store, user_id: str = "default") -> None:
    # ── 멀티턴 메시지 포맷 예시 ─────────────────────────────────────
    # parse_multi_turn_messages() 에 아래 messages 배열을 전달하면
    # (query, file_paths, history) 를 자동으로 추출합니다.
    #
    messages = [
        # {
        #     "role": "user",
        #     "content": "이미지의 텍스트를 추출해줘",
        #     "metadata": {
        #         "files": [
        #             {"filepath": "/Users/sso/Downloads/workspace/계약서.png",
        #              "filename": "계약서.png"},
        #         ]
        #     },
        # },
        {
            "role": "user",
            "content": "무슨 내용이야?",
            "metadata": {
                "files": [
                    {"filepath": "/Users/sso/Downloads/workspace/근로 계약서 양식.docx",
                     "filename": "근로 계약서 양식.docx"},
                ]
            },
        },
        # {
        #     "role": "user",
        #     "content": "첨부한 문서들을 분석해서 보고서로 만들어줘",
        #     "metadata": {
        #         "files": [
        #             {"filepath": "/Users/sso/Downloads/workspace/수강신청_장바구니_메뉴얼.pdf",
        #              "filename": "수강신청_장바구니_메뉴얼.pdf"},
        #             {"filepath": "/Users/sso/Downloads/workspace/files/2023학년도 개척리더 해외연수 공고문_최종.pdf",
        #              "filename": "2023학년도 개척리더 해외연수 공고문_최종.pdf"},
        #         ]
        #     },
        # },
        # {
        #     "role": "assistant",
        #     "content": "1) 작성자 상세 정보 화면 \n- 작성자의 프로필 및 상세 정보를 확인하는 화면 "\
        #         "- 구성: 뒤로가기 버튼 / 작성자 정보 / 작성자 문서 목록  - 작성자 정보: 사진, 이름, 부서, 직급, 연락처, 이메일, 열람수, 좋아요 등 "\
        #         "- 문서 목록: 작성자가 만든/수정한 문서들을 최신순으로 표시  - 정렬 방식 변경 가능 (열람순, 인기순)  - 문서별 미리보기, 이름, 저장 경로 등 확인 가능 ."\
        #         "2) 문서 상세 정보 화면 - 문서 클릭 또는 메뉴에서 ‘세부 정보’ 선택 시 진입  - 구성: 뒤로가기 버튼 / 문서 정보 / 문서 내용  "\
        #         "- 문서 정보: 미리보기, 이름, 카테고리, 해시태그 등  - 상세 메타데이터: 생성일, 수정일, 용량, 작성자, 수정자  - 작성자 클릭 시 작성자 상세 화면으로 이동 "\
        #         "- 열람수·좋아요로 문서 인기도 확인, 좋아요 기능 제공  - 공유 링크로 내부 사용자에게 문서 공유 가능  - 문서 전체 내용을 텍스트로 확인 가능"
        # },
        # {
        #     "role": "user",
        #     "content": "이 내용을 번역해서 보고서로 만들어줘"
        # },
        # {
        #     "role": "assistant",
        #     "content": "보고서를 파일로 저장했어.  \n경로: /worksp ││ ace/document_translation_summary_20260402_043025.md "
        # },
        # {
        #     "role": "user",
        #     "content": "오늘 대한민국의 날씨는 어때?"
            
        # },
        # {
        #     "role": "assistant",
        #     "content": " 오늘(4월 2일) 기준으로 보면 한국은 전반적으로 **봄 날씨**야.  - 아침: 약 4~8도 → 쌀쌀함 │ - 낮: 약 15~20도 → 포근함  - 하늘: 대체로 맑거나 구름 조금"
        # },
        # {
        #     "role": "user",
        #     "content": "첨부한 문서들을 각각 분석하고, 배포용 보고서 양식으로 만들어줘",
        #     "metadata": {
        #         "files": [
        #             {"filepath": "/Users/sso/Downloads/workspace/근로 계약서 양식.docx",
        #              "filename": "근로 계약서 양식.docx"},
        #             {"filepath": "/Users/sso/Downloads/workspace/수강신청_장바구니_메뉴얼.pdf",
        #              "filename": "수강신청_장바구니_메뉴얼.pdf"},
        #         ]
        #     },
        # },
        # {
        #     "role": "assistant",
        #     "content": "비플식권_매뉴얼 사용자용 2023_에 대한 내용입니다.~~"\
        #         "다음은 점심식대 지원제도 안내에 대한 내용입니다.~~"
        # },
        # {
        #     "role": "user",
        #     "content": "langchain에서 개발한 deep agent에 대해 조사해줘. middleware,skills,sandbox 등의 기능에 대해서 세세하게 조사하고, create_agent와의 차이점을 비교해줘. 이 모든 내용을 기반으로 docx 보고서로 만들어줘"
            
        # },
        # {
        #     "role": "user",
        #     "content": "처음에 첨부햇던 파일에 대한 정보 좀 알려줘"
            
        # },
    ]
    query, current_files, all_file_paths, history = parse_multi_turn_messages(messages)

    # ── 단순 쿼리 모드 (기존 방식) ──────────────────────────────────
    # query, current_files = parse_args(sys.argv[1:])
    # query, current_files, all_file_paths, history = (
    #     "엠클라우독이라는 국내 기업과 비슷한 LLM 상용 서비스 관련 한국 경쟁사 5개 조사하고,"
    #     " 각각 제공 기능을 분석하고, 리포트 쓰고, 슬라이드도 만들어줘",
    #     [], [], [],
    # )

    # 히스토리를 단기 메모리로 오프로드 — state["messages"]에 직접 주입하지 않음
    # MemoryMiddleware가 /memories/AGENTS.md 를 읽어 시스템 프롬프트에 자동 주입
    if history:
        await _write_short_term_memory(user_id, history)

    # RAG 인덱싱에는 과거 첨부 파일 포함, 에이전트 지시에는 현재 턴 파일만 명시
    agent = await build_agent(user_id, all_file_paths, store=store)

    user_msg = build_user_message(query, current_files)
    await stream_agent(agent, user_msg, user_id=user_id, verbose=2)


# ── invoke 방식 (비스트리밍 / 프로그래매틱 호출용) ─────────────────

async def invoke_agent(
    query: str,
    user_id: str = "default",
    file_paths: list[str] | None = None,
    history: list[dict] | None = None,
    current_files: list[str] | None = None,
) -> str:
    """에이전트를 비스트리밍으로 실행하고 최종 답변 문자열을 반환한다."""
    require_api_keys()
    _current_user_id.set(user_id)

    # 히스토리는 state["messages"]에 직접 주입하지 않고 단기 메모리로 오프로드
    user_msg = build_user_message(query, current_files or [])
    state = {"messages": [user_msg]}

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url:
        async with AsyncPostgresStore.from_conn_string(db_url) as store:
            await store.setup()
            if history:
                await _write_short_term_memory(user_id, history)
            agent = await build_agent(user_id, file_paths, store=store)
            result = await agent.ainvoke(state)
    else:
        store = InMemoryStore()
        if history:
            await _write_short_term_memory(user_id, history)
        agent = await build_agent(user_id, file_paths, store=store)
        result = await agent.ainvoke(state)
    return result["messages"][-1].content


async def invoke_agent_from_messages(messages: list[dict]) -> str:
    """
    멀티턴 메시지 배열을 받아 에이전트를 비스트리밍으로 실행한다.

    메시지 포맷:
      {"role": "user"|"assistant", "content": "...", "messagesId": "...",
       "metadata": {"files": [{"filepath": "/path/to/file", ...}]}}

    마지막 user 메시지를 현재 쿼리로, 이전 메시지들을 단기 메모리로 오프로드하고,
    모든 메시지의 파일을 RAG로 처리한다.
    """
    query, current_files, all_file_paths, history = parse_multi_turn_messages(messages)
    return await invoke_agent(
        query,
        file_paths=all_file_paths,
        history=history,
        current_files=current_files,
    )


if __name__ == "__main__":
    asyncio.run(main())
