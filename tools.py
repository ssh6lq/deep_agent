"""
Tools

커스텀 도구 모음:
  internet_search : 웹 검색 (Tavily)
  run_code        : Python 코드 실행
  memory_save     : 메모리 저장 (사용자별 JSON 파일 영구 보존)
  memory_recall   : 메모리 조회
  memory_list     : 저장된 키 목록 확인
  memory_delete   : 특정 키 삭제
  memory_search   : 메모리 내 키워드 검색
  memory_clear    : 전체 메모리 초기화

변경 사항 (멀티유저):
  - _current_user_id ContextVar 도입 → 사용자별 메모리/세션 경로 분리
  - _get_user_dir()  : users/{user_id}/ 동적 반환
  - _get_memory_file(): users/{user_id}/.agent_memory.json 동적 반환
  - _get_sessions_dir(): users/{user_id}/sessions/ 동적 반환
"""

import json
import os
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# ── 앱 루트 디렉터리 ────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent

# ── 사용자 컨텍스트 변수 ────────────────────────────────────────────
# main.py에서 ContextVar.set()으로 주입 → 모든 도구가 사용자별 경로 자동 분기
_current_user_id: ContextVar[str] = ContextVar("current_user_id", default="default")


# ── 사용자별 경로 헬퍼 ──────────────────────────────────────────────

def _get_user_dir() -> Path:
    uid = _current_user_id.get()
    path = _APP_DIR / "users" / uid
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_memory_file() -> Path:
    """사용자별 영구 키-값 메모리 파일 경로를 반환한다."""
    return _get_user_dir() / ".agent_memory.json"


def _get_sessions_dir() -> Path:
    """사용자별 세션 저장 디렉터리를 반환한다."""
    path = _get_user_dir() / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════
#  Web Search — Tavily
# ══════════════════════════════════════════════════════════════════

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> str:
    """인터넷 웹 검색을 실행한다. 최신 정보나 일반 지식 검색에 사용."""
    search = TavilySearchResults(
        max_results=max_results,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
    )
    return str(search.invoke(query))


# ══════════════════════════════════════════════════════════════════
#  Code Execution — Python eval/exec sandbox
# ══════════════════════════════════════════════════════════════════

@tool
def run_code(code: str) -> str:
    """Python 코드를 실행하고 결과를 반환한다. 계산, 데이터 변환에 활용."""
    import contextlib
    import io

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            try:
                value = eval(code)  # noqa: S307
                captured = stdout_buf.getvalue().strip()
                if value is not None:
                    return f"{captured}\n{value}".strip()
                return captured if captured else "실행 완료 (출력 없음)"
            except SyntaxError:
                pass

            namespace: dict = {}
            exec(code, namespace)  # noqa: S102

        captured = stdout_buf.getvalue().strip()
        if captured:
            return captured
        if "result" in namespace and namespace["result"] is not None:
            return str(namespace["result"])
        user_vars = {
            k: v for k, v in namespace.items()
            if not k.startswith("_") and not callable(v) and v is not None
        }
        if user_vars:
            last_key, last_val = list(user_vars.items())[-1]
            return f"{last_key} = {last_val}"
        return "실행 완료 (출력 없음)"
    except Exception as exc:
        return f"[Error] {exc}"


# ══════════════════════════════════════════════════════════════════
#  Memory — JSON 파일 기반 영구 키-값 저장소
#
#  .agent_memory.json 에 저장되어 세션 종료 후에도 유지됨.
#  각 항목: { key: { "value": str, "updated_at": str } }
# ══════════════════════════════════════════════════════════════════

def _load_store() -> dict:
    """JSON 파일에서 메모리 전체를 로드한다."""
    mem_file = _get_memory_file()
    if mem_file.exists():
        try:
            return json.loads(mem_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _dump_store(data: dict) -> None:
    """메모리 전체를 JSON 파일에 저장한다."""
    _get_memory_file().write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@tool
def memory_save(key: str, value: str) -> str:
    """
    정보를 키-값으로 메모리에 저장한다.
    세션이 종료되어도 유지된다 (JSON 파일 영구 보존).

    Args:
        key  : 저장 키 (예: "사용자이름", "프로젝트마감일")
        value: 저장할 내용
    """
    from datetime import datetime

    store = _load_store()
    store[key] = {
        "value": value,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _dump_store(store)
    return f"[Memory] '{key}' 저장 완료 (영구 보존)"


@tool
def memory_recall(key: str) -> str:
    """
    메모리에서 저장된 정보를 키로 조회한다.

    Args:
        key: 조회할 키
    """
    store = _load_store()
    entry = store.get(key)
    if entry is None:
        keys = list(store.keys())
        hint = f"저장된 키 목록: {', '.join(keys)}" if keys else "저장된 항목 없음"
        return f"[Memory] '{key}' 없음. {hint}"
    return f"[Memory] {key}: {entry['value']}  (저장일시: {entry.get('updated_at', '?')})"


@tool
def memory_list() -> str:
    """
    저장된 모든 메모리 키와 값을 목록으로 반환한다.
    어떤 정보가 기억되어 있는지 전체 확인 시 사용.
    """
    store = _load_store()
    if not store:
        return "[Memory] 저장된 항목이 없습니다."

    lines = [f"[Memory] 저장된 항목 {len(store)}개:"]
    for key, entry in store.items():
        val_preview = str(entry["value"])[:80].replace("\n", " ")
        updated = entry.get("updated_at", "?")
        lines.append(f"  • {key}: {val_preview}  ({updated})")
    return "\n".join(lines)


@tool
def memory_delete(key: str) -> str:
    """
    메모리에서 특정 키를 삭제한다.

    Args:
        key: 삭제할 키
    """
    store = _load_store()
    if key not in store:
        return f"[Memory] '{key}' 없음. 삭제 불필요."
    del store[key]
    _dump_store(store)
    return f"[Memory] '{key}' 삭제 완료"


@tool
def memory_search(query: str) -> str:
    """
    메모리에서 키 또는 값에 특정 키워드가 포함된 항목을 검색한다.

    Args:
        query: 검색할 키워드
    """
    store = _load_store()
    query_lower = query.lower()
    matches = {
        k: v for k, v in store.items()
        if query_lower in k.lower() or query_lower in str(v["value"]).lower()
    }
    if not matches:
        return f"[Memory] '{query}' 관련 항목을 찾을 수 없습니다."

    lines = [f"[Memory] '{query}' 검색 결과 {len(matches)}건:"]
    for key, entry in matches.items():
        val_preview = str(entry["value"])[:80].replace("\n", " ")
        lines.append(f"  • {key}: {val_preview}")
    return "\n".join(lines)


@tool
def memory_clear() -> str:
    """저장된 모든 메모리를 초기화한다. 되돌릴 수 없으니 신중히 사용."""
    store = _load_store()
    count = len(store)
    _dump_store({})
    return f"[Memory] 전체 {count}개 항목 삭제 완료"


# ══════════════════════════════════════════════════════════════════
#  세션 오프로드 도구
# ══════════════════════════════════════════════════════════════════

@tool
def save_turn_result(thread_id: str, turn_num: int, content: str) -> str:
    """
    현재 턴의 중요 결과를 세션 폴더에 마크다운 파일로 오프로드한다.
    컨텍스트 창 절약이 필요한 긴 중간 결과물을 저장할 때 사용.

    Args:
        thread_id: 세션(스레드) 식별자
        turn_num:  저장할 턴 번호
        content:   저장할 내용 (마크다운 형식 권장)
    """
    session_dir = _get_sessions_dir() / thread_id
    session_dir.mkdir(parents=True, exist_ok=True)

    file_path = session_dir / f"turn_{turn_num}_result.md"
    file_path.write_text(content, encoding="utf-8")
    return f"[저장 완료] {file_path}"


# ══════════════════════════════════════════════════════════════════
#  전체 사용자 정의 도구 목록
# ══════════════════════════════════════════════════════════════════

custom_tools = [
    internet_search,
    run_code,
    memory_save,
    memory_recall,
    memory_list,
    memory_delete,
    memory_search,
    memory_clear,
    save_turn_result,
]
