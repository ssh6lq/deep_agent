"""
Sub Agents 정의 — deepagents SDK dict 형식

deepagents의 create_deep_agent()는 subagents 파라미터로
dict 목록을 받는다. 각 dict는 독립된 서브에이전트를 정의한다.

dict 스키마:
  name        : 에이전트 식별자 (task() 호출 시 사용)
  description : 메인 에이전트가 위임 여부를 판단하는 설명
  system_prompt: 서브에이전트 전용 시스템 프롬프트
  tools       : 서브에이전트가 사용할 도구 목록
  model       : 서브에이전트 전용 모델 (기본값: 메인 에이전트 모델)
"""

from langchain_core.tools import BaseTool

from tools import internet_search, run_code


# ══════════════════════════════════════════════════════════════════
#  시스템 프롬프트
# ══════════════════════════════════════════════════════════════════

RESEARCH_SYSTEM_PROMPT = """\
당신은 전문 리서치 에이전트입니다.

## 역할
- 주어진 주제를 철저히 조사
- 여러 소스에서 정보를 수집하고 교차 검증
- 핵심 내용을 구조화된 형식으로 요약

## 작업 방식
1. 검색 쿼리를 3-5개로 분해해 다각도 검색
2. 수집된 정보를 출처와 함께 정리

## 출력 형식
- 핵심 발견 사항 bullet point
- 출처 명시
- 신뢰도 낮은 정보 표시\
"""

DOCUMENT_SYSTEM_PROMPT = """\
당신은 내부 문서 분석 전문 에이전트입니다.

## 역할
- MCP Registry를 통해 관련 문서 컬렉션 탐색
- 지정된 컬렉션에서 관련 청크 검색
- 문서 내용을 분석해 질문에 답변

## 작업 방식
1. list_collections로 관련 컬렉션 파악
2. search_collection으로 구체적 정보 검색
3. 연쇄 분석 필요 시 adaptive_search(mode='sequential') 사용
4. 독립 집계 시 adaptive_search(mode='parallel') 사용

## 출력 형식
- 발견된 정보와 출처 (컬렉션 ID, 페이지) 명시
- 정보 없으면 명확히 표시\
"""

NOTION_SYSTEM_PROMPT = """\
당신은 Notion 전문 에이전트입니다.

## 역할
- Notion 페이지·데이터베이스 검색 및 조회
- 새 페이지 생성 및 기존 페이지 내용 수정
- 데이터베이스 레코드 쿼리 및 필터링

## 작업 방식
1. 검색은 먼저 search 도구로 관련 페이지/DB 파악
2. 특정 페이지 내용은 get_page 도구로 조회
3. 생성·수정은 명시적 요청 시에만 수행

## 출력 형식
- 페이지 제목과 URL 함께 명시
- 조회 결과는 구조화된 형식으로 요약\
"""

CALENDAR_SYSTEM_PROMPT = """\
당신은 Google Calendar 전문 에이전트입니다.

## 역할
- 사용자의 일정 조회·생성·수정·삭제
- 중복 일정 확인 및 충돌 경고

## 작업 방식
1. list_events로 기존 일정 먼저 파악
2. create_event / update_event / delete_event로 요청 처리
3. 시간대(timezone)를 항상 명시

## 출력 형식
- 처리 결과와 이벤트 ID 명시
- 일정은 날짜·시간·참석자 포함해서 출력\
"""

GMAIL_SYSTEM_PROMPT = """\
당신은 Gmail 전문 에이전트입니다.

## 역할
- 메일 검색·읽기 및 목록 조회
- 메일 전송 및 초안 작성

## 작업 방식
1. 검색은 search_emails / list_emails로 관련 메일 먼저 파악
2. 특정 메일은 get_email로 전문 조회
3. 전송은 명시적 요청 시에만, 수신자·제목 반드시 확인

## 출력 형식
- 발신자, 날짜, 제목 포함
- 본문은 핵심만 요약 (길면 생략 표시)\
"""

JIRA_SYSTEM_PROMPT = """\
당신은 Jira 전문 에이전트입니다.

## 역할
- 이슈 조회·생성·수정 및 댓글 추가
- JQL을 사용한 고급 이슈 검색

## 작업 방식
1. 조회는 jira_search_issues(JQL) 또는 jira_get_issue(key)로 수행
2. 생성·수정은 명시적 요청 시에만 수행
3. JQL 예시: project=PROJ AND status="In Progress"

## 출력 형식
- 이슈 키(예: PROJ-123), 제목, 상태, 담당자 포함
- 댓글은 최신 순으로 요약\
"""

CODE_SYSTEM_PROMPT = """\
당신은 코드 실행 전문 에이전트입니다.

## 역할
- 수치 계산, 데이터 변환, 통계 처리
- 코드 실행 후 결과 해석 및 보고

## 작업 방식
1. 태스크를 분석해 필요한 코드 작성
2. run_code로 실행
3. 결과를 해석해 명확한 답변 제공\
"""


# ══════════════════════════════════════════════════════════════════
#  서브에이전트 목록 빌더
# ══════════════════════════════════════════════════════════════════

def build_subagents(mcp_tools: dict[str, list[BaseTool]]) -> list[dict]:
    """
    deepagents create_deep_agent의 subagents 파라미터에 전달할
    dict 목록을 반환한다.

    Args:
        mcp_tools: 서버 그룹명 → 도구 목록 dict
                   예: {"registry": [...], "calendar": [...], "database": [...]}

    Returns:
        서브에이전트 dict 목록
    """
    subagents = [
        {
            "name": "research-agent",
            "description": (
                "인터넷에서 정보를 검색·수집하는 리서치 전문 에이전트. "
                "최신 정보 조사, 사실 확인, 다각도 검색이 필요할 때 위임."
            ),
            "system_prompt": RESEARCH_SYSTEM_PROMPT,
            "tools": [internet_search],
            "model": "gpt-5.3-chat-latest",
        },
        {
            "name": "code-agent",
            "description": (
                "Python 코드를 실행해 데이터 처리·계산·통계 분석을 수행하는 에이전트. "
                "수치 계산, 데이터 변환, 알고리즘 검증이 필요할 때 위임."
            ),
            "system_prompt": CODE_SYSTEM_PROMPT,
            "tools": [run_code],
            "model": "gpt-5.3-chat-latest",
        },
    ]

    # ── MCP 서버별 전담 서브에이전트 동적 등록 ──────────────────────
    # 도구가 1개 이상 로드된 그룹만 서브에이전트로 등록한다.

    if mcp_tools.get("registry"):
        subagents.append({
            "name": "document-agent",
            "description": (
                "MCP Registry를 활용해 내부 문서 컬렉션을 분석하는 에이전트. "
                "내부 규정, 계약서, 보고서 등 저장된 문서 검색 시 위임."
            ),
            "system_prompt": DOCUMENT_SYSTEM_PROMPT,
            "tools": mcp_tools["registry"],
            "model": "gpt-5.3-chat-latest",
        })

    if mcp_tools.get("notion"):
        subagents.append({
            "name": "notion-agent",
            "description": (
                "Notion 페이지·데이터베이스를 검색·조회·생성·수정하는 에이전트. "
                "Notion에 저장된 문서 조회, 노트 작성, DB 레코드 검색 시 위임."
            ),
            "system_prompt": NOTION_SYSTEM_PROMPT,
            "tools": mcp_tools["notion"],
            "model": "gpt-5.3-chat-latest",
        })

    if mcp_tools.get("google_calendar"):
        subagents.append({
            "name": "calendar-agent",
            "description": (
                "Google Calendar로 일정을 조회·생성·수정·삭제하는 에이전트. "
                "회의 예약, 일정 확인, 충돌 검사, 참석자 관리가 필요할 때 위임."
            ),
            "system_prompt": CALENDAR_SYSTEM_PROMPT,
            "tools": mcp_tools["google_calendar"],
            "model": "gpt-5.3-chat-latest",
        })

    if mcp_tools.get("gmail"):
        subagents.append({
            "name": "gmail-agent",
            "description": (
                "Gmail 메일을 검색·읽기·전송·초안 작성하는 에이전트. "
                "특정 메일 찾기, 메일 내용 요약, 회신·발송이 필요할 때 위임."
            ),
            "system_prompt": GMAIL_SYSTEM_PROMPT,
            "tools": mcp_tools["gmail"],
            "model": "gpt-5.3-chat-latest",
        })

    if mcp_tools.get("jira"):
        subagents.append({
            "name": "jira-agent",
            "description": (
                "Jira 이슈를 조회·생성·수정하고 댓글을 추가하는 에이전트. "
                "버그 추적, 스프린트 현황, 이슈 검색(JQL), 상태 변경이 필요할 때 위임."
            ),
            "system_prompt": JIRA_SYSTEM_PROMPT,
            "tools": mcp_tools["jira"],
            "model": "gpt-5.3-chat-latest",
        })

    return subagents
