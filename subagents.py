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

def build_subagents(mcp_tools: list[BaseTool]) -> list[dict]:
    """
    deepagents create_deep_agent의 subagents 파라미터에 전달할
    dict 목록을 반환한다.

    Args:
        mcp_tools: MCP Registry 도구 목록 (document-agent에 전달)

    Returns:
        서브에이전트 dict 목록
    """
    mcp_tool_names = {"list_collections", "search_collection", "adaptive_search"}
    mcp_for_doc = [t for t in mcp_tools if t.name in mcp_tool_names]

    return [
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
            "name": "document-agent",
            "description": (
                "MCP Registry를 활용해 내부 문서 컬렉션을 분석하는 에이전트. "
                "내부 규정, 계약서, 보고서 등 저장된 문서 검색 시 위임."
            ),
            "system_prompt": DOCUMENT_SYSTEM_PROMPT,
            "tools": mcp_for_doc,
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
