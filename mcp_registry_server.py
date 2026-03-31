"""
MCP Registry Server

fastmcp 기반 독립 실행 MCP 서버.
deepagents의 tools 배열에 MCP 서버 중 하나로 등록됨.

실행: python mcp_registry_server.py

제공 도구:
  - list_collections(query)           : 관련 컬렉션 탐색
  - search_collection(id, query)      : 청크 검색
  - adaptive_search(query, col_ids)   : Adaptive Fan-out 실행
"""

import asyncio
import json
from typing import Literal

from fastmcp import FastMCP

# ── 타입 정의 ───────────────────────────────────────────────────────

CollectionMeta = dict  # {id, description, keywords}
Chunk = dict           # {content, metadata, score}

# ── 인메모리 레지스트리 (실제 서비스: VectorDB로 교체) ──────────────

collection_registry: list[CollectionMeta] = [
    {
        "id": "hr_docs",
        "description": "인사규정, 연차, 복리후생, 채용 정책",
        "keywords": ["인사", "hr", "연차", "복리후생", "규정", "직원"],
    },
    {
        "id": "sales_docs",
        "description": "영업 계약서, 거래 조건, 고객사 현황",
        "keywords": ["영업", "계약", "거래", "고객", "매출", "sales"],
    },
    {
        "id": "marketing_docs",
        "description": "마케팅 캠페인, 브랜드 가이드라인, 광고 예산",
        "keywords": ["마케팅", "광고", "캠페인", "브랜드", "홍보"],
    },
    {
        "id": "legal_docs",
        "description": "법무 검토, 계약 법률 조항, 규제 준수",
        "keywords": ["법무", "법률", "규제", "준수", "계약 조항"],
    },
]

# ── 컬렉션별 샘플 청크 (실제: Chroma/FAISS 등 VectorDB) ────────────

collection_chunks: dict[str, list[Chunk]] = {
    "hr_docs": [
        {
            "content": "연차 휴가는 입사 1년 후 15일이 부여되며, 매년 1일씩 추가됩니다.",
            "metadata": {"source": "hr_policy_2024.pdf", "page": "3"},
            "score": 0.92,
        },
        {
            "content": "복리후생: 식대 월 10만원, 교통비 월 5만원, 의료비 연 50만원 지원.",
            "metadata": {"source": "benefits_guide.pdf", "page": "7"},
            "score": 0.88,
        },
    ],
    "sales_docs": [
        {
            "content": "계약 해지 시 30일 전 서면 통보 필요. 위약금은 잔여 계약금액의 10%.",
            "metadata": {"source": "standard_contract_v3.pdf", "page": "12"},
            "score": 0.91,
        },
        {
            "content": "Q3 영업 실적: 총 매출 42억원, 신규 고객 18개사, 계약 갱신율 87%.",
            "metadata": {"source": "q3_report.pdf", "page": "2"},
            "score": 0.85,
        },
    ],
    "marketing_docs": [
        {
            "content": "2024 마케팅 예산: 디지털 광고 40%, SNS 마케팅 30%, 오프라인 이벤트 30%.",
            "metadata": {"source": "marketing_plan_2024.pdf", "page": "5"},
            "score": 0.89,
        },
    ],
    "legal_docs": [
        {
            "content": "개인정보보호법 준수 의무: 수집 데이터 암호화 필수, 보관 기간 3년.",
            "metadata": {"source": "compliance_checklist.pdf", "page": "1"},
            "score": 0.93,
        },
    ],
}

# ── 키워드 기반 유사도 스코어링 (실제: 임베딩 유사도) ───────────────

def score_collection(query: str, meta: CollectionMeta) -> float:
    q = query.lower()
    score = 0.0
    for kw in meta["keywords"]:
        if kw.lower() in q:
            score += 0.2
    if meta["id"].replace("_", "") in q:
        score += 0.3
    return min(score, 1.0)


# ══════════════════════════════════════════════════════════════════
#  FastMCP 서버 생성
# ══════════════════════════════════════════════════════════════════

mcp = FastMCP("mcp-registry")


# ── 도구 1: list_collections ────────────────────────────────────

@mcp.tool()
def list_collections(query: str, k: int = 5) -> str:
    """질의와 관련된 컬렉션 목록과 유사도 점수를 반환한다."""
    scored = [
        {
            "id": meta["id"],
            "name": meta["id"],
            "description": meta["description"],
            "score": score_collection(query, meta),
        }
        for meta in collection_registry
    ]
    scored = [c for c in scored if c["score"] > 0.1]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(scored[:k], ensure_ascii=False, indent=2)


# ── 도구 2: search_collection ───────────────────────────────────

@mcp.tool()
def search_collection(collection_id: str, query: str, k: int = 4) -> str:
    """지정한 컬렉션에서 쿼리와 관련된 청크를 검색한다."""
    chunks = collection_chunks.get(collection_id)
    if chunks is None:
        available = ", ".join(collection_chunks.keys())
        return f"[Error] Collection '{collection_id}' not found. Available: {available}"

    # 간단한 키워드 re-ranking (실제: 임베딩 유사도)
    words = query.split()
    results = [
        {
            **c,
            "score": sum(1 for w in words if w in c["content"]) * 0.15 + c["score"],
        }
        for c in chunks
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(results[:k], ensure_ascii=False, indent=2)


# ── 도구 3: adaptive_search (Adaptive Fan-out) ───────────────────

@mcp.tool()
def adaptive_search(
    query: str,
    collection_ids: list[str],
    mode: Literal["sequential", "parallel"],
) -> str:
    """
    질의 성격에 따라 Sequential 또는 Fan-out(병렬) 방식으로
    여러 컬렉션을 검색한다.
    mode='sequential'이면 이전 결과를 다음 쿼리에 포함.
    """
    results = []

    if mode == "parallel":
        # Mode B: Fan-out — 동시 실행
        for col_id in collection_ids:
            chunks = collection_chunks.get(col_id, [])
            results.append({
                "collection_id": col_id,
                "mode": "parallel",
                "chunks": chunks[:4],
            })
    else:
        # Mode A: Sequential — 이전 결과를 다음 쿼리에 포함
        accumulated_context = ""
        for col_id in collection_ids:
            enriched_query = query + accumulated_context
            chunks = collection_chunks.get(col_id, [])[:4]
            results.append({
                "collection_id": col_id,
                "mode": "sequential",
                "chunks": chunks,
                "enrichedQuery": enriched_query,
            })
            # 다음 쿼리를 위해 컨텍스트 누적
            summary = " | ".join(c["content"][:80] for c in chunks)
            accumulated_context += f"\n[{col_id}]: {summary}"

    return json.dumps(results, ensure_ascii=False, indent=2)


# ── 서버 실행 ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("[MCP Registry] 서버 실행 중 (stdio transport)", file=sys.stderr)
    mcp.run(transport="stdio")
