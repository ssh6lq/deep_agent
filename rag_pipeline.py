"""
RAG Pipeline — Context Injection / Chunk-Docs / VectorDB / Retriever

다이어그램 v8: DocumentLoader → RAG Pipeline
  Path A: ConversationBuffer + Memory → LangChain Memory
  Path B: VectorDB Retriever (filename, session_id)

구성:
  1. 문서 청킹   : RecursiveCharacterTextSplitter
  2. 임베딩      : OpenAIEmbeddings (또는 HuggingFace fallback)
  3. VectorStore : FAISS (인메모리) — 실제 서비스: Chroma / Pinecone 교체 가능
  4. Retriever   : similarity_search with score
  5. Context Injection: 검색 결과를 시스템 프롬프트에 주입
"""

import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import Document, load_documents


# ══════════════════════════════════════════════════════════════════
#  청킹 설정
# ══════════════════════════════════════════════════════════════════

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def chunk_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[LCDocument]:
    """
    Document 리스트를 LangChain Document 청크로 분할한다.
    이미지는 청킹에서 제외(이미지는 별도 Vision 경로 처리).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    lc_docs: list[LCDocument] = []
    for doc in docs:
        if doc.doc_type == "image":
            continue  # 이미지는 Vision LLM 경로 사용
        if not doc.content.strip():
            continue

        chunks = splitter.split_text(doc.content)
        for i, chunk in enumerate(chunks):
            lc_docs.append(
                LCDocument(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "doc_type": doc.doc_type,
                    },
                )
            )

    return lc_docs


# ══════════════════════════════════════════════════════════════════
#  임베딩 초기화 (OpenAI 우선, 실패 시 HuggingFace)
# ══════════════════════════════════════════════════════════════════

def get_embeddings():
    """임베딩 모델을 반환한다. OpenAI API 키가 없으면 HuggingFace를 시도한다."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except ImportError:
            pass

    # HuggingFace fallback (로컬, API 키 불필요)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    except ImportError:
        raise RuntimeError(
            "임베딩 모델을 초기화할 수 없습니다.\n"
            "  pip install langchain-openai  또는\n"
            "  pip install langchain-huggingface sentence-transformers"
        )


# ══════════════════════════════════════════════════════════════════
#  RAGPipeline 클래스
# ══════════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float


class RAGPipeline:
    """
    RAG Pipeline — 문서 인덱싱 및 검색.

    사용:
        pipeline = RAGPipeline()
        pipeline.add_documents(docs)       # Document 리스트 추가
        chunks = pipeline.retrieve(query)  # 검색
        context = pipeline.get_context(query)  # 컨텍스트 문자열
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        k: int = 4,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self._vectorstore = None
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def add_documents(self, docs: list[Document]) -> int:
        """Document 리스트를 청킹 후 VectorStore에 추가한다. 추가된 청크 수 반환."""
        lc_docs = chunk_documents(docs, self.chunk_size, self.chunk_overlap)
        if not lc_docs:
            return 0

        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise RuntimeError("pip install langchain-community faiss-cpu")

        if self._vectorstore is None:
            self._vectorstore = FAISS.from_documents(lc_docs, self.embeddings)
        else:
            self._vectorstore.add_documents(lc_docs)

        print(f"[RAG] {len(lc_docs)}개 청크 인덱싱 완료")
        return len(lc_docs)

    def add_files(self, paths: list[str]) -> int:
        """파일 경로 목록을 로드 후 인덱싱한다."""
        docs = load_documents(paths)
        return self.add_documents(docs)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """쿼리와 가장 관련 있는 청크를 반환한다."""
        if self._vectorstore is None:
            return []

        k = k or self.k
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        return [
            RetrievedChunk(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
            )
            for doc, score in results
        ]

    def get_context(self, query: str, k: int | None = None) -> str:
        """
        검색 결과를 시스템 프롬프트 주입용 컨텍스트 문자열로 반환한다.
        (Context Injection)
        """
        chunks = self.retrieve(query, k)
        if not chunks:
            return ""

        lines = ["[관련 문서 컨텍스트]"]
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "")
            page_str = f" p.{page}" if page else ""
            lines.append(f"\n--- 출처 {i}: {source}{page_str} (유사도: {chunk.score:.3f}) ---")
            lines.append(chunk.content)

        return "\n".join(lines)

    def as_retriever_tool(self):
        """LangChain @tool 형식의 검색 도구를 반환한다."""
        from langchain_core.tools import tool

        pipeline = self

        @tool
        def rag_search(query: str, k: int = 4) -> str:
            """내부 문서 저장소에서 쿼리와 관련된 청크를 검색한다."""
            chunks = pipeline.retrieve(query, k)
            if not chunks:
                return "[RAG] 관련 문서를 찾을 수 없습니다."
            results = []
            for chunk in chunks:
                source = chunk.metadata.get("source", "?")
                results.append(f"[{source}] {chunk.content}")
            return "\n\n".join(results)

        return rag_search


# ══════════════════════════════════════════════════════════════════
#  ConversationMemory (Path A: Buffer Memory)
# ══════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    세션 내 대화 기록을 관리하는 Buffer Memory.
    다이어그램의 Path A: ConversationBuffer + Memory.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_context(self) -> str:
        """최근 대화를 텍스트로 반환한다."""
        if not self._history:
            return ""
        lines = ["[대화 히스토리]"]
        for turn in self._history[-10:]:
            role = "사용자" if turn["role"] == "user" else "에이전트"
            lines.append(f"{role}: {turn['content'][:200]}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()
