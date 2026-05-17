"""
Agent state schema for the FinSight RAG graph.

"""
from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


class Document(TypedDict):
    content: str
    source: str
    ticker: str
    filing_type: str
    year: int
    section: str
    relevance_score: float
    grade: str


class AgentState(TypedDict):
    query: str
    query_intent: str
    sub_queries: list[str]

    retrieved_docs: Annotated[list[Document], operator.add]
    reranked_docs: list[Document]
    web_search_results: Annotated[list[str], operator.add]

    answer: str
    citations: list[dict[str, Any]]
    confidence: float

    retrieval_attempts: int
    needs_web_fallback: bool
    critique: str
    is_answer_grounded: bool

    latency_ms: dict[str, float]
    token_usage: dict[str, int]
