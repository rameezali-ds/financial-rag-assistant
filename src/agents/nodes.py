"""
LangGraph node implementations for the FinSight RAG agent.

"""
from __future__ import annotations

import time
import logging
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.agents.state import AgentState, Document
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompts import (
    PLANNER_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# Shared LLM clients — lazy singletons, created on first use

_fast_llm: ChatGroq | None = None
_strong_llm: ChatGroq | None = None


def _get_fast_llm() -> ChatGroq:
    global _fast_llm
    if _fast_llm is None:
        _fast_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    return _fast_llm


def _get_strong_llm() -> ChatGroq:
    global _strong_llm
    if _strong_llm is None:
        _strong_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    return _strong_llm


# Node: Planner

def planner_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()

    response = _get_fast_llm().invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {state['query']}"),
    ])

    import json
    try:
        parsed = json.loads(response.content)
        intent = parsed.get("intent", "fact_lookup")
        sub_queries = parsed.get("sub_queries", [state["query"]])
    except json.JSONDecodeError:
        logger.warning("Planner returned non-JSON; using raw query as sub-query")
        intent = "fact_lookup"
        sub_queries = [state["query"]]

    latency = (time.perf_counter() - t0) * 1000
    return {
        "query_intent": intent,
        "sub_queries": sub_queries,
        "retrieval_attempts": 0,
        "latency_ms": {"planner": latency},
    }


# Node: Retriever

def retriever_node(state: AgentState, retriever: HybridRetriever) -> dict[str, Any]:
    t0 = time.perf_counter()

    all_docs: list[Document] = []
    for sub_query in state["sub_queries"]:
        docs = retriever.retrieve(sub_query, top_k=10)
        all_docs.extend(docs)

    seen: set[str] = set()
    unique_docs: list[Document] = []
    for doc in all_docs:
        key = hash(doc["content"][:200])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    reranked = retriever.rerank(state["query"], unique_docs, top_n=6)

    latency = (time.perf_counter() - t0) * 1000
    return {
        "retrieved_docs": unique_docs,
        "reranked_docs": reranked,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "latency_ms": {**state.get("latency_ms", {}), "retriever": latency},
    }


# Node: Critic

def critic_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()

    graded_docs: list[Document] = []
    for doc in state["reranked_docs"]:
        grade_prompt = (
            f"Query: {state['query']}\n\n"
            f"Document excerpt:\n{doc['content'][:500]}\n\n"
            "Grade this document: relevant | partially_relevant | irrelevant\n"
            "Respond with ONLY the grade word."
        )
        response = _get_fast_llm().invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=grade_prompt),
        ])
        grade = response.content.strip().lower()
        if grade not in {"relevant", "partially_relevant", "irrelevant"}:
            grade = "partially_relevant"

        graded_docs.append({**doc, "grade": grade})

    relevant = [d for d in graded_docs if d["grade"] != "irrelevant"]
    needs_fallback = len(relevant) < 2

    latency = (time.perf_counter() - t0) * 1000
    return {
        "reranked_docs": graded_docs,
        "needs_web_fallback": needs_fallback,
        "critique": f"Found {len(relevant)}/{len(graded_docs)} relevant documents.",
        "latency_ms": {**state.get("latency_ms", {}), "critic": latency},
    }


# Node: Web Search Fallback
def web_search_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        results = search.invoke(state["query"])
        snippets = [r["content"] for r in results if "content" in r]
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        snippets = []

    latency = (time.perf_counter() - t0) * 1000
    return {
        "web_search_results": snippets,
        "latency_ms": {**state.get("latency_ms", {}), "web_search": latency},
    }


# Node: Synthesizer

def synthesizer_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()

    context_parts = []
    relevant_docs = [d for d in state["reranked_docs"] if d["grade"] != "irrelevant"]

    for i, doc in enumerate(relevant_docs, 1):
        context_parts.append(
            f"[Source {i}] {doc['ticker']} {doc['filing_type']} ({doc['year']}) "
            f"— {doc['section']}\n{doc['content']}"
        )

    for i, snippet in enumerate(state.get("web_search_results", []), len(relevant_docs) + 1):
        context_parts.append(f"[Source {i}] Web Search Result\n{snippet}")

    context = "\n\n---\n\n".join(context_parts)
    total_context_chars = len(context)

    response = _get_strong_llm().invoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Question: {state['query']}\n\n"
            f"Context:\n{context}\n\n"
            "Provide a comprehensive, cited answer."
        )),
    ])

    import re
    cited_indices = [int(m) for m in re.findall(r"\[Source (\d+)\]", response.content)]
    citations = []
    for idx in set(cited_indices):
        if idx <= len(relevant_docs):
            doc = relevant_docs[idx - 1]
            citations.append({
                "index": idx,
                "source": doc["source"],
                "ticker": doc["ticker"],
                "year": doc["year"],
                "filing_type": doc["filing_type"],
            })

    sentences = response.content.split(".")
    cited_sentences = sum(1 for s in sentences if "[Source" in s)
    confidence = min(cited_sentences / max(len(sentences), 1), 1.0)

    latency = (time.perf_counter() - t0) * 1000
    return {
        "answer": response.content,
        "citations": citations,
        "confidence": round(confidence, 2),
        "is_answer_grounded": confidence > 0.3,
        "token_usage": {"context_chars": total_context_chars},
        "latency_ms": {**state.get("latency_ms", {}), "synthesizer": latency},
    }
