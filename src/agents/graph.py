"""
FinSight RAG — LangGraph agent graph definition.

"""
from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.nodes import (
    planner_node,
    retriever_node,
    critic_node,
    web_search_node,
    synthesizer_node,
)
from src.retrieval.hybrid_retriever import HybridRetriever


def _route_after_critic(
    state: AgentState,
) -> Literal["synthesizer", "web_search", "retriever"]:
    if not state.get("needs_web_fallback", False):
        return "synthesizer"

    if state.get("retrieval_attempts", 0) < 2:
        return "retriever"

    return "web_search"


def build_graph(retriever: HybridRetriever) -> StateGraph:
    graph = StateGraph(AgentState)

    retriever_with_dep = partial(retriever_node, retriever=retriever)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_with_dep)
    graph.add_node("critic", critic_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_edge("web_search", "synthesizer")
    graph.add_edge("synthesizer", END)

    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {
            "synthesizer": "synthesizer",
            "retriever": "retriever",
            "web_search": "web_search",
        },
    )

    return graph.compile()


def run_query(query: str, retriever: HybridRetriever) -> AgentState:
    graph = build_graph(retriever)
    initial_state: AgentState = {
        "query": query,
        "query_intent": "",
        "sub_queries": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "web_search_results": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "retrieval_attempts": 0,
        "needs_web_fallback": False,
        "critique": "",
        "is_answer_grounded": False,
        "latency_ms": {},
        "token_usage": {},
    }
    return graph.invoke(initial_state)
