from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.schemas import QueryRequest, QueryResponse, HealthResponse
from src.agents.graph import build_graph, run_query
from src.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Prometheus metrics
QUERY_COUNTER = Counter("finsight_queries_total", "Total queries processed", ["status"])
QUERY_LATENCY = Histogram(
    "finsight_query_latency_seconds",
    "Query latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# App lifecycle
_retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _retriever
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    logger.info(f"Loading retriever from {persist_dir}...")
    _retriever = HybridRetriever.from_config(persist_dir=persist_dir)
    logger.info("Retriever ready.")
    yield
    logger.info("Shutting down FinSight API.")


app = FastAPI(
    title="FinSight RAG API",
    description="Agentic RAG for financial document analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    t0 = time.perf_counter()
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_query, request.query, _retriever
        )
        QUERY_COUNTER.labels(status="success").inc()
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            confidence=result["confidence"],
            intent=result["query_intent"],
            latency_ms=result["latency_ms"],
        )
    except Exception as e:
        QUERY_COUNTER.labels(status="error").inc()
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        QUERY_LATENCY.observe(time.perf_counter() - t0)


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    async def event_generator():
        graph = build_graph(_retriever)
        initial_state = {
            "query": request.query,
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
        async for chunk in graph.astream(initial_state):
            node_name = list(chunk.keys())[0]
            yield f"data: {{\"node\": \"{node_name}\", \"status\": \"complete\"}}\n\n"
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    doc_count = 0
    if _retriever:
        try:
            collection = _retriever._get_collection()
            doc_count = collection.count()
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if _retriever else "degraded",
        documents_indexed=doc_count,
        retriever_ready=_retriever is not None,
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
