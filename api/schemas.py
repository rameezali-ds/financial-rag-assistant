from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000, example="What were Apple's main risk factors in fiscal year 2023?")
    tickers: list[str] | None = Field(None, example=["AAPL", "MSFT"])
    years: list[int] | None = Field(None, example=[2023])


class CitationModel(BaseModel):
    index: int
    source: str
    ticker: str
    year: int
    filing_type: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    confidence: float = Field(ge=0.0, le=1.0)
    intent: str
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    retriever_ready: bool
