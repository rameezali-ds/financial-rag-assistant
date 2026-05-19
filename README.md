# FinSight RAG — Agentic Financial Research Assistant

**Agentic RAG system** for financial document analysis — SEC 10-K filings, earnings call transcripts, and financial news. Built with LangGraph, ChromaDB, hybrid search (BM25 + dense retrieval), cross-encoder reranking, and a full evaluation pipeline using RAGAs.

---

## What This Project Demonstrates

This project showcases a **complete, production-ready RAG pipeline**:

| Skill Area | Implementation |
|---|---|
| **Agentic RAG** | LangGraph multi-agent graph with planner, retriever, critic, and synthesizer nodes |
| **Hybrid Search** | BM25 + dense embeddings with Reciprocal Rank Fusion (RRF) |
| **Advanced Chunking** | Semantic chunking with `sentence-transformers`, not naive fixed-size splitting |
| **Cross-encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision after retrieval |
| **Evaluation Pipeline** | RAGAs metrics: faithfulness, answer relevancy, context precision/recall |
| **Baseline Comparison** | Naive RAG vs Agentic RAG side-by-side |
| **Vector Store** | ChromaDB with `BAAI/bge-large-en-v1.5` embeddings |
| **CI/CD + Eval Gate** | GitHub Actions: lint → test → RAGAs quality gate → Docker build |
| **API + UI** | FastAPI with streaming SSE endpoints + interactive Streamlit UI |
| **Free Data** | SEC EDGAR 10-K filings — no paid subscriptions needed |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT GRAPH                         │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐   │
│  │ Planner  │──▶│Retriever │──▶│  Critic  │──▶│ Synthesizer │   │
│  │ (intent  │   │(hybrid   │   │(grade    │   │ (Llama3 70b │   │
│  │ routing) │   │ search)  │   │ docs)    │   │  + cites)   │   │
│  └──────────┘   └──────────┘   └────┬─────┘   └─────────────┘   │
│                      ▲         retry│if poor docs                │
│                      └─────────────┘                            │
│                                     │                            │
│                               ┌─────▼──────┐                    │
│                               │ Web Search │ (fallback)          │
│                               └────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐   │
│  │  ChromaDB   │   │ BM25 Index  │   │   SEC EDGAR Filings  │   │
│  │  (vectors)  │   │  (sparse)   │   │  10-K, 10-Q, 8-K     │   │
│  └─────────────┘   └─────────────┘   └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Free [Groq API key](https://console.groq.com)

### Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=gsk_...
```

### Ingest Documents

```bash
# Quick start — built-in sample data (AAPL, MSFT, NVDA)
python scripts/ingest_pipeline.py --use-sample-data

# Full pipeline — real SEC EDGAR 10-K filings
python scripts/ingest_pipeline.py --tickers AAPL MSFT NVDA GOOGL --years 2022 2023 2024 --email your@email.com
```

### Run the App

```bash
# Interactive Streamlit UI
streamlit run app/streamlit_app.py
# → opens at http://localhost:8501

# FastAPI REST server (with streaming)
uvicorn api.main:app --reload
# → docs at http://localhost:8000/docs
```

---

## Evaluation

This project includes a two-tier evaluation setup — not just a single score, but a **baseline comparison** that proves the agentic approach outperforms naive RAG.

### How it works

**Tier 1 — Single pipeline evaluation:**
Runs the agentic pipeline on financial Q&A pairs and scores with RAGAs.

```bash
python scripts/run_evaluation.py --output results/eval_report.csv
```

**Tier 2 — Baseline comparison:**
Runs both a naive pipeline (simple retrieve → answer) and the full agentic pipeline, then prints them side by side.

```bash
python scripts/run_evaluation.py --compare
```

The naive pipeline is a simple single-step RAG: retrieve top-5 chunks → send to LLM → return answer. No agent loop, no critic, no reranking. The comparison proves that the additional complexity is actually worth it.

### Results

Evaluated on financial Q&A pairs (AAPL, MSFT, NVDA sample corpus) using RAGAs with Groq Llama-3.1-8b:

**Agentic pipeline scores:**

| Metric | Score | What It Measures |
|---|---|---|
| **Context Recall** | 0.750 | Did retrieval find all necessary information? |
| **Context Precision** | 0.583 | Were retrieved chunks actually useful? |
| **Answer Relevancy** | 0.474 | Does the answer address the question? |
| **Faithfulness** | 0.454 | Is the answer grounded in retrieved context? |

**Naive RAG vs FinSight RAG comparison:**

| Metric | Naive RAG | FinSight RAG | Improvement |
|---|---|---|---|
| Answer Relevancy | 0.397 | **0.474** | **+20%** |
| Context Recall | 0.667 | **0.750** | **+13%** |
| Context Precision | 0.583 | 0.583 | — |

Full per-question breakdown: [results/eval_report.csv](results/eval_report.csv)

---

## Data Sources

| Source | How to Access | Used For |
|---|---|---|
| **SEC EDGAR** | `sec-edgar-downloader` library | 10-K, 10-Q, 8-K annual/quarterly filings |
| **FinanceBench** | [HuggingFace](https://huggingface.co/datasets/PatronusAI/financebench) | 150 Q&A evaluation benchmark pairs |
| **HuggingFace Datasets** | `datasets` library | `gbharti/finance-alpaca` for testing |

---

## Project Structure

```
financial-rag-assistant/
├── src/
│   ├── agents/
│   │   ├── graph.py            # LangGraph agent graph definition
│   │   ├── nodes.py            # Planner, retriever, critic, synthesizer nodes
│   │   └── state.py            # Typed AgentState schema
│   ├── retrieval/
│   │   └── hybrid_retriever.py # BM25 + dense + RRF fusion + cross-encoder reranking
│   ├── ingestion/
│   │   ├── sec_downloader.py   # Free SEC EDGAR downloader
│   │   ├── document_parser.py  # HTML/PDF parser
│   │   ├── chunker.py          # Semantic chunker (section-aware)
│   │   └── embedder.py         # Batch embedder with ChromaDB storage
│   ├── generation/
│   │   └── prompts.py          # Prompt templates for all agent nodes
│   └── evaluation/
│       └── ragas_eval.py       # RAGAs evaluation suite (Groq-powered, free)
├── api/
│   ├── main.py                 # FastAPI app with streaming SSE endpoint
│   └── schemas.py              # Pydantic request/response models
├── app/
│   └── streamlit_app.py        # Interactive Streamlit UI
├── scripts/
│   ├── ingest_pipeline.py      # End-to-end ingestion script
│   └── run_evaluation.py       # Evaluation runner (--compare for baseline)
├── tests/
│   ├── test_chunker.py         # Unit tests for semantic chunker
│   └── test_retrieval.py       # Unit tests for RRF fusion logic
├── results/
│   └── eval_report.csv         # RAGAs evaluation output (per-question scores)
├── .github/workflows/
│   └── ci.yml                  # CI/CD: lint → test → eval gate → Docker
├── notebooks/
│   └── 01_data_exploration.ipynb
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Tech Stack

| Category | Tool |
|---|---|
| **Agent Orchestration** | LangChain, LangGraph |
| **LLM** | Groq (Llama-3.3-70b-versatile, Llama-3.1-8b-instant) — free tier |
| **Embeddings** | `BAAI/bge-large-en-v1.5` via sentence-transformers (local) |
| **Vector Store** | ChromaDB (local persistence) |
| **Sparse Retrieval** | rank-bm25 |
| **Reranking** | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| **Evaluation** | RAGAs |
| **API** | FastAPI + SSE streaming |
| **UI** | Streamlit |
| **Data** | sec-edgar-downloader (free SEC filings) |
| **DevOps** | Docker, GitHub Actions |

---

## Roadmap

- [ ] Add multi-modal support (parse charts/tables from PDFs with `unstructured`)
- [ ] Fine-tune embedding model on financial domain with `sentence-transformers`
- [ ] GraphRAG for company relationship networks
- [ ] Real-time ingestion from SEC EDGAR RSS feed (new 8-K filings as they drop)
- [ ] Multi-tenant ChromaDB with namespace isolation
