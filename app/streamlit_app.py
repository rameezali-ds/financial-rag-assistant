"""
FinSight RAG — Streamlit interactive UI.

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Make src importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="FinSight RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar — configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.divider()
    st.subheader("Data Filters")
    selected_tickers = st.multiselect(
        "Filter by Ticker",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ"],
        default=[],
    )
    selected_years = st.multiselect(
        "Filter by Year",
        [2022, 2023, 2024],
        default=[2023, 2024],
    )

    st.divider()
    st.subheader("Retrieval Settings")
    top_k = st.slider("Retrieval top-k", min_value=3, max_value=20, value=10)
    rerank_n = st.slider("Rerank top-n", min_value=2, max_value=10, value=5)

    st.divider()
    st.markdown("**About**")
    st.markdown("""
    FinSight RAG uses:
    - 🔍 Hybrid BM25 + Dense retrieval
    - 🔄 Cross-encoder reranking
    - 🤖 LangGraph agentic pipeline
    - 📋 RAGAs evaluation
    """)

# Main area
st.title("📊 FinSight — Financial Research Assistant")
st.caption("Ask questions about SEC 10-K filings, earnings calls, and financial reports.")

# Example questions
with st.expander("💡 Example Questions", expanded=False):
    examples = [
        "What were Apple's main risk factors in fiscal year 2023?",
        "How did Microsoft's cloud revenue grow from 2022 to 2023?",
        "Compare NVIDIA and AMD's R&D spending in 2023.",
        "What did Apple say about supply chain risks after COVID-19?",
        "What is JPMorgan's exposure to commercial real estate?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["query_input"] = ex

# Query input
query = st.text_area(
    "Your question",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="Ask anything about the indexed financial documents...",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
run_btn = col1.button("🔍 Analyze", type="primary", use_container_width=True)
clear_btn = col2.button("Clear", use_container_width=False)

if clear_btn:
    st.session_state["query_input"] = ""
    st.rerun()

# Run pipeline
if run_btn and query.strip():
    if not groq_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    with st.spinner("🤖 Analyzing financial documents..."):
        try:
            from src.retrieval.hybrid_retriever import HybridRetriever
            from src.agents.graph import run_query

            if "retriever" not in st.session_state:
                with st.spinner("Loading vector store..."):
                    st.session_state["retriever"] = HybridRetriever.from_config(
                        persist_dir="./data/chroma"
                    )

            retriever = st.session_state["retriever"]
            t0 = time.perf_counter()
            result = run_query(query, retriever)
            elapsed = time.perf_counter() - t0

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.info("Make sure you've run `python scripts/ingest_pipeline.py` first.")
            st.stop()

    # Display results
    st.divider()

    # Metadata row
    meta_cols = st.columns(4)
    meta_cols[0].metric("Confidence", f"{result['confidence']:.0%}")
    meta_cols[1].metric("Intent", result["query_intent"].replace("_", " ").title())
    meta_cols[2].metric("Total Latency", f"{elapsed:.1f}s")
    meta_cols[3].metric("Sources Cited", len(result["citations"]))

    # Answer
    st.subheader("📝 Answer")
    st.markdown(result["answer"])

    # Citations
    if result["citations"]:
        st.subheader("📚 Sources")
        for citation in result["citations"]:
            st.markdown(
                f"**[Source {citation['index']}]** "
                f"`{citation['ticker']}` · {citation['filing_type']} · {citation['year']}"
            )

    # Debug expander
    with st.expander("🔧 Debug: Retrieved Documents", expanded=False):
        for i, doc in enumerate(result.get("reranked_docs", []), 1):
            grade_emoji = {"relevant": "✅", "partially_relevant": "🟡", "irrelevant": "❌"}.get(
                doc.get("grade", ""), "❓"
            )
            st.markdown(f"**Doc {i}** {grade_emoji} `{doc['ticker']}` {doc['filing_type']} ({doc['year']}) — {doc['section']}")
            st.markdown(f"> {doc['content'][:300]}...")
            st.divider()

    with st.expander("⏱️ Latency Breakdown", expanded=False):
        latencies = result.get("latency_ms", {})
        if latencies:
            import pandas as pd
            df = pd.DataFrame(
                [(k, f"{v:.0f}ms") for k, v in latencies.items()],
                columns=["Node", "Latency"],
            )
            st.dataframe(df, use_container_width=True)

elif run_btn:
    st.warning("Please enter a question.")
