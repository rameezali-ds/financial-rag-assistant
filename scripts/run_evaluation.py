"""
Run the RAGAs evaluation suite and output a report.

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()  # Must be before any LLM imports
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


def load_financebench(n_samples: int = 50) -> list[dict]:
    """
    Load a subset of FinanceBench from HuggingFace.
    100% free, public benchmark — 150 Q&A pairs from real 10-K filings.

    Citation: Islam et al., "FinanceBench: A New Benchmark for Financial Question Answering"
    HuggingFace: https://huggingface.co/datasets/PatronusAI/financebench
    """
    try:
        from datasets import load_dataset
        console.print("[blue]Loading FinanceBench from HuggingFace...[/blue]")
        ds = load_dataset("PatronusAI/financebench", split="train")
        samples = []
        for row in list(ds)[:n_samples]:
            samples.append({
                "question": row["question"],
                "ground_truth": row["answer"],
            })
        console.print(f"Loaded [bold]{len(samples)}[/bold] samples from FinanceBench")
        return samples
    except Exception as e:
        logger.error(f"Failed to load FinanceBench: {e}")
        console.print("[yellow]Using built-in mini eval set instead[/yellow]")
        return _get_builtin_eval_set()


def _get_builtin_eval_set() -> list[dict]:
    return [
        {
            "question": "What was Apple's total net sales in fiscal year 2023?",
            "ground_truth": "Apple's total net sales were $383.3 billion in fiscal year 2023.",
        },
        {
            "question": "What was Microsoft's Intelligent Cloud revenue in fiscal 2023?",
            "ground_truth": "Microsoft's Intelligent Cloud revenue was $87.9 billion in fiscal year 2023.",
        },
        {
            "question": "What was NVIDIA's revenue growth rate in fiscal year 2024?",
            "ground_truth": "NVIDIA's revenue increased 122% to $60.9 billion in fiscal year 2024.",
        },
        {
            "question": "What were the main risk factors Apple identified in 2023?",
            "ground_truth": "Apple identified global economic uncertainty, intense market competition, and supply chain risks as main risk factors in 2023.",
        },
        {
            "question": "What was Microsoft's net income in fiscal 2023?",
            "ground_truth": "Microsoft's net income was $72.4 billion in fiscal year 2023, up 20% year-over-year.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAs evaluation")
    parser.add_argument("--dataset", help="Path to custom eval dataset JSON")
    parser.add_argument("--use-financebench", action="store_true")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--output", default="results/eval_report.csv")
    parser.add_argument("--compare", action="store_true", help="Compare naive vs agentic RAG")
    args = parser.parse_args()

    # Load eval set
    if args.use_financebench:
        eval_samples = load_financebench(n_samples=args.n_samples)
    elif args.dataset:
        with open(args.dataset) as f:
            eval_samples = json.load(f)
    else:
        console.print("[yellow]No dataset specified; using built-in mini eval set[/yellow]")
        eval_samples = _get_builtin_eval_set()

    # Load retriever
    console.print("[blue]Loading retriever...[/blue]")
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.agents.graph import run_query

    retriever = HybridRetriever.from_config(persist_dir="./data/chroma")

    def agentic_pipeline(question: str) -> dict:
        result = run_query(question, retriever)
        return {
            "answer": result["answer"],
            "contexts": [d["content"] for d in result.get("reranked_docs", [])],
        }

    from src.evaluation.ragas_eval import run_ragas_evaluation, compare_naive_vs_agentic

    if args.compare:
        from src.retrieval.hybrid_retriever import HybridRetriever
        from langchain_groq import ChatGroq

        naive_retriever = HybridRetriever.from_config(persist_dir="./data/chroma", load_reranker=False)
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        def naive_pipeline(question: str) -> dict:
            docs = naive_retriever.retrieve(question, top_k=5)
            context = "\n\n".join(d["content"] for d in docs)
            response = llm.invoke(f"Answer based on context:\n{context}\n\nQuestion: {question}")
            return {"answer": response.content, "contexts": [d["content"] for d in docs]}

        compare_naive_vs_agentic(eval_samples, naive_pipeline, agentic_pipeline)
    else:
        scores = run_ragas_evaluation(
            eval_samples,
            agentic_pipeline,
            output_path=args.output,
        )
        console.print(f"\n[bold green]Evaluation complete![/bold green]")
        console.print(f"Results saved to [cyan]{args.output}[/cyan]")


if __name__ == "__main__":
    main()
