"""
RAGAs evaluation pipeline for FinSight — configured to use Groq.

RAGAs metrics used:
  - faithfulness:        Is the answer supported by the retrieved context?
  - answer_relevancy:    Does the answer address the question?
  - context_precision:   Are retrieved chunks actually useful?
  - context_recall:      Did we retrieve all necessary information?
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

RAGAS_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _get_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    )


def _get_ragas_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )


def load_eval_dataset(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def run_ragas_evaluation(
    eval_samples: list[dict[str, Any]],
    rag_pipeline_fn,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    logger.info(f"Running RAGAs evaluation on {len(eval_samples)} samples...")

    ragas_llm = _get_ragas_llm()
    ragas_embeddings = _get_ragas_embeddings()

    for metric in RAGAS_METRICS:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_embeddings

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, sample in enumerate(eval_samples):
        question = sample["question"]
        logger.info(f"  [{i+1}/{len(eval_samples)}] {question[:80]}...")
        try:
            result = rag_pipeline_fn(question)
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(sample["ground_truth"])
        except Exception as e:
            logger.error(f"Pipeline failed on sample {i}: {e}")

    ragas_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        ragas_dataset,
        metrics=RAGAS_METRICS,
        raise_exceptions=False,
    )

    try:
        df = result.to_pandas()
        metric_cols = [c for c in df.columns if c in
                       ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
        scores = {col: float(df[col].mean()) for col in metric_cols}
    except Exception:
        scores = {}
        for metric in RAGAS_METRICS:
            name = metric.name if hasattr(metric, "name") else str(metric)
            try:
                scores[name] = float(result[name])
            except Exception:
                scores[name] = 0.0

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result.to_pandas().to_csv(output_path, index=False)
            logger.info(f"Detailed results saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save CSV: {e}")

    _print_results_table(scores, len(questions))
    return scores


def compare_naive_vs_agentic(
    eval_samples: list[dict[str, Any]],
    naive_pipeline_fn,
    agentic_pipeline_fn,
) -> pd.DataFrame:
    """Compare naive RAG vs FinSight Agentic RAG side-by-side."""
    logger.info("Evaluating naive RAG baseline...")
    naive_scores = run_ragas_evaluation(eval_samples, naive_pipeline_fn)

    logger.info("Evaluating agentic RAG (FinSight)...")
    agentic_scores = run_ragas_evaluation(eval_samples, agentic_pipeline_fn)

    rows = []
    for metric in naive_scores:
        naive = naive_scores[metric]
        agentic = agentic_scores.get(metric, 0.0)
        improvement = ((agentic - naive) / naive) * 100 if naive > 0 else 0
        rows.append({
            "Metric": metric,
            "Naive RAG": round(naive, 3),
            "FinSight RAG": round(agentic, 3),
            "Improvement": f"+{improvement:.0f}%" if improvement > 0 else f"{improvement:.0f}%",
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


def _print_results_table(scores: dict[str, float], n_samples: int) -> None:
    print(f"\n{'='*50}")
    print(f"RAGAs Evaluation Results (n={n_samples})")
    print(f"{'='*50}")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.3f}  {bar}")
    print(f"{'='*50}\n")
