"""Tests for retrieval components — focuses on RRF fusion logic (no vector DB needed)."""
import pytest
from src.agents.state import Document
from src.retrieval.hybrid_retriever import HybridRetriever


def make_doc(content: str, score: float = 0.5) -> Document:
    return Document(
        content=content,
        source="test.txt",
        ticker="TEST",
        filing_type="10-K",
        year=2023,
        section="Test",
        relevance_score=score,
        grade="",
    )


class TestReciprocalRankFusion:
    def test_rrf_merges_two_lists(self):
        list1 = [(make_doc("doc A"), 0), (make_doc("doc B"), 1), (make_doc("doc C"), 2)]
        list2 = [(make_doc("doc B"), 0), (make_doc("doc C"), 1), (make_doc("doc D"), 2)]

        merged = HybridRetriever._reciprocal_rank_fusion(list1, list2, k=60)
        assert len(merged) > 0
        contents = [d["content"] for d in merged]
        assert "doc B" in contents
        assert "doc C" in contents

    def test_rrf_higher_rank_scores_higher(self):
        top_doc = make_doc("top doc")
        bottom_doc = make_doc("bottom doc")

        list1 = [(top_doc, 0), (bottom_doc, 5)]
        list2 = [(top_doc, 0), (bottom_doc, 5)]

        merged = HybridRetriever._reciprocal_rank_fusion(list1, list2, k=60)
        assert merged[0]["content"] == "top doc"

    def test_rrf_deduplicates(self):
        shared_doc = make_doc("shared")
        list1 = [(shared_doc, 0), (make_doc("other A"), 1)]
        list2 = [(shared_doc, 0), (make_doc("other B"), 1)]

        merged = HybridRetriever._reciprocal_rank_fusion(list1, list2, k=60)
        shared_count = sum(1 for d in merged if d["content"] == "shared")
        assert shared_count == 1

    def test_rrf_single_list(self):
        ranked = [(make_doc(f"doc {i}"), i) for i in range(5)]
        merged = HybridRetriever._reciprocal_rank_fusion(ranked, k=60)
        assert len(merged) == 5

    def test_rrf_empty_lists(self):
        merged = HybridRetriever._reciprocal_rank_fusion([], [], k=60)
        assert merged == []
