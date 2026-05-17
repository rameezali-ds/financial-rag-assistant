"""Unit tests for FinSight RAG components."""
import pytest
from src.ingestion.chunker import semantic_chunk, Chunk
from src.ingestion.document_parser import _clean_text


class TestSemanticChunker:
    SAMPLE_TEXT = """
ITEM 1. BUSINESS

Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, and wearables.
The Company sells its products worldwide through its retail stores, online stores, and direct sales force.

The Company's products and services include iPhone, Mac, iPad, Apple Watch, and various subscription services.

ITEM 1A. RISK FACTORS

The Company faces intense competition in all its markets.
Uncertainty about global economic conditions poses a risk to consumer spending.
The Company is exposed to foreign currency exchange rate fluctuations.

The Company's supply chain is concentrated in Asia, which creates geographic concentration risk.
Natural disasters, political instability, or labor disputes could disrupt production.
    """

    def test_returns_list_of_chunks(self):
        chunks = semantic_chunk(
            text=self.SAMPLE_TEXT,
            source="test.txt",
            ticker="AAPL",
            filing_type="10-K",
            year=2023,
        )
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata_populated(self):
        chunks = semantic_chunk(
            text=self.SAMPLE_TEXT,
            source="test/AAPL_10K.txt",
            ticker="AAPL",
            filing_type="10-K",
            year=2023,
        )
        for chunk in chunks:
            assert chunk.ticker == "AAPL"
            assert chunk.filing_type == "10-K"
            assert chunk.year == 2023
            assert len(chunk.content) > 0

    def test_max_tokens_respected(self):
        chunks = semantic_chunk(
            text=self.SAMPLE_TEXT,
            source="test.txt",
            ticker="AAPL",
            filing_type="10-K",
            year=2023,
            max_tokens=100,
        )
        for chunk in chunks:
            assert len(chunk.content) <= 100 * 4 * 1.2

    def test_section_detection(self):
        chunks = semantic_chunk(
            text=self.SAMPLE_TEXT,
            source="test.txt",
            ticker="AAPL",
            filing_type="10-K",
            year=2023,
        )
        sections = {c.section for c in chunks}
        assert len(sections) >= 1

    def test_empty_text_returns_empty(self):
        chunks = semantic_chunk(
            text="   \n\n  ",
            source="test.txt",
            ticker="AAPL",
            filing_type="10-K",
            year=2023,
        )
        assert chunks == []

    def test_chunk_index_sequential(self):
        chunks = semantic_chunk(
            text=self.SAMPLE_TEXT * 5,
            source="test.txt",
            ticker="AAPL",
            filing_type="10-K",
            year=2023,
            max_tokens=50,
        )
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(indices)))


class TestDocumentCleaner:
    def test_collapses_whitespace(self):
        text = "Hello    World\t\tTest"
        cleaned = _clean_text(text)
        assert "    " not in cleaned
        assert "\t" not in cleaned

    def test_preserves_paragraph_breaks(self):
        text = "Para one.\n\nPara two."
        cleaned = _clean_text(text)
        assert "\n\n" in cleaned

    def test_removes_page_numbers(self):
        text = "Some content.\n\n42\n\nMore content."
        cleaned = _clean_text(text)
        assert "\n42\n" not in cleaned

    def test_handles_empty_string(self):
        assert _clean_text("") == ""
