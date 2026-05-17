"""
Semantic chunking for financial documents.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

SEC_SECTION_PATTERNS = [
    r"item\s+1[a-z]?\.\s+(business|risk factors|unresolved staff comments)",
    r"item\s+2\.\s+properties",
    r"item\s+3\.\s+legal proceedings",
    r"item\s+7[a-z]?\.\s+(management.s discussion|quantitative)",
    r"item\s+8\.\s+financial statements",
    r"item\s+9[a-z]?\.\s+(changes in|controls and procedures)",
]
_SECTION_RE = re.compile("|".join(SEC_SECTION_PATTERNS), re.IGNORECASE)


@dataclass
class Chunk:
    content: str
    source: str
    ticker: str
    filing_type: str
    year: int
    section: str
    chunk_index: int
    char_count: int


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"(U\.S\.|Inc\.|Corp\.|Ltd\.|et al\.|vs\.)", lambda m: m.group().replace(".", "<!DOT!>"), text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.replace("<!DOT!>", ".") for s in sentences if s.strip()]


def semantic_chunk(
    text: str,
    source: str,
    ticker: str,
    filing_type: str,
    year: int,
    max_tokens: int = 512,
    overlap_sentences: int = 2,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    current_section = "General"

    paragraphs = re.split(r"\n{2,}", text.strip())

    current_chunk_sentences: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if _SECTION_RE.search(para) and len(para) < 200:
            if current_chunk_sentences:
                content = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    content=content,
                    source=source,
                    ticker=ticker,
                    filing_type=filing_type,
                    year=year,
                    section=current_section,
                    chunk_index=chunk_index,
                    char_count=len(content),
                ))
                chunk_index += 1
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_tokens = sum(_estimate_tokens(s) for s in current_chunk_sentences)

            current_section = para[:100]
            continue

        sentences = _split_into_sentences(para)

        for sentence in sentences:
            sentence_tokens = _estimate_tokens(sentence)

            if current_tokens + sentence_tokens > max_tokens and current_chunk_sentences:
                content = " ".join(current_chunk_sentences)
                chunks.append(Chunk(
                    content=content,
                    source=source,
                    ticker=ticker,
                    filing_type=filing_type,
                    year=year,
                    section=current_section,
                    chunk_index=chunk_index,
                    char_count=len(content),
                ))
                chunk_index += 1
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_tokens = sum(_estimate_tokens(s) for s in current_chunk_sentences)

            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk_sentences:
        content = " ".join(current_chunk_sentences)
        chunks.append(Chunk(
            content=content,
            source=source,
            ticker=ticker,
            filing_type=filing_type,
            year=year,
            section=current_section,
            chunk_index=chunk_index,
            char_count=len(content),
        ))

    logger.info(f"Chunked {source} → {len(chunks)} chunks (avg {sum(c.char_count for c in chunks) // max(len(chunks), 1)} chars)")
    return chunks
