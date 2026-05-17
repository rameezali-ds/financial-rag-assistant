"""
Document parser for SEC EDGAR filings.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_sec_filing(file_path: str | Path) -> str:
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in {".htm", ".html"}:
        return _parse_html(file_path)
    elif suffix == ".txt":
        return _parse_txt(file_path)
    elif suffix == ".pdf":
        return _parse_pdf(file_path)
    else:
        logger.warning(f"Unknown file type {suffix}, attempting plain text read")
        return file_path.read_text(encoding="utf-8", errors="ignore")


def _parse_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")

    text = soup.get_text(separator="\n")
    return _clean_text(text)


def _parse_txt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")

    raw = re.sub(r"<SEC-HEADER>.*?</SEC-HEADER>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", " ", raw)

    return _clean_text(raw)


def _parse_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return _clean_text("\n\n".join(pages))


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def infer_metadata_from_path(file_path: Path) -> dict:
    parts = file_path.parts
    metadata = {"ticker": "UNKNOWN", "filing_type": "UNKNOWN", "year": 0}

    for i, part in enumerate(parts):
        if part == "sec-edgar-filings" and i + 2 < len(parts):
            metadata["ticker"] = parts[i + 1].upper()
            metadata["filing_type"] = parts[i + 2]
            break

    year_match = re.search(r"20(1[5-9]|2[0-9])", str(file_path))
    if year_match:
        metadata["year"] = int(year_match.group())

    return metadata
