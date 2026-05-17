"""
SEC EDGAR document downloader.

Uses the free SEC EDGAR EDGAR full-text search API and the
`sec-edgar-downloader` library to fetch 10-K, 10-Q, and 8-K filings.

Data is 100% free and public — no API key required.
EDGAR Terms of Service: https://efts.sec.gov/LATEST/search-index/tos

Rate limit: 10 requests/second (enforced automatically).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

from sec_edgar_downloader import Downloader
from tqdm import tqdm

logger = logging.getLogger(__name__)

FilingType = Literal["10-K", "10-Q", "8-K"]

DEFAULT_TICKERS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "META",   # Meta
    "NVDA",   # NVIDIA
    "JPM",    # JPMorgan Chase
    "JNJ",    # Johnson & Johnson
]


def download_filings(
    tickers: list[str],
    filing_types: list[FilingType],
    output_dir: str | Path,
    years: list[int] | None = None,
    email: str = "research@example.com",
) -> dict[str, list[Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dl = Downloader("FinSight Research", email, str(output_dir))

    downloaded: dict[str, list[Path]] = {}
    years = years or [2022, 2023, 2024]

    for ticker in tqdm(tickers, desc="Downloading filings"):
        downloaded[ticker] = []
        for filing_type in filing_types:
            try:
                dl.get(
                    filing_type,
                    ticker,
                    after=f"{min(years)}-01-01",
                    before=f"{max(years)}-12-31",
                    download_details=True,
                )
                filing_dir = output_dir / "sec-edgar-filings" / ticker / filing_type
                if filing_dir.exists():
                    paths = list(filing_dir.rglob("*.htm")) + list(filing_dir.rglob("*.html"))
                    downloaded[ticker].extend(paths)
                    logger.info(f"Downloaded {len(paths)} {filing_type} files for {ticker}")

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to download {filing_type} for {ticker}: {e}")

    total = sum(len(v) for v in downloaded.values())
    logger.info(f"Download complete: {total} files across {len(tickers)} tickers")
    return downloaded


def get_sample_data_instructions() -> str:
    return """
    ============================================================
    FREE DATA SOURCES FOR THIS PROJECT
    ============================================================

    1. SEC EDGAR (Primary — No API key needed)
       pip install sec-edgar-downloader
       from sec_edgar_downloader import Downloader
       dl = Downloader("YourCompany", "your@email.com")
       dl.get("10-K", "AAPL", after="2020-01-01")

    2. FinanceBench Evaluation Dataset (150 Q&A pairs from real 10-Ks)
       pip install datasets
       from datasets import load_dataset
       ds = load_dataset("PatronusAI/financebench")

    3. Hugging Face Finance Dataset
       ds = load_dataset("gbharti/finance-alpaca")

    4. Alternative: Download 10-K PDFs directly from company IR pages:
       - Apple:     https://investor.apple.com/sec-filings/annual-reports
       - Microsoft: https://www.microsoft.com/en-us/Investor/earnings/FY-2024-Q4/
       - NVIDIA:    https://investor.nvidia.com/financial-info/sec-filings/
    ============================================================
    """
