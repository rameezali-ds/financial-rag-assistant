"""
End-to-end ingestion pipeline - Download → Parse → Chunk → Embed → Store.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track

from src.ingestion.sec_downloader import download_filings, DEFAULT_TICKERS
from src.ingestion.document_parser import parse_sec_filing, infer_metadata_from_path
from src.ingestion.chunker import semantic_chunk
from src.ingestion.embedder import embed_and_store

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_DOCUMENTS = [
    {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "year": 2023,
        "content": """
APPLE INC. ANNUAL REPORT ON FORM 10-K

ITEM 1. BUSINESS
Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52 or 53-week period that ends on the last Saturday of September.

ITEM 1A. RISK FACTORS
The Company's operations and performance depend significantly on worldwide economic conditions. Uncertainty about current global economic conditions poses a risk as consumers and businesses may postpone spending in response to tighter credit, negative financial news and declines in income or asset values.

The Company faces substantial competition in each of the markets it operates in. The markets for the Company's products and services are highly competitive and subject to rapid technological change.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Net sales for fiscal 2023 were $383.3 billion, a decrease of 3% compared to fiscal 2022. The decrease was driven primarily by lower iPhone and Mac net sales, partially offset by higher Services net sales.

iPhone net sales were $200.6 billion, representing 52% of total net sales. Services net sales were $85.2 billion, a new all-time record, representing 22% of total net sales.

The Company generated $111.4 billion in operating cash flow during fiscal 2023. The Company returned over $19 billion to shareholders during the fourth quarter of fiscal 2023 in the form of dividends and share repurchases.

Gross margin was 44.1% for fiscal 2023, compared to 43.3% for fiscal 2022.
        """,
    },
    {
        "ticker": "MSFT",
        "filing_type": "10-K",
        "year": 2023,
        "content": """
MICROSOFT CORPORATION ANNUAL REPORT ON FORM 10-K

ITEM 1. BUSINESS
Microsoft is a technology company. We develop and support software, services, devices and solutions. Our mission is to empower every person and organization on the planet to achieve more.

We have three segments: Productivity and Business Processes, Intelligent Cloud, and More Personal Computing.

ITEM 1A. RISK FACTORS
We face intense competition across all markets for our products and services, which may lead to lower revenue or operating margins. Our competitors range from large technology companies to innovative startups.

Cybersecurity threats are a significant risk. We face sophisticated threat actors who target our systems and the systems of our customers. Security incidents could damage our reputation and lead to financial losses.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Revenue increased 7% to $211.9 billion for fiscal year 2023 compared to fiscal year 2022.

Intelligent Cloud revenue was $87.9 billion, an increase of 17%. Azure and other cloud services revenue grew 29%.

Productivity and Business Processes revenue was $69.3 billion, an increase of 9%.

Operating income increased 6% to $88.5 billion. Net income was $72.4 billion, up 20% year-over-year.

Capital expenditures were $28.1 billion, primarily to support cloud infrastructure growth.
        """,
    },
    {
        "ticker": "NVDA",
        "filing_type": "10-K",
        "year": 2024,
        "content": """
NVIDIA CORPORATION ANNUAL REPORT ON FORM 10-K

ITEM 1. BUSINESS
NVIDIA is a leading inventor of the GPU, which creates interactive graphics on smartphones, tablets, PCs, and workstations. NVIDIA's focus on AI computing has driven extraordinary demand for its data center products.

We operate in two segments: Graphics and Compute & Networking.

ITEM 1A. RISK FACTORS
We are dependent on a highly concentrated customer base. A significant portion of our data center revenue comes from a limited number of customers. The loss of any of these customers could materially reduce our revenue.

Export control regulations represent a material risk. U.S. government restrictions on exports of our products to certain customers in China and other regions have impacted and may continue to impact our business.

Supply chain constraints remain a risk. Our ability to secure sufficient supply of advanced semiconductors and packaging capacity from TSMC and other partners is critical.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
Revenue for fiscal year 2024 was $60.9 billion, an increase of 122% compared to fiscal year 2023.

Data Center revenue was $47.5 billion, up 217% year-over-year, driven by demand for the NVIDIA H100 GPU for AI training and inference workloads.

Gross margin was 72.7% for fiscal 2024, compared to 56.9% for fiscal 2023, reflecting the strong pricing power of H100 and A100 systems.

Net income was $29.8 billion for fiscal 2024. R&D expenses were $8.7 billion, up 20% from the prior year.
        """,
    },
]


def run_sample_ingestion(output_dir: str = "./data") -> None:
    console.print("[bold green]Ingesting sample data (no SEC download needed)[/bold green]")

    all_chunks = []
    for doc in track(SAMPLE_DOCUMENTS, description="Chunking sample docs..."):
        chunks = semantic_chunk(
            text=doc["content"],
            source=f"sample/{doc['ticker']}_{doc['filing_type']}_{doc['year']}.txt",
            ticker=doc["ticker"],
            filing_type=doc["filing_type"],
            year=doc["year"],
        )
        all_chunks.extend(chunks)

    console.print(f"Generated [bold]{len(all_chunks)}[/bold] chunks")

    added = embed_and_store(
        chunks=all_chunks,
        persist_dir=f"{output_dir}/chroma",
    )
    console.print(f"[bold green]✓ Stored {added} chunks in ChromaDB[/bold green]")
    console.print(f"\n[bold]Ready![/bold] Run: [cyan]streamlit run app/streamlit_app.py[/cyan]")


def run_full_ingestion(
    tickers: list[str],
    filing_types: list[str],
    years: list[int],
    data_dir: str = "./data",
    email: str = "research@example.com",
) -> None:
    console.print(f"[bold]Starting ingestion:[/bold] {tickers} | {filing_types} | {years}")

    # Step 1: Download
    console.print("\n[bold blue]Step 1/3: Downloading from SEC EDGAR...[/bold blue]")
    downloaded = download_filings(
        tickers=tickers,
        filing_types=filing_types,
        output_dir=f"{data_dir}/raw",
        years=years,
        email=email,
    )
    total_files = sum(len(v) for v in downloaded.values())
    console.print(f"Downloaded [bold]{total_files}[/bold] files")

    # Step 2: Parse + Chunk
    console.print("\n[bold blue]Step 2/3: Parsing and chunking...[/bold blue]")
    all_chunks = []
    for ticker, file_paths in downloaded.items():
        for file_path in track(file_paths, description=f"Processing {ticker}..."):
            try:
                text = parse_sec_filing(file_path)
                meta = infer_metadata_from_path(Path(file_path))
                chunks = semantic_chunk(
                    text=text,
                    source=str(file_path),
                    ticker=meta["ticker"],
                    filing_type=meta["filing_type"],
                    year=meta["year"],
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

    console.print(f"Generated [bold]{len(all_chunks)}[/bold] chunks total")

    # Step 3: Embed + Store
    console.print("\n[bold blue]Step 3/3: Embedding and storing...[/bold blue]")
    added = embed_and_store(
        chunks=all_chunks,
        persist_dir=f"{data_dir}/chroma",
    )
    console.print(f"\n[bold green]✓ Ingestion complete! {added} new chunks stored.[/bold green]")
    console.print(f"Run: [cyan]streamlit run app/streamlit_app.py[/cyan]")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight ingestion pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS[:3])
    parser.add_argument("--filing-types", nargs="+", default=["10-K"])
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--email", default="research@example.com",
                        help="Your email for SEC EDGAR user-agent (required by EDGAR ToS)")
    parser.add_argument("--use-sample-data", action="store_true",
                        help="Skip SEC download and use built-in sample data")
    args = parser.parse_args()

    if args.use_sample_data:
        run_sample_ingestion(output_dir=args.data_dir)
    else:
        run_full_ingestion(
            tickers=args.tickers,
            filing_types=args.filing_types,
            years=args.years,
            data_dir=args.data_dir,
            email=args.email,
        )


if __name__ == "__main__":
    main()
