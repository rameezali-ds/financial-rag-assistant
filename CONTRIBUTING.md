# Contributing to FinSight RAG

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/financial-rag-assistant
cd financial-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env
```

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Project Conventions

- **Commits**: conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
- **Code style**: Ruff (enforced in CI)
- **Type hints**: all public functions must have type annotations
- **Docstrings**: Google style

## Adding a New Retriever

1. Create `src/retrieval/my_retriever.py` with a `retrieve(query, top_k) -> list[Document]` method
2. Register it in `HybridRetriever` as an optional backend
3. Add unit tests in `tests/test_retrieval.py`
4. Benchmark against the existing hybrid retriever using `scripts/run_evaluation.py --compare`

## Evaluation-Driven Development

Before merging any retrieval change:
```bash
python scripts/run_evaluation.py --use-financebench --compare
```
The CI eval gate will block merges that drop faithfulness below 0.70.
