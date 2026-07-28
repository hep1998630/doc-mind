# DocMind

Document understanding tool that extracts structured data — fields and line
items — from **invoice and receipt images**. Built for bilingual
(Arabic + English) Saudi Arabian business documents.

DocMind ships two pipeline modes behind a single interface:

- **VLM mode (default, recommended):** the image is sent directly to a
  multimodal LLM. Best accuracy for Arabic invoices in benchmarking.
- **OCR + LLM mode:** PaddleOCR extracts text, then a text LLM structures it.
  Lower Arabic accuracy, but runs against text-only models.

## Why VLM

Across 17 model configurations on 21 verified Arabic invoices, VLM
consistently beat OCR + LLM:

| Rank | Configuration        | Field Fuzzy | Item F1 | Numeric Acc | Avg Time |
|------|----------------------|-------------|---------|-------------|----------|
| 1    | VLM Claude Opus 4.7  | 75.4%       | 60.2%   | 71.2%       | 14.2s    |
| 2    | VLM Gemini 2.5 Pro   | 71.7%       | 75.9%   | 71.2%       | 24.4s    |
| 3    | VLM Gemini 2.5 Flash | 71.0%       | 77.0%   | 63.5%       | 5.1s     |
| 4    | OCR + LLM (best)     | 48.6%       | 68.6%   | 53.8%       | varies   |

**Gemini 2.5 Flash** is the recommended production model — best balance of
speed, cost, and accuracy.

## Quick Start

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# Configure your provider and keys
cp .env.example .env
# Edit .env — set EXTRACTION__API_KEY (and provider/model if changing defaults)
```

The defaults in `.env.example` use Gemini 2.5 Flash via OpenRouter. Any
OpenAI-compatible provider works by setting `EXTRACTION__BASE_URL`.

### Run the API

```bash
uvicorn docmind.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Extract from a file upload:

```bash
curl -X POST localhost:8000/extract \
  -F "file=@invoice.jpg;type=image/jpeg" \
  -F "document_type=invoice"
```

Endpoints: `POST /extract` (file upload), `POST /extract/base64`,
`GET /health`, `GET /info`. Optional API-key auth via the `X-API-Key` header —
enable it by setting `API_KEYS` (comma-separated) in `.env`; leave it empty for
open access in development.

### Try it from the command line

The standalone tools live in `scripts/` and are run from the project root:

```bash
# Full pipeline (mode comes from PIPELINE_MODE)
python scripts/sanity_check.py invoice.jpg --pipeline invoice

# VLM extraction only
python scripts/sanity_check.py invoice.jpg --vlm invoice

# OCR + LLM extraction
python scripts/sanity_check.py invoice.jpg --extract invoice
```

## Configuration

All settings load from `.env` via `pydantic-settings` (nested keys use `__`).
Key options:

| Variable                | Purpose                                         |
|-------------------------|-------------------------------------------------|
| `PIPELINE_MODE`         | `vlm` (default) or `ocr`                         |
| `EXTRACTION__PROVIDER`  | `anthropic`, `openai`, or OpenAI-compatible      |
| `EXTRACTION__MODEL`     | Model id (e.g. `google/gemini-2.5-flash`)        |
| `EXTRACTION__API_KEY`   | Provider API key                                 |
| `EXTRACTION__BASE_URL`  | Custom base URL (Groq, OpenRouter, Ollama, …)    |
| `API_KEYS`              | Comma-separated keys for API auth (empty = off)  |

See `.env.example` for the full list, including OCR and preprocessing options.

## Architecture

```
docmind/
├── api/          # FastAPI app (/extract, /extract/base64, /health, /info)
├── core/         # Pipeline orchestrator (single process() entry point)
├── config/       # Centralized Pydantic settings from .env
├── models/       # Pydantic data schemas (the contract between modules)
└── modules/
    ├── preprocessing/  # OpenCV cleanup (OCR mode only)
    ├── ocr/            # PaddleOCR (DeepSeek-OCR experimental)
    ├── layout/         # YOLO + DocLayNet (available, not in MVP pipeline)
    ├── mapping/        # Spatial OCR ↔ layout matching (not in MVP pipeline)
    └── extraction/     # langchain_extractor (OCR+LLM) + vlm_extractor (VLM)
```

Swappable OCR and extraction implementations sit behind abstract base classes,
so a new engine or provider drops in without touching the pipeline. LLM
responses use structured output with a JSON-parsing fallback for messy output.

Evaluation, benchmarking, and ground-truth tooling live in `scripts/`.
Developer reference and design notes are in [CLAUDE.md](CLAUDE.md).

## Status

- **Phase 1 — Core pipeline:** complete (schemas, preprocessing, OCR, VLM &
  OCR+LLM extraction, orchestrator, evaluation harness, benchmarking).
- **Phase 2 — API layer:** in progress (endpoints and auth done; Dockerfile,
  CI/CD, and monitoring pending).
- **Phase 3+ — Cloud deployment, CI/CD, polish:** planned.
