# DocMind — Developer Reference

## Project Overview

DocMind is a document understanding tool that extracts structured data (fields and line items) from invoice and receipt images. It supports two pipeline modes: VLM (vision-language model, recommended) and OCR+LLM (PaddleOCR + text-based LLM). The project is bilingual (Arabic + English), targeting Saudi Arabian business documents.

## Architecture

```
docmind/
├── api/                        # FastAPI application (Phase 2)
│   └── main.py                 # Endpoints: /extract, /extract/base64, /health, /info
├── core/
│   └── pipeline.py             # Pipeline orchestrator — single process() method
├── config/
│   └── settings.py             # Centralized Pydantic BaseSettings from .env
├── models/                     # Pydantic data schemas (no logic)
│   ├── common.py               # Point, BoundingBox, ImageSize, enums
│   ├── preprocessing.py        # PreprocessingMetadata
│   ├── ocr.py                  # TextRegion, OCRResult (has raw_text field)
│   ├── layout.py               # LayoutCategory, LayoutRegion, LayoutResult
│   ├── mapping.py              # MappedRegion, MappingResult
│   └── extraction.py           # ExtractionField, LineItem, ExtractionResult,
│                               # LLMExtractionResponse, ExtractionError,
│                               # parse_llm_json_response()
├── modules/
│   ├── preprocessing/
│   │   └── processor.py        # OpenCV image cleaning (deskew, denoise, binarize)
│   ├── ocr/
│   │   ├── base.py             # BaseOCREngine abstract interface
│   │   ├── paddle_ocr.py       # PaddleOCR implementation (3.x API, GPU)
│   │   └── deepseek_ocr.py     # DeepSeek-OCR via Ollama (broken — needs vLLM)
│   ├── layout/                 # Not used in MVP pipeline
│   │   ├── base.py             # BaseLayoutAnalyzer abstract interface
│   │   └── yolo_layout.py      # YOLOv8 on DocLayNet
│   ├── mapping/                # Not used in MVP pipeline
│   │   └── region_mapper.py    # Spatial OCR→layout matching
│   └── extraction/
│       ├── base.py             # BaseExtractor + BaseVLMExtractor interfaces
│       ├── langchain_extractor.py  # OCR text → LLM → structured JSON
│       └── vlm_extractor.py    # Image → VLM → structured JSON
```

## Pipeline Modes

### VLM Mode (Recommended, Default)
Image → VLM Extractor → ExtractionResult

Best performer from benchmarking. Gemini 2.5 Flash is the recommended model (best speed/accuracy/cost balance). The **original** image is base64-encoded and sent directly to a multimodal LLM — the OCR-oriented preprocessing (grayscale/deskew/denoise) is intentionally skipped in VLM mode because it strips colour cues that aid extraction. The VLM extractor handles its own resizing (`EXTRACTION__MAX_IMAGE_LONG_EDGE`).

### OCR+LLM Mode
Image → Preprocessing → PaddleOCR → LangChain Extractor → ExtractionResult

Lower accuracy for Arabic due to PaddleOCR limitations (only mobile-tier Arabic recognition model available). OCR output can be coordinate-tagged text (PaddleOCR) or raw markdown (DeepSeek-OCR — currently broken on Ollama).

## Key Design Decisions

- **Abstract interfaces** for OCR (`BaseOCREngine`) and extraction (`BaseExtractor`, `BaseVLMExtractor`) enable swapping implementations without touching pipeline code
- **Pydantic v2** for all schemas with strict validation. LLM response schemas (`LLMExtractionResponse`) have `field_validator` decorators to coerce messy LLM output (None→str, float→str)
- **LangChain** (modular packages: langchain-core, langchain-anthropic, langchain-openai) for LLM abstraction. Supports Anthropic, OpenAI, and OpenAI-compatible providers (Groq, OpenRouter, Ollama)
- **Structured output with fallback** — tries `with_structured_output()` first, falls back to JSON prompting + `parse_llm_json_response()` which handles preamble text, markdown fences, and truncated JSON
- **`OCRResult.raw_text`** field allows OCR engines that produce flowing text (DeepSeek-OCR) to work alongside engines that produce individual text regions (PaddleOCR)
- **`ExtractionError`** with typed `error_type` field for categorized failure reporting
- **Settings** loaded from `.env` via `pydantic-settings` with `env_nested_delimiter="__"`. Cached singleton via `get_settings()`
- **Pipeline singleton** in API — initialized once at startup, reused for all requests

## Benchmarking Results Summary

Evaluated 17 model configurations across 21 verified Arabic invoices:

| Rank | Configuration | Field Fuzzy | Item F1 | Numeric Acc | Avg Time |
|------|--------------|-------------|---------|-------------|----------|
| 1 | VLM Claude Opus 4.7 | 75.4% | 60.2% | 71.2% | 14.2s |
| 2 | VLM Gemini 2.5 Pro | 71.7% | 75.9% | 71.2% | 24.4s |
| 3 | VLM Gemini 2.5 Flash | 71.0% | 77.0% | 63.5% | 5.1s |
| 4 | OCR+LLM (best) | 48.6% | 68.6% | 53.8% | varies |

**Key finding:** VLM consistently outperforms OCR+LLM for Arabic invoices. Gemini 2.5 Flash recommended for production (best speed/cost with competitive accuracy).

## What's Been Completed

### Phase 1 — Core Pipeline ✅
- [x] All data schemas (models/)
- [x] Centralized configuration (config/settings.py)
- [x] Image preprocessing module (OpenCV)
- [x] PaddleOCR integration (3.x API, GPU, bilingual)
- [x] Layout analysis module (YOLO + DocLayNet) — available but not in MVP pipeline
- [x] Region mapping module — available but not in MVP pipeline
- [x] LangChain-based LLM extractor (structured output + JSON fallback)
- [x] VLM extractor (multimodal LLM, base64 image input)
- [x] Pipeline orchestrator (core/pipeline.py)
- [x] Sanity check script with all pipeline modes
- [x] OCR evaluation script (containment-based matching)
- [x] Ground truth generation pipeline (hybrid OCR+GT annotation → LLM structuring)
- [x] Streamlit review app for ground truth validation
- [x] Streamlit image selection app
- [x] Extraction evaluation script with per-field and per-item metrics
- [x] Model comparison script
- [x] Benchmarked 17 model configurations

### Phase 2 — API Layer (In Progress)
- [x] FastAPI application with /extract and /extract/base64 endpoints
- [x] API key authentication (optional, configurable)
- [x] Health check and info endpoints
- [ ] Dockerfile and docker-compose.yml
- [ ] CI/CD with GitHub Actions
- [ ] Monitoring and logging infrastructure

### Phase 3 — Cloud Deployment (Planned)
- [ ] Deploy to GCP Cloud Run or AWS ECS
- [ ] Cloud storage for uploaded documents (S3/GCS)
- [ ] Managed database for extraction results
- [ ] HTTPS/TLS, custom domain
- [ ] Billing alerts and cost monitoring

### Phase 4 — CI/CD & Monitoring (Planned)
- [ ] GitHub Actions: test → build → push → deploy
- [ ] Request tracing with unique IDs
- [ ] Prometheus + Grafana or CloudWatch
- [ ] Rate limiting
- [ ] Error alerting

### Phase 5 — Polish & Ship (Planned)
- [ ] Frontend demo page (Streamlit or React)
- [ ] Strong README with architecture diagram and demo GIF
- [ ] Blog post about the project
- [ ] Open source on GitHub

## Immediate Next Steps

1. **Dockerize the application** — Dockerfile for the API server, docker-compose.yml with the service definition. No GPU needed for VLM mode (it calls external APIs). Consider multi-stage build to keep image small.

2. **Test the API thoroughly** — verify both upload and base64 endpoints work, test error cases (invalid images, unsupported formats, large files), verify API key auth works correctly.

3. **Add request logging** — structured JSON logs with request ID, processing time, document type, field count, success/failure status.

4. **Cloud deployment** — start with GCP Cloud Run (simplest for containerized apps). Set up billing alerts immediately.

## Code Conventions

- Python 3.10+, type hints everywhere
- Pydantic v2 for data validation (use `model_dump()` not `.dict()`)
- Abstract base classes for swappable components
- Settings from .env via pydantic-settings, never hardcoded
- Logging via Python's `logging` module, not print statements
- Error handling: use ExtractionError with typed error_type for extraction failures
- Each module receives its own settings slice, not the full Settings object
- `get_settings()` returns cached singleton — call it, don't pass settings through every layer

## Common Commands

```bash
# Run API server
uvicorn docmind.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run sanity check (VLM mode)
python scripts/sanity_check.py invoice.jpg --vlm invoice

# Run sanity check (OCR mode)
python scripts/sanity_check.py invoice.jpg --extract invoice

# Run the full pipeline orchestrator (mode from PIPELINE_MODE)
python scripts/sanity_check.py invoice.jpg --pipeline invoice

# Run evaluation
python scripts/evaluate_extraction.py ground_truth/ samples/invoices/ --mode vlm --report-name my_test

# Compare reports
python scripts/compare_reports.py ground_truth/

# Review ground truth
streamlit run scripts/review_app.py

# Select images for ground truth
streamlit run scripts/select_images.py
```

All standalone tooling lives in `scripts/` and is run from the project root
(paths are resolved relative to your working directory, not the script).

## Environment Setup

```bash
pip install -r requirements.txt

# For PaddleOCR GPU (only needed for OCR mode):
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Copy and configure .env:
cp .env.example .env
# (.env.example is the maintained template — VLM/Gemini defaults, PIPELINE_MODE, API_KEYS)
# Edit .env with your API keys and preferences
```

## Working With This Codebase

When making changes:
- Read the relevant abstract interface (base.py) before modifying an implementation
- Check models/ schemas before adding new data fields — the schema is the contract
- Run sanity_check.py after changes to verify the pipeline still works
- The evaluation scripts in `scripts/` are standalone tools, not part of the package (run them from the project root)
- Layout and mapping modules exist but are NOT in the MVP pipeline — they're available for future use
- DeepSeek-OCR engine exists but is broken on Ollama (needs vLLM for custom logit processor) — don't try to fix it without vLLM

When adding a new LLM provider:
1. It likely works already via ChatOpenAI with a custom base_url
2. Test structured output — if it fails, the JSON fallback handles it
3. Watch for type coercion issues in LLM responses (the validators in extraction.py handle most cases)

When adding a new OCR engine:
1. Implement BaseOCREngine from modules/ocr/base.py
2. If it produces flowing text (not individual regions), populate raw_text in OCRResult
3. The langchain_extractor automatically uses raw_text when available
