# DocMind — Project Structure

## Directory Structure

```
docmind/
├── docmind/                        # Main package
│   ├── __init__.py
│   │
│   ├── models/                     # Data models / schemas
│   │   ├── __init__.py
│   │   ├── common.py               # Shared types (BoundingBox, Point, etc.)
│   │   ├── preprocessing.py        # Preprocessing input/output schemas
│   │   ├── ocr.py                  # OCR schemas (TextRegion, OCRResult)
│   │   ├── layout.py               # Layout schemas (LayoutRegion, LayoutResult)
│   │   ├── mapping.py              # Mapping schemas (MappedRegion, MappingResult)
│   │   └── extraction.py           # Extraction schemas (ExtractionField, ExtractionResult)
│   │
│   ├── modules/                    # Processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   └── processor.py        # Image preprocessing (OpenCV)
│   │   ├── ocr/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract OCR interface
│   │   │   └── paddle_ocr.py       # PaddleOCR implementation
│   │   ├── layout/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract layout interface
│   │   │   └── yolo_layout.py      # YOLO layout implementation
│   │   ├── mapping/
│   │   │   ├── __init__.py
│   │   │   └── region_mapper.py    # Spatial mapping: OCR regions → layout regions
│   │   └── extraction/
│   │       ├── __init__.py
│   │       ├── base.py             # Abstract extraction interface
│   │       └── llm_extractor.py    # LLM-based structured extraction
│   │
│   ├── core/                       # Pipeline orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py             # Main pipeline: wires all modules together
│   │
│   └── config/                     # Configuration
│       ├── __init__.py
│       └── settings.py             # Centralized settings (Pydantic BaseSettings)
│
├── tests/                          # Tests (mirrors docmind/ structure)
│   ├── __init__.py
│   ├── test_models/
│   │   └── __init__.py
│   ├── test_modules/
│   │   ├── __init__.py
│   │   ├── test_preprocessing.py
│   │   ├── test_ocr.py
│   │   ├── test_layout.py
│   │   ├── test_mapping.py
│   │   └── test_extraction.py
│   └── test_pipeline.py
│
├── samples/                        # Sample documents for testing
│   ├── invoices/
│   └── receipts/
│
├── requirements.txt
├── .env.example                    # Example environment variables
├── .gitignore
└── README.md
```

## Notes

- `api/` directory will be added in Phase 2 when we build the FastAPI layer.
- `Dockerfile` and `docker-compose.yml` will be added in Phase 2.
- The `samples/` directory is for you to collect test documents — gather 5-10 invoices and receipts (both Arabic and English) early on.

## Initial requirements.txt

```
# Core
pydantic>=2.0,<3.0
pydantic-settings>=2.0,<3.0
python-dotenv>=1.0,<2.0

# Preprocessing
opencv-python>=4.8,<5.0
Pillow>=10.0,<11.0
numpy>=1.24,<2.0

# OCR
paddlepaddle>=2.5,<3.0
paddleocr>=2.7,<3.0

# Testing
pytest>=7.0,<9.0
```

## .env.example

```
# OCR
OCR_ENGINE=paddleocr
OCR_LANGUAGES=ar,en

# Layout
LAYOUT_MODEL_PATH=./weights/yolo_layout.pt
LAYOUT_CONFIDENCE_THRESHOLD=0.5

# LLM Extraction
LLM_PROVIDER=anthropic
LLM_API_KEY=your-api-key-here
LLM_MODEL=claude-sonnet-4-20250514

# General
LOG_LEVEL=INFO
```

## .gitignore

```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
venv/
.venv/

# Environment
.env

# IDE
.vscode/
.idea/

# Model weights
weights/
*.pt
*.onnx
*.pdparams

# OS
.DS_Store
Thumbs.db
```
