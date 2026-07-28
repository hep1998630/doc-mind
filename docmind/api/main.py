"""
DocMind API — Document extraction service.

A FastAPI application that accepts invoice/receipt images and
returns structured extraction results.

Usage:
    uvicorn docmind.api.main:app --host 0.0.0.0 --port 8000

Configuration is read from .env file.
"""

import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Response,
    Security,
    UploadFile,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from docmind.config.settings import get_settings
from docmind.core.pipeline import Pipeline, PipelineResult
from docmind.models.common import DocumentType
from docmind.models.extraction import ExtractionError

logger = logging.getLogger(__name__)

# --- App Setup ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline on startup, clean up on shutdown."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting DocMind API")

    # Pre-initialize pipeline so the first request isn't slow.
    get_pipeline()
    logger.info("Pipeline ready")

    yield

    logger.info("Shutting down DocMind API")


app = FastAPI(
    title="DocMind",
    description=(
        "Document extraction API — extracts structured data from "
        "invoices and receipts using AI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# --- Security ---

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_keys() -> list[str]:
    """Load valid API keys from settings."""
    settings = get_settings()
    keys_str = settings.api_keys
    if not keys_str:
        return []
    return [k.strip() for k in keys_str.split(",") if k.strip()]


async def verify_api_key(
    api_key: Optional[str] = Security(API_KEY_HEADER),
) -> str:
    """
    Verify the API key from the request header.

    If no API keys are configured, authentication is disabled
    (open access for development).
    """
    valid_keys = get_api_keys()

    # If no keys configured, allow open access (dev mode)
    if not valid_keys:
        return "dev-mode"

    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )

    return api_key


# --- Pipeline Singleton ---

_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = Pipeline(settings=settings, mode=settings.pipeline_mode)
    return _pipeline


# --- Request/Response Models ---

class Base64ExtractionRequest(BaseModel):
    """Request body for base64 image extraction."""
    image_base64: str = Field(
        description="Base64-encoded image data (JPEG or PNG).",
    )
    document_type: str = Field(
        default="invoice",
        description="Document type: 'invoice' or 'receipt'.",
    )


class ExtractionResponse(BaseModel):
    """API response for extraction requests."""
    status: str = Field(description="'success' or 'error'.")
    data: Optional[dict] = Field(
        default=None,
        description="Extraction result with fields and line items.",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Processing metadata (timing, pipeline info).",
    )
    error: Optional[dict] = Field(
        default=None,
        description="Error details if status is 'error'.",
    )


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    pipeline_mode: str
    model: str


class InfoResponse(BaseModel):
    """Response for info endpoint."""
    version: str
    pipeline_mode: str
    model: str
    provider: str
    supported_document_types: list[str]


# --- Helper Functions ---

def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode a base64 string to a numpy image array."""
    import cv2

    try:
        # Handle data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_base64)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image from base64 data")

        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 image data: {e}",
        )


def read_upload_image(contents: bytes) -> np.ndarray:
    """Read an uploaded file's bytes into a numpy image array."""
    import cv2

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode uploaded image. Ensure it is a valid JPEG or PNG.",
        )

    return image


def validate_document_type(doc_type: str) -> DocumentType:
    """Validate and convert document type string."""
    try:
        return DocumentType(doc_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type: '{doc_type}'. Must be 'invoice' or 'receipt'.",
        )


def build_response(pipeline_result: PipelineResult) -> ExtractionResponse:
    """Build a successful API response from pipeline result."""
    result_dict = pipeline_result.to_dict()

    return ExtractionResponse(
        status="success",
        data=result_dict["extraction"],
        metadata=result_dict["metadata"],
    )


def build_error_response(
    response: Response, error_type: str, message: str, status_code: int
) -> ExtractionResponse:
    """Build an error API response and set the HTTP status code."""
    response.status_code = status_code
    return ExtractionResponse(
        status="error",
        error={
            "type": error_type,
            "message": message,
        },
    )


# --- Endpoints ---

@app.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extract data from a document image (file upload)",
    description=(
        "Upload a document image (JPEG or PNG) and receive "
        "structured extraction results."
    ),
)
async def extract_from_upload(
    response: Response,
    file: UploadFile = File(..., description="Document image file (JPEG or PNG)"),
    document_type: str = Form(
        default="invoice",
        description="Document type: 'invoice' or 'receipt'",
    ),
    api_key: str = Depends(verify_api_key),
):
    """Extract structured data from an uploaded document image."""
    doc_type = validate_document_type(document_type)

    # Validate file type
    if file.content_type and file.content_type not in [
        "image/jpeg", "image/png", "image/jpg",
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Read and decode image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    image = read_upload_image(contents)

    # Process
    pipeline = get_pipeline()
    try:
        result = pipeline.process(image, doc_type)
        return build_response(result)
    except ExtractionError as e:
        logger.error("Extraction failed [%s]: %s", e.error_type, e)
        return build_error_response(
            response,
            error_type=e.error_type,
            message=str(e),
            status_code=422,
        )
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        return build_error_response(
            response,
            error_type="internal_error",
            message=str(e),
            status_code=500,
        )


@app.post(
    "/extract/base64",
    response_model=ExtractionResponse,
    summary="Extract data from a base64-encoded document image",
    description=(
        "Send a base64-encoded document image and receive "
        "structured extraction results."
    ),
)
async def extract_from_base64(
    request: Base64ExtractionRequest,
    response: Response,
    api_key: str = Depends(verify_api_key),
):
    """Extract structured data from a base64-encoded document image."""
    doc_type = validate_document_type(request.document_type)

    image = decode_base64_image(request.image_base64)

    pipeline = get_pipeline()
    try:
        result = pipeline.process(image, doc_type)
        return build_response(result)
    except ExtractionError as e:
        logger.error("Extraction failed [%s]: %s", e.error_type, e)
        return build_error_response(
            response,
            error_type=e.error_type,
            message=str(e),
            status_code=422,
        )
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        return build_error_response(
            response,
            error_type="internal_error",
            message=str(e),
            status_code=500,
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check():
    """Check if the service is running and the pipeline is ready."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="healthy",
        pipeline_mode=pipeline.mode,
        model=pipeline.model_name,
    )


@app.get(
    "/info",
    response_model=InfoResponse,
    summary="Service information",
)
async def service_info():
    """Get information about the service configuration."""
    settings = get_settings()
    pipeline = get_pipeline()
    return InfoResponse(
        version="0.1.0",
        pipeline_mode=pipeline.mode,
        model=settings.extraction.model,
        provider=settings.extraction.provider,
        supported_document_types=["invoice", "receipt"],
    )
