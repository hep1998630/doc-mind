"""Schemas for the OCR module output."""

from pydantic import BaseModel, Field

from docmind.models.common import (
    BoundingBox,
    Confidence,
    ImageSize,
    ScriptDirection,
)


class TextRegion(BaseModel):
    """
    A single detected text region from the OCR engine.

    Represents one contiguous block of text (typically a word or a line,
    depending on the OCR engine's granularity) along with its spatial
    position, confidence, and script direction.
    """
    text: str = Field(description="The recognized text content.")
    bbox: BoundingBox = Field(description="Spatial position of the text region.")
    confidence: Confidence = Field(description="Recognition confidence score (0 to 1).")
    script_direction: ScriptDirection = Field(
        default=ScriptDirection.LTR,
        description="Reading direction of the detected text.",
    )


class OCRResult(BaseModel):
    """
    Complete output of an OCR engine for a single page/image.

    Contains all detected text regions along with metadata about
    the source image and the engine that produced the result.
    """
    text_regions: list[TextRegion] = Field(
        default_factory=list,
        description="List of all detected text regions.",
    )
    image_size: ImageSize = Field(
        description="Dimensions of the source image in pixels.",
    )
    engine: str = Field(
        description="Name of the OCR engine that produced this result (e.g., 'paddleocr').",
    )
    page_index: int = Field(
        default=0,
        description="Zero-based page index for multi-page documents.",
    )
