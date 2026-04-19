"""Centralized configuration for all DocMind modules."""

from functools import lru_cache

from typing import Optional

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings


# --- Module-Level Settings ---


class PreprocessingSettings(BaseModel):
    """Configuration for the image preprocessing module."""
    enable_deskew: bool = Field(
        default=True,
        description="Whether to apply automatic deskewing.",
    )
    enable_denoise: bool = Field(
        default=True,
        description="Whether to apply denoising.",
    )
    enable_binarize: bool = Field(
        default=False,
        description=(
            "Whether to apply binarization. Disabled by default as it "
            "can hurt accuracy on already-clean documents."
        ),
    )
    binarize_block_size: int = Field(
        default=11,
        description="Block size for adaptive thresholding during binarization.",
    )
    denoise_kernel_size: int = Field(
        default=3,
        description="Kernel size for denoising filter.",
    )


class OCRSettings(BaseModel):
    """Configuration for the OCR module."""
    engine: str = Field(
        default="paddleocr",
        description="OCR engine to use.",
    )
    languages: list[str] = Field(
        default=["ar", "en"],
        description="Languages to detect, in priority order.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        description=(
            "Minimum confidence score for a text region to be included. "
            "Regions below this threshold are discarded."
        ),
    )


class LayoutSettings(BaseModel):
    """Configuration for the layout analysis module."""
    model_path: str = Field(
        default="./weights/yolo_layout.pt",
        description="Path to the YOLO layout model weights.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence score for a layout region to be included.",
    )
    categories_to_keep: list[str] | None = Field(
        default=None,
        description=(
            "List of layout categories to keep (e.g., ['table', 'text', 'title']). "
            "If None, all detected categories are kept."
        ),
    )


class MappingSettings(BaseModel):
    """Configuration for the region mapping module."""
    overlap_threshold: float = Field(
        default=0.5,
        description=(
            "Minimum overlap ratio (intersection area / text region area) "
            "required to assign a text region to a layout region."
        ),
    )
    use_center_containment: bool = Field(
        default=True,
        description=(
            "If True, assign text regions based on whether their center point "
            "falls within a layout region. Faster but less precise than "
            "overlap-based assignment."
        ),
    )


class ExtractionSettings(BaseModel):
    """Configuration for the LLM extraction module."""
    provider: str = Field(
        default="anthropic",
        description=(
            "LLM provider to use ('anthropic', 'openai', or 'local' "
            "for locally hosted models)."
        ),
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "API key for the LLM provider. "
            "Not required for local models."
        ),
    )
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for the LLM API. Required for local models "
            "(e.g., 'http://localhost:11434/v1' for Ollama). "
            "If None, the provider's default URL is used."
        ),
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model name to use for extraction.",
    )
    temperature: float = Field(
        default=0.0,
        description=(
            "Sampling temperature. Set to 0 for deterministic, "
            "reproducible extractions."
        ),
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens in the LLM response.",
    )


# --- Top-Level Settings ---


class Settings(BaseSettings):
    """
    Top-level application settings.

    Loads configuration from environment variables. Nested settings
    use prefixed variable names:
        PREPROCESSING__ENABLE_DESKEW=true
        OCR__LANGUAGES='["ar","en"]'
        EXTRACTION__API_KEY=sk-...

    Usage:
        settings = get_settings()
        threshold = settings.ocr.confidence_threshold
    """
    preprocessing: PreprocessingSettings = Field(
        default_factory=PreprocessingSettings,
    )
    ocr: OCRSettings = Field(
        default_factory=OCRSettings,
    )
    layout: LayoutSettings = Field(
        default_factory=LayoutSettings,
    )
    mapping: MappingSettings = Field(
        default_factory=MappingSettings,
    )
    extraction: ExtractionSettings = Field(
        default_factory=ExtractionSettings,
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    model_config = {
        "env_nested_delimiter": "__",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached singleton of the application settings.

    Uses lru_cache so the settings are loaded once from environment
    variables and reused across the application.
    """
    return Settings()
