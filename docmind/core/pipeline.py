"""
Pipeline orchestrator for DocMind.

Wires together the processing modules into a single callable pipeline.
Supports two modes: VLM (image → structured data) and OCR+LLM
(image → OCR text → structured data).

The pipeline mode and model configuration are read from settings.
"""

import logging
import time
from pathlib import Path

import numpy as np

from docmind.config.settings import Settings, get_settings
from docmind.models.common import DocumentType
from docmind.models.extraction import ExtractionResult
from docmind.models.preprocessing import PreprocessingMetadata

logger = logging.getLogger(__name__)


class PipelineResult:
    """
    Complete result of a pipeline run.

    Contains the extraction result plus metadata about the
    processing (timing, pipeline mode, preprocessing details).
    """

    def __init__(
        self,
        extraction: ExtractionResult,
        preprocessing_metadata: PreprocessingMetadata,
        pipeline_mode: str,
        processing_time_seconds: float,
    ):
        self.extraction = extraction
        self.preprocessing_metadata = preprocessing_metadata
        self.pipeline_mode = pipeline_mode
        self.processing_time_seconds = processing_time_seconds

    def to_dict(self) -> dict:
        """Serialize to a dictionary for API responses."""
        return {
            "extraction": self.extraction.model_dump(),
            "metadata": {
                "pipeline_mode": self.pipeline_mode,
                "processing_time_seconds": round(
                    self.processing_time_seconds, 2
                ),
                "preprocessing": self.preprocessing_metadata.model_dump(),
            },
        }


class Pipeline:
    """
    Main document processing pipeline.

    Initializes all required components on construction and exposes
    a single process() method. Components are initialized once and
    reused across calls.

    Args:
        settings: Application settings. If None, loads from .env.
        mode: Pipeline mode override. If None, auto-detects:
            uses 'vlm' if the configured model supports vision,
            otherwise 'ocr'. Explicit values: 'vlm' or 'ocr'.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        mode: str | None = None,
    ):
        self._settings = settings or get_settings()
        self._mode = mode or self._detect_mode()
        self._components = self._initialize_components()

        logger.info(
            "Pipeline initialized: mode=%s, model=%s",
            self._mode, self._settings.extraction.model,
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def model_name(self) -> str:
        return self._settings.extraction.model

    def _detect_mode(self) -> str:
        """
        Auto-detect pipeline mode from settings.

        Defaults to 'vlm' since it outperforms OCR+LLM for Arabic
        invoices based on benchmarking results.
        """
        # Default to VLM — override with explicit mode if needed
        return "vlm"

    def _initialize_components(self) -> dict:
        """Initialize pipeline components based on mode."""
        from docmind.modules.preprocessing.processor import ImagePreprocessor

        components = {
            "preprocessor": ImagePreprocessor(
                settings=self._settings.preprocessing
            ),
        }

        if self._mode == "vlm":
            from docmind.modules.extraction.vlm_extractor import VLMExtractor

            components["extractor"] = VLMExtractor(
                settings=self._settings.extraction
            )
            logger.info("VLM pipeline components initialized")

        elif self._mode == "ocr":
            from docmind.modules.extraction.langchain_extractor import (
                LangChainExtractor,
            )

            ocr_engine_name = self._settings.ocr.engine

            if ocr_engine_name == "deepseek-ocr":
                from docmind.modules.ocr.deepseek_ocr import DeepSeekOCREngine

                components["ocr_engine"] = DeepSeekOCREngine()
            else:
                from docmind.modules.ocr.paddle_ocr import PaddleOCREngine

                components["ocr_engine"] = PaddleOCREngine(
                    settings=self._settings.ocr
                )

            components["extractor"] = LangChainExtractor(
                settings=self._settings.extraction
            )
            logger.info("OCR+LLM pipeline components initialized")

        else:
            raise ValueError(f"Unknown pipeline mode: {self._mode}")

        return components

    def process(
        self,
        image: np.ndarray | str | Path,
        document_type: DocumentType = DocumentType.INVOICE,
    ) -> PipelineResult:
        """
        Process a document image and extract structured data.

        Args:
            image: Document image as a numpy array, file path string,
                or Path object.
            document_type: Type of document for extraction prompt
                selection. Defaults to INVOICE.

        Returns:
            PipelineResult containing extraction results and metadata.

        Raises:
            ExtractionError: If the LLM fails to produce valid output.
            FileNotFoundError: If the image path doesn't exist.
            ValueError: If the image is invalid.
        """
        start_time = time.time()

        preprocessor = self._components["preprocessor"]

        if self._mode == "vlm":
            # VLM mode: preprocess for basic cleanup, then send
            # original image to VLM (VLM handles its own image reading)
            processed_image, prep_metadata = preprocessor.process(image)

            extractor = self._components["extractor"]
            # Pass the original image path if available (better quality
            # than re-encoding the preprocessed numpy array).
            # Fall back to preprocessed array if input was an array.
            if isinstance(image, (str, Path)):
                extraction = extractor.extract_from_image(
                    image, document_type
                )
            else:
                extraction = extractor.extract_from_image(
                    processed_image, document_type
                )

        elif self._mode == "ocr":
            # OCR mode: preprocess → OCR → LLM extraction
            processed_image, prep_metadata = preprocessor.process(image)

            ocr_engine = self._components["ocr_engine"]
            ocr_result = ocr_engine.recognize(processed_image)

            extractor = self._components["extractor"]
            extraction = extractor.extract(ocr_result, document_type)

        processing_time = time.time() - start_time

        logger.info(
            "Pipeline completed: mode=%s, time=%.2fs, fields=%d, items=%d",
            self._mode,
            processing_time,
            len(extraction.fields),
            len(extraction.line_items),
        )

        return PipelineResult(
            extraction=extraction,
            preprocessing_metadata=prep_metadata,
            pipeline_mode=self._mode,
            processing_time_seconds=processing_time,
        )
