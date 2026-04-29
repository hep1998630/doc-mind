"""Abstract interfaces for structured data extraction."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from docmind.models.common import DocumentType
from docmind.models.extraction import ExtractionResult
from docmind.models.ocr import OCRResult


class BaseExtractor(ABC):
    """
    Abstract interface for text-based extraction implementations.

    Takes OCR results (text with spatial coordinates) and extracts
    structured data. Used in the OCR+LLM pipeline.
    """

    @abstractmethod
    def extract(
        self,
        ocr_result: OCRResult,
        document_type: DocumentType,
    ) -> ExtractionResult:
        """
        Extract structured data from OCR results.

        Args:
            ocr_result: Output from the OCR module, containing
                text regions with spatial coordinates.
            document_type: The type of document being processed,
                used to select the appropriate extraction schema.

        Returns:
            ExtractionResult containing extracted fields and line items.
        """

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """
        Return the name of this extractor.

        Used for traceability and logging.
        """


class BaseVLMExtractor(ABC):
    """
    Abstract interface for vision-language model extraction.

    Takes a document image directly and extracts structured data
    without a separate OCR step. Used in the VLM pipeline.
    """

    @abstractmethod
    def extract_from_image(
        self,
        image: np.ndarray | str | Path,
        document_type: DocumentType,
    ) -> ExtractionResult:
        """
        Extract structured data directly from a document image.

        Args:
            image: Document image as a numpy array, file path string,
                or Path object.
            document_type: The type of document being processed,
                used to select the appropriate extraction prompt.

        Returns:
            ExtractionResult containing extracted fields and line items.
        """

    @property
    @abstractmethod
    def extractor_name(self) -> str:
        """
        Return the name of this extractor.

        Used for traceability and logging.
        """
