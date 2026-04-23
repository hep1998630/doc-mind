"""Abstract interface for structured data extraction."""

from abc import ABC, abstractmethod

from docmind.models.common import DocumentType
from docmind.models.extraction import ExtractionResult
from docmind.models.ocr import OCRResult


class BaseExtractor(ABC):
    """
    Abstract interface that all extraction implementations must follow.

    Defines the contract for extracting structured data from OCR results.
    Implementations handle the specifics of how extraction is performed
    (e.g., LLM prompting, rule-based parsing).
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
