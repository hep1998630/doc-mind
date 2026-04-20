"""Abstract interface for OCR engines."""

from abc import ABC, abstractmethod

import numpy as np

from docmind.models.ocr import OCRResult


class BaseOCREngine(ABC):
    """
    Abstract interface that all OCR engine implementations must follow.

    Defines the contract between the OCR module and the rest of the
    pipeline. Implementations handle engine-specific initialization,
    inference, and output normalization internally — the pipeline
    only interacts through this interface.
    """

    @abstractmethod
    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Run OCR on a preprocessed image.

        Args:
            image: Preprocessed grayscale image as a numpy array.

        Returns:
            OCRResult containing all detected text regions with
            their bounding boxes, text content, confidence scores,
            and script directions.
        """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """
        Return the name of this OCR engine.

        Used to populate the 'engine' field in OCRResult for
        traceability and logging.
        """

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """
        Return the list of languages this engine supports.

        Used by the pipeline to verify that the configured languages
        are supported before processing begins.
        """
