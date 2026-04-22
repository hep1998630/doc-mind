"""Abstract interface for layout analysis models."""

from abc import ABC, abstractmethod

import numpy as np

from docmind.models.layout import LayoutResult


class BaseLayoutAnalyzer(ABC):
    """
    Abstract interface that all layout analysis implementations must follow.

    Defines the contract between the layout module and the rest of the
    pipeline. Implementations handle model-specific initialization,
    inference, and output normalization internally.
    """

    @abstractmethod
    def analyze(self, image: np.ndarray) -> LayoutResult:
        """
        Run layout analysis on a preprocessed image.

        Args:
            image: Preprocessed image as a numpy array.

        Returns:
            LayoutResult containing all detected structural regions
            with their categories, bounding boxes, and confidence scores.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the name of this layout model.

        Used to populate the 'model' field in LayoutResult for
        traceability and logging.
        """

    @property
    @abstractmethod
    def supported_categories(self) -> list[str]:
        """
        Return the list of layout categories this model can detect.

        Used by the pipeline to verify compatibility with expected
        document structures.
        """
