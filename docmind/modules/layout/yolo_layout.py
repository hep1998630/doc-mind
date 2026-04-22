"""YOLO-based layout analysis implementation."""

import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from docmind.config.settings import LayoutSettings, get_settings
from docmind.models.common import BoundingBox, ImageSize
from docmind.models.layout import LayoutCategory, LayoutRegion, LayoutResult
from docmind.modules.layout.base import BaseLayoutAnalyzer

logger = logging.getLogger(__name__)

# Default class index to LayoutCategory mapping for DocLayNet.
# This matches the standard DocLayNet label ordering.
# Override via the class_mapping parameter if using a different dataset.
DOCLAYNET_CLASS_MAPPING: dict[int, LayoutCategory] = {
    0: LayoutCategory.CAPTION,
    1: LayoutCategory.FOOTNOTE,
    2: LayoutCategory.FORMULA,
    3: LayoutCategory.LIST_ITEM,
    4: LayoutCategory.PAGE_FOOTER,
    5: LayoutCategory.PAGE_HEADER,
    6: LayoutCategory.PICTURE,
    7: LayoutCategory.SECTION_HEADER,
    8: LayoutCategory.TABLE,
    9: LayoutCategory.TEXT,
    10: LayoutCategory.TITLE,
}


class YOLOLayoutAnalyzer(BaseLayoutAnalyzer):
    """
    Layout analysis implementation using YOLOv8.

    Detects structural regions in document images (tables, headers,
    text blocks, etc.) using a YOLO model trained on a document
    layout dataset such as DocLayNet.

    Args:
        settings: Layout configuration. If None, loads from the
            application settings.
        class_mapping: Optional mapping from YOLO class indices to
            LayoutCategory values. Defaults to DocLayNet mapping.
    """

    def __init__(
        self,
        settings: LayoutSettings | None = None,
        class_mapping: dict[int, LayoutCategory] | None = None,
    ) -> None:
        self._settings = settings or get_settings().layout
        self._class_mapping = class_mapping or DOCLAYNET_CLASS_MAPPING
        self._model = self._load_model()

    def _load_model(self) -> YOLO:
        """
        Load the YOLO model from the configured path.

        Raises:
            FileNotFoundError: If the model weights file does not exist.
        """
        model_path = Path(self._settings.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Layout model weights not found at: {model_path.resolve()}\n"
                f"Please download a YOLOv8 model trained on DocLayNet and "
                f"place it at the configured path, or update "
                f"LAYOUT__MODEL_PATH in your .env file.\n"
                f"Pretrained models are available on HuggingFace — search "
                f"for 'yolov8 doclaynet'."
            )

        logger.info("Loading YOLO layout model from: %s", model_path)

        return YOLO(str(model_path))

    def analyze(self, image: np.ndarray) -> LayoutResult:
        """
        Run layout analysis on a preprocessed image.

        Args:
            image: Preprocessed image as a numpy array (grayscale or BGR).

        Returns:
            LayoutResult with detected structural regions.
        """
        h, w = image.shape[:2]
        image_size = ImageSize(width=w, height=h)

        # YOLO expects BGR 3-channel input
        model_input = self._ensure_bgr(image)

        results = self._model(
            model_input,
            conf=self._settings.confidence_threshold,
            verbose=False,
        )

        regions = self._parse_results(results)

        logger.info("Layout analysis detected %d regions", len(regions))

        return LayoutResult(
            regions=regions,
            image_size=image_size,
            model=self.model_name,
        )

    @property
    def model_name(self) -> str:
        return f"yolo_layout_{self._settings.model_path}"

    @property
    def supported_categories(self) -> list[str]:
        return [cat.value for cat in self._class_mapping.values()]

    # --- Private ---

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Convert grayscale to BGR if needed."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _parse_results(self, results) -> list[LayoutRegion]:
        """
        Parse YOLO detection results into LayoutRegion objects.

        Args:
            results: Raw YOLO inference results.

        Returns:
            List of LayoutRegion objects, filtered by configured
            categories if specified.
        """
        regions: list[LayoutRegion] = []

        if not results or len(results) == 0:
            return regions

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return regions

        for i in range(len(boxes)):
            class_idx = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            # Map class index to category
            category = self._class_mapping.get(class_idx)
            if category is None:
                logger.warning(
                    "Unknown class index %d, skipping region", class_idx
                )
                continue

            # Filter by allowed categories if configured
            if self._settings.categories_to_keep is not None:
                if category.value not in self._settings.categories_to_keep:
                    continue

            region = LayoutRegion(
                region_id=f"region_{len(regions)}",
                category=category,
                bbox=BoundingBox.from_xyxy(x1, y1, x2, y2),
                confidence=confidence,
            )
            regions.append(region)

        return regions
