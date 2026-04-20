"""PaddleOCR implementation of the OCR engine interface."""

import logging

import cv2
import numpy as np
from paddleocr import PaddleOCR

from docmind.config.settings import OCRSettings, get_settings
from docmind.models.common import BoundingBox, ImageSize, ScriptDirection
from docmind.models.ocr import OCRResult, TextRegion
from docmind.modules.ocr.base import BaseOCREngine

logger = logging.getLogger(__name__)

# Arabic Unicode ranges for script detection
_ARABIC_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)


class PaddleOCREngine(BaseOCREngine):
    """
    OCR engine implementation using PaddleOCR (3.x API).

    Supports bilingual Arabic + English text detection and recognition.
    Handles initialization of the PaddleOCR model, inference, output
    normalization to the common schema, and script direction detection.

    Args:
        settings: OCR configuration. If None, loads from the
            application settings.
    """

    def __init__(self, settings: OCRSettings | None = None) -> None:
        self._settings = settings or get_settings().ocr
        self._engine = self._initialize_engine()

    def _initialize_engine(self) -> PaddleOCR:
        """
        Initialize the PaddleOCR engine with configured languages, device,
        and optional model selections.

        PaddleOCR's 'ar' language setting handles both Arabic and
        Latin text recognition in the same image.
        """
        primary_lang = self._settings.languages[0]
        device = self._settings.device

        logger.info(
            "Initializing PaddleOCR with language='%s', device='%s'",
            primary_lang, device,
        )

        kwargs = {
            "use_angle_cls": True,
            "lang": primary_lang,
            "show_log": False,
            "device": device,
            # Disable PaddleOCR's internal document preprocessing.
            # We handle our own preprocessing, so letting PaddleOCR
            # also transform the image causes coordinate mismatches
            # between the returned bounding boxes and our input image.
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }

        # MKL-DNN is only relevant for CPU inference
        if device == "cpu":
            kwargs["enable_mkldnn"] = False

        # Allow specifying detection and recognition models independently
        if self._settings.detection_model:
            kwargs["text_detection_model_name"] = self._settings.detection_model
            logger.info("Using detection model: %s", self._settings.detection_model)

        if self._settings.recognition_model:
            kwargs["text_recognition_model_name"] = self._settings.recognition_model
            logger.info("Using recognition model: %s", self._settings.recognition_model)

        return PaddleOCR(**kwargs)

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Run PaddleOCR on a preprocessed image.

        Uses the PaddleOCR 3.x predict() API. The result is a generator
        of result objects, each containing a 'res' dict with detection
        polygons, recognized texts, and confidence scores.

        Args:
            image: Preprocessed image as a numpy array.

        Returns:
            OCRResult with detected text regions, filtered by the
            configured confidence threshold.
        """
        h, w = image.shape[:2]
        image_size = ImageSize(width=w, height=h)

        # PaddleOCR 3.x expects a 3-channel BGR image.
        # Convert grayscale to BGR if needed.
        ocr_input = self._ensure_bgr(image)

        results = list(self._engine.predict(ocr_input))

        text_regions = self._parse_results(results)

        logger.info(
            "PaddleOCR detected %d text regions (%d after filtering)",
            self._count_raw_regions(results),
            len(text_regions),
        )

        return OCRResult(
            text_regions=text_regions,
            image_size=image_size,
            engine=self.engine_name,
        )

    @property
    def engine_name(self) -> str:
        return "paddleocr"

    @property
    def supported_languages(self) -> list[str]:
        return ["ar", "en", "ch", "fr", "de", "ko", "ja"]

    # --- Private: Image Handling ---

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """
        Ensure the image is in 3-channel BGR format.

        PaddleOCR 3.x requires a 3-channel image for its internal
        normalization pipeline. If a grayscale image is passed,
        it is converted to BGR.

        Args:
            image: Input image as a numpy array.

        Returns:
            3-channel BGR image.
        """
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    # --- Private: Result Parsing ---

    def _parse_results(
        self, results: list
    ) -> list[TextRegion]:
        """
        Parse PaddleOCR 3.x output into TextRegion objects.

        PaddleOCR 3.x predict() returns result objects with a 'res' dict
        containing parallel arrays:
            - dt_polys: detection polygons (array of 4-point polygons)
            - rec_texts: recognized text strings (list)
            - rec_scores: recognition confidence scores (array)
            - rec_polys: recognition polygons (array, may differ from dt_polys)

        Args:
            results: List of result objects from PaddleOCR predict().

        Returns:
            List of TextRegion objects, filtered by confidence threshold.
        """
        text_regions: list[TextRegion] = []

        if not results:
            return text_regions

        for result in results:
            res = self._extract_res_dict(result)

            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            # Prefer dt_polys (detection polygons) — these are always
            # relative to the input image. rec_polys may be transformed
            # by internal PaddleOCR post-processing.
            polys = res.get("dt_polys", res.get("rec_polys", []))

            if len(rec_texts) == 0:
                continue

            for i in range(len(rec_texts)):
                text = rec_texts[i]
                confidence = float(rec_scores[i])
                bbox_points = polys[i].tolist()

                if confidence < self._settings.confidence_threshold:
                    continue

                if not text.strip():
                    continue

                bbox = BoundingBox.from_points_list(bbox_points)
                script_direction = self._detect_script_direction(text)

                region = TextRegion(
                    text=text.strip(),
                    bbox=bbox,
                    confidence=confidence,
                    script_direction=script_direction,
                )
                text_regions.append(region)

        return text_regions

    def _extract_res_dict(self, result) -> dict:
        """
        Extract the 'res' dictionary from a PaddleOCR result object.

        Handles both dict-style results and object-style results
        for compatibility across PaddleOCR versions.

        Args:
            result: A single result from PaddleOCR predict().

        Returns:
            The 'res' dictionary containing detection and recognition data.
        """
        if isinstance(result, dict):
            return result.get("res", result)
        if hasattr(result, "res"):
            res = result.res
            return res if isinstance(res, dict) else result
        return result

    def _detect_script_direction(self, text: str) -> ScriptDirection:
        """
        Detect the script direction of a text string.

        Checks whether the majority of alphabetic characters fall
        within Arabic Unicode ranges. If so, the text is RTL;
        otherwise LTR.

        Args:
            text: The text string to analyze.

        Returns:
            ScriptDirection.RTL if the text is predominantly Arabic,
            ScriptDirection.LTR otherwise.
        """
        arabic_count = 0
        total_alpha = 0

        for char in text:
            if char.isalpha():
                total_alpha += 1
                code_point = ord(char)
                for start, end in _ARABIC_RANGES:
                    if start <= code_point <= end:
                        arabic_count += 1
                        break

        if total_alpha == 0:
            return ScriptDirection.LTR

        if arabic_count / total_alpha > 0.5:
            return ScriptDirection.RTL

        return ScriptDirection.LTR

    def _count_raw_regions(self, results: list) -> int:
        """Count the total number of regions in raw PaddleOCR output."""
        total = 0
        for result in results:
            res = self._extract_res_dict(result)
            rec_texts = res.get("rec_texts", [])
            total += len(rec_texts)
        return total
