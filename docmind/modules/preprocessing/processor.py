"""Image preprocessing module for cleaning document images before OCR."""

from pathlib import Path

import cv2
import numpy as np

from docmind.config.settings import PreprocessingSettings, get_settings
from docmind.models.common import ImageSize
from docmind.models.preprocessing import PreprocessingMetadata, PreprocessingOperation


class ImagePreprocessor:
    """
    Cleans and normalizes document images for optimal OCR performance.

    Applies a configurable sequence of operations: grayscale conversion,
    deskewing, denoising, and binarization. Each operation can be
    enabled or disabled through settings.

    Args:
        settings: Preprocessing configuration. If None, loads from
            the application settings.
    """

    def __init__(self, settings: PreprocessingSettings | None = None) -> None:
        self._settings = settings or get_settings().preprocessing

    def process(
        self, image: np.ndarray | str | Path
    ) -> tuple[np.ndarray, PreprocessingMetadata]:
        """
        Preprocess a document image for OCR.

        Args:
            image: Input image as a numpy array, file path string,
                or Path object.

        Returns:
            A tuple of (processed_image, metadata) where processed_image
            is the cleaned numpy array and metadata records what
            operations were applied.

        Raises:
            FileNotFoundError: If a file path is provided and the file
                does not exist.
            ValueError: If the image is empty or could not be loaded.
        """
        image = self._load_image(image)
        original_size = ImageSize(
            width=image.shape[1], height=image.shape[0]
        )

        operations: list[PreprocessingOperation] = []

        image, grayscale_op = self._to_grayscale(image)
        if grayscale_op:
            operations.append(grayscale_op)

        if self._settings.enable_deskew:
            image, deskew_op = self._deskew(image)
            if deskew_op:
                operations.append(deskew_op)

        if self._settings.enable_denoise:
            image, denoise_op = self._denoise(image)
            if denoise_op:
                operations.append(denoise_op)

        if self._settings.enable_binarize:
            image, binarize_op = self._binarize(image)
            if binarize_op:
                operations.append(binarize_op)

        processed_size = ImageSize(
            width=image.shape[1], height=image.shape[0]
        )

        metadata = PreprocessingMetadata(
            original_size=original_size,
            processed_size=processed_size,
            operations_applied=operations,
            was_modified=len(operations) > 0,
        )

        return image, metadata

    # --- Private: Image Loading ---

    def _load_image(self, image: np.ndarray | str | Path) -> np.ndarray:
        """
        Load an image from a file path or validate a numpy array.

        Args:
            image: Input image as a numpy array, file path string,
                or Path object.

        Returns:
            The loaded image as a numpy array.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            loaded = cv2.imread(str(path))
            if loaded is None:
                raise ValueError(f"Could not load image from: {path}")
            return loaded

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy array, str, or Path, got {type(image).__name__}."
            )

        if image.size == 0:
            raise ValueError("Input image is empty.")

        return image

    # --- Private: Operations ---

    def _to_grayscale(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, PreprocessingOperation | None]:
        """Convert a color image to grayscale. No-op if already grayscale."""
        if len(image.shape) == 2:
            return image, None

        if image.shape[2] == 1:
            return image[:, :, 0], None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        op = PreprocessingOperation(
            name="grayscale",
            parameters={"original_channels": image.shape[2]},
        )
        return gray, op

    def _deskew(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, PreprocessingOperation | None]:
        """
        Detect and correct document skew.

        Uses angle detection (isolated in _detect_skew_angle) to find
        the rotation angle, then applies an affine rotation to correct it.
        """
        angle = self._detect_skew_angle(image)

        if abs(angle) < 0.5:
            return image, None

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions to avoid cropping
        cos = abs(rotation_matrix[0, 0])
        sin = abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust the rotation matrix for the new dimensions
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        op = PreprocessingOperation(
            name="deskew",
            parameters={"angle_degrees": round(angle, 2)},
        )
        return rotated, op

    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detect the skew angle of a document image.

        Uses morphological operations to find text contours, then
        fits a minimum area rectangle to the largest contour cluster
        to estimate the dominant text rotation angle.

        This method is isolated so it can be easily swapped for an
        alternative approach (e.g., Hough line detection, projection
        profiling) without modifying the rest of the deskew logic.

        Args:
            image: Grayscale document image.

        Returns:
            Estimated skew angle in degrees. Positive means
            counter-clockwise rotation is needed to correct.
        """
        # Binarize for contour detection
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Dilate to merge text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 0.0

        # Combine all contour points and fit a minimum area rectangle
        all_points = np.concatenate(contours)
        rect = cv2.minAreaRect(all_points)
        angle = rect[-1]

        # Normalize angle to [-45, 45] range
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90

        return angle

    def _denoise(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, PreprocessingOperation | None]:
        """Apply Gaussian blur denoising."""
        kernel_size = self._settings.denoise_kernel_size

        # Kernel size must be odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        op = PreprocessingOperation(
            name="denoise",
            parameters={
                "method": "gaussian",
                "kernel_size": kernel_size,
            },
        )
        return denoised, op

    def _binarize(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, PreprocessingOperation | None]:
        """Apply adaptive thresholding for binarization."""
        block_size = self._settings.binarize_block_size

        # Block size must be odd and greater than 1
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        binary = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            10,
        )

        op = PreprocessingOperation(
            name="binarize",
            parameters={
                "method": "adaptive_gaussian",
                "block_size": block_size,
            },
        )
        return binary, op
