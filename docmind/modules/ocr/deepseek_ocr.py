"""DeepSeek-OCR implementation using Ollama."""

import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import requests

from docmind.config.settings import OCRSettings, get_settings
from docmind.models.common import ImageSize
from docmind.models.ocr import OCRResult
from docmind.modules.ocr.base import BaseOCREngine

logger = logging.getLogger(__name__)

# DeepSeek-OCR specific prompts
PROMPT_FREE_OCR = "Free OCR."
PROMPT_MARKDOWN = "<|grounding|>Convert the document to markdown."
PROMPT_EXTRACT_TEXT = "Extract the text in the image."
PROMPT_LAYOUT = "<|grounding|>Given the layout of the image."


class DeepSeekOCREngine(BaseOCREngine):
    """
    OCR engine implementation using DeepSeek-OCR via Ollama.

    DeepSeek-OCR is a specialized vision-language model for OCR tasks.
    It returns flowing text or markdown rather than individual text
    regions with bounding boxes. The output is stored in OCRResult's
    raw_text field.

    Args:
        settings: OCR configuration. If None, loads from the
            application settings.
        ollama_base_url: Base URL for the Ollama API.
            Defaults to http://localhost:11434.
        model_name: Ollama model name. Defaults to 'deepseek-ocr'.
        prompt_mode: Which prompt to use. Options:
            'markdown' — structured markdown output (recommended
                for invoices, preserves tables and layout)
            'free' — raw text extraction
            'extract' — extract text
            'layout' — layout-aware extraction
    """

    def __init__(
        self,
        settings: OCRSettings | None = None,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "deepseek-ocr",
        prompt_mode: str = "markdown",
    ) -> None:
        self._settings = settings or get_settings().ocr
        self._base_url = ollama_base_url.rstrip("/")
        self._model_name = model_name
        self._prompt = self._get_prompt(prompt_mode)

        logger.info(
            "Initializing DeepSeek-OCR (model=%s, prompt_mode=%s)",
            model_name, prompt_mode,
        )

        # Verify Ollama is running and model is available
        self._verify_connection()

    @staticmethod
    def _get_prompt(mode: str) -> str:
        """Get the prompt string for the given mode."""
        prompts = {
            "markdown": PROMPT_MARKDOWN,
            "free": PROMPT_FREE_OCR,
            "extract": PROMPT_EXTRACT_TEXT,
            "layout": PROMPT_LAYOUT,
        }
        if mode not in prompts:
            raise ValueError(
                f"Unknown prompt mode '{mode}'. "
                f"Choose from: {list(prompts.keys())}"
            )
        return prompts[mode]

    def _verify_connection(self) -> None:
        """Verify Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if self._model_name not in model_names:
                available = ", ".join(model_names[:10])
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s. "
                    "Run 'ollama pull %s' to download it.",
                    self._model_name, available, self._model_name,
                )
        except requests.ConnectionError:
            logger.warning(
                "Could not connect to Ollama at %s. "
                "Make sure Ollama is running.",
                self._base_url,
            )

    def recognize(self, image: np.ndarray) -> OCRResult:
        """
        Run DeepSeek-OCR on a preprocessed image.

        Sends the image to Ollama's DeepSeek-OCR model with the
        configured prompt and returns the raw text output.

        Args:
            image: Preprocessed image as a numpy array.

        Returns:
            OCRResult with raw_text populated and empty text_regions.
        """
        h, w = image.shape[:2]
        image_size = ImageSize(width=w, height=h)

        # Encode image to base64
        image_b64 = self._encode_image(image)

        # Call Ollama API
        raw_text = self._call_ollama(image_b64)

        logger.info(
            "DeepSeek-OCR produced %d characters of text",
            len(raw_text),
        )

        return OCRResult(
            text_regions=[],
            image_size=image_size,
            engine=self.engine_name,
            raw_text=raw_text,
        )

    @property
    def engine_name(self) -> str:
        return f"deepseek-ocr-{self._model_name}"

    @property
    def supported_languages(self) -> list[str]:
        return ["ar", "en", "ch", "fr", "de", "ko", "ja", "ru"]

    # --- Private ---

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """Encode numpy image to base64 string."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        _, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        return base64.b64encode(buffer).decode("utf-8")

    def _call_ollama(self, image_b64: str) -> str:
        """
        Call Ollama's chat API with the image.

        Uses the /api/chat endpoint with streaming disabled.
        """
        url = f"{self._base_url}/api/chat"

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "user",
                    "content": self._prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 8192,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running."
            )
        except requests.Timeout:
            raise TimeoutError(
                "Ollama request timed out after 120 seconds. "
                "The image may be too large or the model too slow."
            )
        except requests.HTTPError as e:
            raise RuntimeError(
                f"Ollama API error: {e.response.status_code} — "
                f"{e.response.text[:200]}"
            )

        result = response.json()
        message = result.get("message", {})
        content = message.get("content", "")

        if not content.strip():
            logger.warning("DeepSeek-OCR returned empty content")

        return content.strip()
