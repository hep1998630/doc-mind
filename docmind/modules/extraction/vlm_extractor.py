"""Vision-Language Model (VLM) based extraction implementation."""

import base64
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from docmind.config.settings import ExtractionSettings, get_settings
from docmind.models.common import DocumentType
from docmind.models.extraction import (
    ExtractionField,
    ExtractionResult,
    LineItem,
    LLMExtractionResponse,
)
from docmind.modules.extraction.base import BaseVLMExtractor

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

_SYSTEM_PROMPT = (
    "You are a document data extraction assistant with vision capabilities. "
    "You receive an image of a document and must extract structured data "
    "from it. Read all text in the image carefully, including both Arabic "
    "and English text. Pay close attention to numbers, dates, and amounts."
)

_INVOICE_INSTRUCTIONS = (
    "This image is an invoice. Extract the following fields:\n"
    "- vendor_name: Name of the vendor/supplier\n"
    "- customer_name: Name of the customer/buyer\n"
    "- invoice_number: Invoice or document number\n"
    "- invoice_date: Date of the invoice (preserve original format)\n"
    "- due_date: Payment due date (preserve original format, null if not found)\n"
    "- subtotal: Subtotal amount before tax (as string number)\n"
    "- tax_amount: Tax amount (as string number, null if not found)\n"
    "- total_amount: Total amount due (as string number)\n"
    "- currency: Currency code or symbol\n\n"
    "Also extract each line item from the invoice table with: "
    "description, amount, quantity, unit_price, and item_code where available."
)

_RECEIPT_INSTRUCTIONS = (
    "This image is a receipt. Extract the following fields:\n"
    "- store_name: Name of the store/merchant\n"
    "- receipt_number: Receipt or transaction number\n"
    "- receipt_date: Date of the transaction (preserve original format)\n"
    "- receipt_time: Time of the transaction (null if not found)\n"
    "- subtotal: Subtotal before tax (as string number, null if not found)\n"
    "- tax_amount: Tax amount (as string number, null if not found)\n"
    "- total_amount: Total amount paid (as string number)\n"
    "- payment_method: Payment method used (null if not found)\n"
    "- currency: Currency code or symbol\n\n"
    "Also extract each purchased item with: "
    "description, amount, quantity, unit_price, and item_code where available."
)

_DOCUMENT_INSTRUCTIONS: dict[DocumentType, str] = {
    DocumentType.INVOICE: _INVOICE_INSTRUCTIONS,
    DocumentType.RECEIPT: _RECEIPT_INSTRUCTIONS,
}


class VLMExtractor(BaseVLMExtractor):
    """
    Extracts structured data directly from document images using a
    Vision-Language Model via LangChain.

    Sends the document image to a multimodal LLM (GPT-4o, Claude,
    Gemini, etc.) which reads the text and extracts structured data
    in a single pass — no separate OCR step required.

    Args:
        settings: Extraction configuration. If None, loads from the
            application settings.
    """

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self._settings = settings or get_settings().extraction
        self._llm = self._initialize_llm()
        self._structured_llm = self._initialize_structured_output()

    def _initialize_llm(self):
        """Initialize the LangChain chat model based on provider."""
        provider = self._settings.provider.lower()
        api_key = (
            self._settings.api_key.get_secret_value()
            if self._settings.api_key
            else None
        )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            logger.info(
                "Initializing VLM ChatAnthropic with model='%s'",
                self._settings.model,
            )
            return ChatAnthropic(
                model=self._settings.model,
                api_key=api_key,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )
        else:
            from langchain_openai import ChatOpenAI

            logger.info(
                "Initializing VLM ChatOpenAI with model='%s', base_url='%s'",
                self._settings.model,
                self._settings.base_url or "default",
            )

            kwargs = {
                "model": self._settings.model,
                "temperature": self._settings.temperature,
                "max_tokens": self._settings.max_tokens,
            }

            if api_key:
                kwargs["api_key"] = api_key
            else:
                kwargs["api_key"] = "not-needed"

            if self._settings.base_url:
                kwargs["base_url"] = self._settings.base_url

            return ChatOpenAI(**kwargs)

    def _initialize_structured_output(self):
        """Try to create structured output chain with fallback."""
        try:
            structured = self._llm.with_structured_output(
                LLMExtractionResponse
            )
            logger.info("VLM structured output initialized successfully")
            return structured
        except (NotImplementedError, TypeError, Exception) as e:
            logger.warning(
                "VLM structured output not supported, "
                "falling back to JSON prompting: %s", e
            )
            return None

    def extract_from_image(
        self,
        image: np.ndarray | str | Path,
        document_type: DocumentType,
    ) -> ExtractionResult:
        """
        Extract structured data directly from a document image.

        Resizes the image if needed, encodes to base64, sends to
        the VLM, and parses the structured response.
        """
        # Load and prepare image
        image_array = self._load_image(image)
        original_h, original_w = image_array.shape[:2]

        # Resize if needed
        image_array = self._resize_image(image_array)
        resized_h, resized_w = image_array.shape[:2]

        # Encode to base64
        image_b64 = self._encode_image(image_array)

        # Build messages with image
        messages = self._build_messages(image_b64, document_type)

        logger.info(
            "Sending image (%dx%d -> %dx%d) to VLM (%s) for %s extraction",
            original_w, original_h, resized_w, resized_h,
            self._settings.model, document_type.value,
        )

        # Call LLM
        if self._structured_llm is not None:
            try:
                llm_response = self._call_structured(messages)
            except Exception as e:
                logger.warning(
                    "VLM structured output failed at runtime, "
                    "falling back to JSON prompting: %s", e
                )
                llm_response = self._call_with_fallback(messages)
        else:
            llm_response = self._call_with_fallback(messages)

        # Build result
        raw_text = (
            f"[VLM extraction from image: "
            f"original={original_w}x{original_h}, "
            f"sent={resized_w}x{resized_h}]"
        )

        return self._build_result(
            llm_response, document_type, raw_text
        )

    @property
    def extractor_name(self) -> str:
        return f"vlm_{self._settings.provider}_{self._settings.model}"

    # --- Private: Image Handling ---

    @staticmethod
    def _load_image(image: np.ndarray | str | Path) -> np.ndarray:
        """Load image from path or validate numpy array."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            loaded = cv2.imread(str(path))
            if loaded is None:
                raise ValueError(f"Could not load image: {path}")
            return loaded

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image input.")

        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image if its long edge exceeds the configured maximum.

        Preserves aspect ratio. Avoids sending unnecessarily large
        images that the VLM would downscale anyway.
        """
        h, w = image.shape[:2]
        max_edge = self._settings.max_image_long_edge
        long_edge = max(h, w)

        if long_edge <= max_edge:
            return image

        scale = max_edge / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        logger.info(
            "Resized image from %dx%d to %dx%d (max_edge=%d)",
            w, h, new_w, new_h, max_edge,
        )
        return resized

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """Encode a numpy image array to base64 JPEG string."""
        # Ensure BGR for encoding
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        _, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        return base64.b64encode(buffer).decode("utf-8")

    # --- Private: Message Construction ---

    def _build_messages(
        self, image_b64: str, document_type: DocumentType
    ) -> list:
        """Build LangChain messages with image content."""
        from langchain_core.messages import HumanMessage, SystemMessage

        instructions = _DOCUMENT_INSTRUCTIONS.get(
            document_type,
            _INVOICE_INSTRUCTIONS,
        )

        human_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                },
            },
            {
                "type": "text",
                "text": instructions,
            },
        ]

        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

    # --- Private: LLM Calling ---

    def _call_structured(self, messages: list) -> LLMExtractionResponse:
        """Call the VLM with structured output enforcement."""
        response = self._structured_llm.invoke(messages)

        if isinstance(response, LLMExtractionResponse):
            return response
        if isinstance(response, dict):
            return LLMExtractionResponse(**response)

        raise ValueError(
            f"Unexpected response type: {type(response).__name__}"
        )

    def _call_with_fallback(self, messages: list) -> LLMExtractionResponse:
        """Call the VLM with manual JSON prompting and parsing."""
        from langchain_core.messages import SystemMessage

        json_instruction = SystemMessage(
            content=(
                "You MUST respond with valid JSON only. No markdown, "
                "no explanation, no preamble. The JSON must conform to "
                "this schema:\n"
                "{\n"
                '  "fields": [{"field_name": "...", "value": "...", '
                '"confidence": 0.0-1.0}],\n'
                '  "line_items": [{"description": "...", "amount": 0.0, '
                '"quantity": null, "unit_price": null, "item_code": null, '
                '"confidence": 0.0-1.0}]\n'
                "}"
            )
        )
        messages_with_json = [messages[0], json_instruction] + messages[1:]

        response = self._llm.invoke(messages_with_json)
        response_text = response.content

        # Clean markdown fences
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = [
                l for l in lines
                if not l.strip().startswith("```")
            ]
            response_text = "\n".join(lines)

        try:
            parsed = json.loads(response_text)
            return LLMExtractionResponse(**parsed)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(
                "Failed to parse VLM response: %s\nResponse: %s",
                e, response_text[:500],
            )
            return LLMExtractionResponse(fields=[], line_items=[])

    # --- Private: Result Building ---

    def _build_result(
        self,
        llm_response: LLMExtractionResponse,
        document_type: DocumentType,
        raw_text: str,
    ) -> ExtractionResult:
        """Convert the VLM response into a full ExtractionResult."""
        fields = [
            ExtractionField(
                field_name=f.field_name,
                value=f.value,
                confidence=max(0.0, min(1.0, f.confidence)),
            )
            for f in llm_response.fields
        ]

        line_items = [
            LineItem(
                description=item.description,
                amount=item.amount,
                quantity=item.quantity,
                unit_price=item.unit_price,
                item_code=item.item_code,
                confidence=max(0.0, min(1.0, item.confidence)),
            )
            for item in llm_response.line_items
        ]

        return ExtractionResult(
            document_type=document_type,
            fields=fields,
            line_items=line_items,
            raw_text=raw_text,
            model=self._settings.model,
        )
