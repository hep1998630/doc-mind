"""LangChain-based structured data extraction implementation."""

import json
import logging

from docmind.config.settings import ExtractionSettings, get_settings
from docmind.models.common import DocumentType
from docmind.models.extraction import (
    ExtractionField,
    ExtractionResult,
    LineItem,
    LLMExtractionResponse,
)
from docmind.models.ocr import OCRResult
from docmind.modules.extraction.base import BaseExtractor

logger = logging.getLogger(__name__)


# --- Prompt Templates ---

_SYSTEM_PROMPT = (
    "You are a document data extraction assistant. You receive OCR text "
    "from a document, where each text region includes its spatial "
    "coordinates [y, x] on the page. Use these coordinates to understand "
    "the document layout — text at similar y-values is on the same line, "
    "text with smaller y-values is higher on the page. "
    "The document may contain both Arabic and English text. "
    "Extract the requested fields accurately based on the text content "
    "and spatial relationships."
)

_INVOICE_INSTRUCTIONS = (
    "This is an invoice document. Extract the following fields:\n"
    "- vendor_name: Name of the vendor/supplier\n"
    "- customer_name: Name of the customer/buyer\n"
    "- invoice_number: Invoice or document number\n"
    "- invoice_date: Date of the invoice (preserve original format)\n"
    "- due_date: Payment due date (preserve original format, null if not found)\n"
    "- subtotal: Subtotal amount before tax (as number)\n"
    "- tax_amount: Tax amount (as number, null if not found)\n"
    "- total_amount: Total amount due (as number)\n"
    "- currency: Currency code or symbol\n\n"
    "Also extract each line item from the invoice table with: "
    "description, amount, quantity, unit_price, and item_code where available."
)

_RECEIPT_INSTRUCTIONS = (
    "This is a receipt document. Extract the following fields:\n"
    "- store_name: Name of the store/merchant\n"
    "- receipt_number: Receipt or transaction number\n"
    "- receipt_date: Date of the transaction (preserve original format)\n"
    "- receipt_time: Time of the transaction (null if not found)\n"
    "- subtotal: Subtotal before tax (as number, null if not found)\n"
    "- tax_amount: Tax amount (as number, null if not found)\n"
    "- total_amount: Total amount paid (as number)\n"
    "- payment_method: Payment method used (null if not found)\n"
    "- currency: Currency code or symbol\n\n"
    "Also extract each purchased item with: "
    "description, amount, quantity, unit_price, and item_code where available."
)

_DOCUMENT_INSTRUCTIONS: dict[DocumentType, str] = {
    DocumentType.INVOICE: _INVOICE_INSTRUCTIONS,
    DocumentType.RECEIPT: _RECEIPT_INSTRUCTIONS,
}


class LangChainExtractor(BaseExtractor):
    """
    Extracts structured data from OCR results using an LLM via LangChain.

    Formats OCR text regions with spatial coordinates into a prompt,
    sends it to the configured LLM provider, and parses the structured
    response into an ExtractionResult.

    Supports Anthropic, OpenAI, and OpenAI-compatible local providers
    (Ollama, vLLM, etc.) through LangChain's unified interface.

    Args:
        settings: Extraction configuration. If None, loads from the
            application settings.
    """

    def __init__(self, settings: ExtractionSettings | None = None) -> None:
        self._settings = settings or get_settings().extraction
        self._llm = self._initialize_llm()
        self._structured_llm = self._initialize_structured_output()

    def _initialize_llm(self):
        """
        Initialize the LangChain chat model based on the configured provider.

        Returns:
            A LangChain BaseChatModel instance.
        """
        provider = self._settings.provider.lower()
        api_key = (
            self._settings.api_key.get_secret_value()
            if self._settings.api_key
            else None
        )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            logger.info(
                "Initializing ChatAnthropic with model='%s'",
                self._settings.model,
            )
            return ChatAnthropic(
                model=self._settings.model,
                api_key=api_key,
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
            )

        else:
            # OpenAI and OpenAI-compatible providers (local, ollama, vllm)
            from langchain_openai import ChatOpenAI

            logger.info(
                "Initializing ChatOpenAI with model='%s', base_url='%s'",
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
        """
        Try to create a structured output chain.

        Falls back to None if the provider doesn't support structured
        output, in which case we use manual JSON prompting and parsing.

        Returns:
            A LangChain runnable with structured output, or None.
        """
        try:
            structured = self._llm.with_structured_output(
                LLMExtractionResponse
            )
            logger.info("Structured output initialized successfully")
            return structured
        except (NotImplementedError, TypeError, Exception) as e:
            logger.warning(
                "Structured output not supported by this provider, "
                "falling back to JSON prompting: %s", e
            )
            return None

    def extract(
        self,
        ocr_result: OCRResult,
        document_type: DocumentType,
    ) -> ExtractionResult:
        """
        Extract structured data from OCR results.

        Formats OCR text with coordinates, sends to the LLM, and
        parses the structured response.

        Args:
            ocr_result: Output from the OCR module.
            document_type: Type of document for prompt selection.

        Returns:
            ExtractionResult with extracted fields and line items.
        """
        formatted_text = self._format_ocr_text(ocr_result)
        messages = self._build_messages(formatted_text, document_type)

        logger.info(
            "Sending extraction request to LLM (%s) for %s",
            self._settings.model, document_type.value,
        )

        if self._structured_llm is not None:
            try:
                llm_response = self._call_structured(messages)
            except Exception as e:
                logger.warning(
                    "Structured output failed at runtime, falling back "
                    "to JSON prompting: %s", e,
                )
                llm_response = self._call_with_fallback(messages)
        else:
            llm_response = self._call_with_fallback(messages)

        return self._build_result(
            llm_response, document_type, formatted_text, ocr_result
        )

    @property
    def extractor_name(self) -> str:
        return f"langchain_{self._settings.provider}_{self._settings.model}"

    # --- Private: OCR Text Formatting ---

    def _format_ocr_text(self, ocr_result: OCRResult) -> str:
        """
        Format OCR text regions as coordinate-tagged lines.

        Sorts regions top-to-bottom, then left-to-right within
        the same line. Each region is formatted as:
            [y=N, x=N] "text"

        Args:
            ocr_result: OCR results with text regions.

        Returns:
            Formatted string with all text regions.
        """
        if not ocr_result.text_regions:
            return "(No text detected)"

        # Sort by y first (top to bottom), then x (left to right)
        sorted_regions = sorted(
            ocr_result.text_regions,
            key=lambda r: (r.bbox.center.y, r.bbox.center.x),
        )

        lines = []
        for region in sorted_regions:
            center = region.bbox.center
            direction = region.script_direction.value.upper()
            lines.append(
                f'[y={center.y:.0f}, x={center.x:.0f}] '
                f'({direction}) "{region.text}"'
            )

        return "\n".join(lines)

    # --- Private: Message Construction ---

    def _build_messages(
        self, formatted_text: str, document_type: DocumentType
    ) -> list:
        """
        Build the message list for the LLM call.

        Args:
            formatted_text: Formatted OCR text with coordinates.
            document_type: Document type for instruction selection.

        Returns:
            List of LangChain message tuples.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        instructions = _DOCUMENT_INSTRUCTIONS.get(
            document_type,
            _INVOICE_INSTRUCTIONS,
        )

        user_content = (
            f"{instructions}\n\n"
            f"--- Document OCR Text ---\n"
            f"{formatted_text}"
        )

        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    # --- Private: LLM Calling ---

    def _call_structured(self, messages: list) -> LLMExtractionResponse:
        """
        Call the LLM with structured output enforcement.

        Args:
            messages: LangChain message list.

        Returns:
            Parsed LLMExtractionResponse.
        """
        response = self._structured_llm.invoke(messages)

        if isinstance(response, LLMExtractionResponse):
            return response

        # Some providers return a dict instead of a Pydantic model
        if isinstance(response, dict):
            return LLMExtractionResponse(**response)

        raise ValueError(
            f"Unexpected response type from structured LLM: "
            f"{type(response).__name__}"
        )

    def _call_with_fallback(self, messages: list) -> LLMExtractionResponse:
        """
        Call the LLM with manual JSON prompting and parsing.

        Used when structured output is not supported by the provider.

        Args:
            messages: LangChain message list.

        Returns:
            Parsed LLMExtractionResponse.
        """
        from langchain_core.messages import SystemMessage

        # Add JSON instruction to the system prompt
        json_instruction = SystemMessage(
            content=(
                "You MUST respond with valid JSON only. No markdown, "
                "no explanation, no preamble. The JSON must conform to "
                "this schema:\n"
                "{\n"
                '  "fields": [{"field_name": "...", "value": ..., "confidence": 0.0-1.0}],\n'
                '  "line_items": [{"description": "...", "amount": 0.0, '
                '"quantity": null, "unit_price": null, "item_code": null, '
                '"confidence": 0.0-1.0}]\n'
                "}"
            )
        )
        messages_with_json = [messages[0], json_instruction] + messages[1:]

        response = self._llm.invoke(messages_with_json)
        response_text = response.content

        # Clean up potential markdown fences
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (``` markers)
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
                "Failed to parse LLM response as JSON: %s\nResponse: %s",
                e, response_text[:500],
            )
            # Return empty result rather than crashing
            return LLMExtractionResponse(fields=[], line_items=[])

    # --- Private: Result Building ---

    def _build_result(
        self,
        llm_response: LLMExtractionResponse,
        document_type: DocumentType,
        raw_text: str,
        ocr_result: OCRResult,
    ) -> ExtractionResult:
        """
        Convert the LLM response into a full ExtractionResult.

        Maps the LLM's simplified response models to our full
        extraction schemas and adds metadata.

        Args:
            llm_response: Parsed LLM response.
            document_type: Type of document processed.
            raw_text: The formatted text sent to the LLM.
            ocr_result: Original OCR result for metadata.

        Returns:
            Complete ExtractionResult with metadata.
        """
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
            page_index=ocr_result.page_index,
        )
