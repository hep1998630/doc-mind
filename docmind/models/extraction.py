"""Schemas for the LLM structured extraction module output."""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from docmind.models.common import Confidence, DocumentType


class ExtractionField(BaseModel):
    """
    A single extracted key-value pair from the document.

    Used for scalar fields such as vendor name, invoice number,
    date, or total amount. Each field carries a confidence score
    and an optional reference to the source layout region.
    """
    field_name: str = Field(
        description="Name of the extracted field (e.g., 'vendor_name', 'invoice_date').",
    )
    value: Any = Field(
        description="The extracted value. Type varies by field (str, float, date string, etc.).",
    )
    confidence: Confidence = Field(
        description="Confidence score for this extraction (0 to 1).",
    )
    source_region_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the layout region this field was extracted from. "
            "Enables traceability back to the source document area."
        ),
    )


class LineItem(BaseModel):
    """
    A single line item row extracted from a document table.

    Represents one item in an invoice or receipt with explicit
    typed fields. Required fields are description and amount;
    all others are optional to accommodate varying document formats.
    """
    description: str = Field(
        description="Description of the item or service.",
    )
    amount: float = Field(
        description="Total amount for this line item.",
    )
    quantity: Optional[float] = Field(
        default=None,
        description="Quantity of items. Optional — not always present on receipts.",
    )
    unit_price: Optional[float] = Field(
        default=None,
        description="Price per unit. Optional — not always present on receipts.",
    )
    item_code: Optional[str] = Field(
        default=None,
        description="Item code or SKU. Optional — varies by document.",
    )
    confidence: Confidence = Field(
        description="Overall confidence score for this line item extraction (0 to 1).",
    )
    source_region_id: Optional[str] = Field(
        default=None,
        description="ID of the layout region (typically a table) this item was extracted from.",
    )


class ExtractionResult(BaseModel):
    """
    Complete output of the LLM structured extraction module.

    Contains scalar fields (key-value pairs), structured line items,
    and metadata about the extraction.
    """
    document_type: DocumentType = Field(
        description="The type of document that was processed.",
    )
    fields: list[ExtractionField] = Field(
        default_factory=list,
        description=(
            "Extracted scalar fields (e.g., vendor name, date, total). "
            "These are top-level document attributes, not line items."
        ),
    )
    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Extracted line items from tables in the document.",
    )
    raw_text: str = Field(
        default="",
        description=(
            "The full text that was sent to the LLM for extraction. "
            "Useful for debugging and auditing."
        ),
    )
    model: str = Field(
        description="Name of the LLM used for extraction (e.g., 'claude-sonnet-4-20250514').",
    )
    page_index: int = Field(
        default=0,
        description="Zero-based page index for multi-page documents.",
    )


# --- LLM Response Schemas ---
# These models define what the LLM should return, regardless of the
# LLM implementation (LangChain, direct API, etc.). They are separate
# from ExtractionResult because the LLM should only fill data fields,
# not metadata like model name or raw text.


class LLMExtractedField(BaseModel):
    """A single extracted key-value pair for the LLM to produce."""
    field_name: str = Field(
        description=(
            "Name of the extracted field "
            "(e.g., 'vendor_name', 'invoice_date', 'total_amount')."
        ),
    )
    value: Optional[str] = Field(
        default=None,
        description=(
            "The extracted value as a string. "
            "For numbers, use the string representation (e.g., '150.00'). "
            "Use null if the field is not found in the document."
        ),
    )
    confidence: float = Field(
        default=0.0,
        description=(
            "Your confidence in this extraction from 0.0 to 1.0. "
            "Use lower values when text is ambiguous or partially illegible."
        ),
    )

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v):
        """Coerce non-string values to strings. Keep None as None."""
        if v is None:
            return None
        return str(v)


class LLMExtractedLineItem(BaseModel):
    """A single line item row for the LLM to produce."""
    description: Optional[str] = Field(
        default="",
        description="Description of the item or service.",
    )
    amount: Optional[float] = Field(
        default=None,
        description="Total amount for this line item.",
    )
    quantity: Optional[float] = Field(
        default=None,
        description="Quantity of items. Use null if not present.",
    )
    unit_price: Optional[float] = Field(
        default=None,
        description="Price per unit. Use null if not present.",
    )
    item_code: Optional[str] = Field(
        default=None,
        description="Item code or SKU. Use null if not present.",
    )
    confidence: float = Field(
        default=0.0,
        description=(
            "Your confidence in this line item extraction from 0.0 to 1.0."
        ),
    )

    @field_validator("description", mode="before")
    @classmethod
    def coerce_description(cls, v):
        """Coerce None description to empty string."""
        if v is None:
            return ""
        return str(v)

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v):
        """Coerce string amounts to float."""
        if v is None:
            return None
        try:
            return float(str(v).replace(",", ""))
        except (ValueError, TypeError):
            return None


class LLMExtractionResponse(BaseModel):
    """
    The structured response schema that the LLM must produce.

    This is the unified contract for any LLM implementation.
    It contains only the data fields the LLM should fill —
    metadata like model name and raw text are added by the
    extractor after the LLM call.
    """
    fields: list[LLMExtractedField] = Field(
        description=(
            "Extracted scalar fields from the document "
            "(e.g., vendor name, date, total)."
        ),
    )
    line_items: list[LLMExtractedLineItem] = Field(
        default_factory=list,
        description="Extracted line items from tables in the document.",
    )


# --- Utilities ---


class ExtractionError(Exception):
    """Raised when extraction fails in a categorizable way."""

    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.error_type = error_type


def parse_llm_json_response(response_text: str) -> LLMExtractionResponse:
    """
    Parse an LLM response into an LLMExtractionResponse.

    Handles common LLM output issues:
    - Markdown fences (```json ... ```)
    - Preamble text before JSON
    - Trailing text after JSON
    - Truncated JSON (attempts repair)
    - Empty responses

    Args:
        response_text: Raw LLM response string.

    Returns:
        Parsed LLMExtractionResponse.

    Raises:
        ExtractionError: With categorized error type:
            'empty_response', 'no_json_found', 'truncated_json',
            'invalid_json', 'schema_validation_error'
    """
    import json
    import logging

    logger = logging.getLogger(__name__)

    if not response_text or not response_text.strip():
        raise ExtractionError(
            "LLM returned an empty response",
            error_type="empty_response",
        )

    text = response_text.strip()

    # Strategy 1: Find JSON between first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1:
        raise ExtractionError(
            f"No JSON object found in response: {text[:200]}",
            error_type="no_json_found",
        )

    if last_brace > first_brace:
        json_str = text[first_brace:last_brace + 1]
        try:
            parsed = json.loads(json_str)
            return LLMExtractionResponse(**parsed)
        except json.JSONDecodeError:
            pass  # Try repair strategies below
        except Exception as e:
            raise ExtractionError(
                f"Schema validation failed: {e}",
                error_type="schema_validation_error",
            )

    # Strategy 2: JSON might be truncated — try to repair
    json_str = text[first_brace:]

    # Close any unclosed brackets/braces
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")

    if open_braces > 0 or open_brackets > 0:
        # Truncated — attempt repair by closing open structures
        repaired = json_str
        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)

        try:
            parsed = json.loads(repaired)
            logger.warning(
                "Repaired truncated JSON (closed %d braces, %d brackets)",
                max(0, open_braces), max(0, open_brackets),
            )
            return LLMExtractionResponse(**parsed)
        except json.JSONDecodeError:
            raise ExtractionError(
                f"JSON appears truncated and could not be repaired. "
                f"Response ends with: ...{text[-100:]}",
                error_type="truncated_json",
            )
        except Exception as e:
            raise ExtractionError(
                f"Schema validation failed on repaired JSON: {e}",
                error_type="schema_validation_error",
            )

    raise ExtractionError(
        f"Could not parse JSON from response: {text[:200]}",
        error_type="invalid_json",
    )
