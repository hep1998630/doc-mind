"""Schemas for the LLM structured extraction module output."""

from typing import Any, Optional

from pydantic import BaseModel, Field

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
