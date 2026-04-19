"""Schemas for the layout analysis module output."""

from enum import Enum

from pydantic import BaseModel, Field

from docmind.models.common import BoundingBox, Confidence, ImageSize


class LayoutCategory(str, Enum):
    """
    Document region categories aligned with the DocLayNet label set.

    Covers the full DocLayNet taxonomy to support future document types
    beyond invoices and receipts.
    """
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    LIST_ITEM = "list_item"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    PICTURE = "picture"
    SECTION_HEADER = "section_header"
    TABLE = "table"
    TEXT = "text"
    TITLE = "title"


class LayoutRegion(BaseModel):
    """
    A single detected structural region in the document.

    Represents a semantically meaningful area of the document
    (e.g., a table, a header, a text block) without any knowledge
    of the text content within it.
    """
    region_id: str = Field(
        description="Unique identifier for this region (e.g., 'region_0').",
    )
    category: LayoutCategory = Field(
        description="The semantic category of this region.",
    )
    bbox: BoundingBox = Field(
        description="Spatial position of the region.",
    )
    confidence: Confidence = Field(
        description="Detection confidence score (0 to 1).",
    )


class LayoutResult(BaseModel):
    """
    Complete output of the layout analysis model for a single page/image.

    Contains all detected structural regions along with metadata
    about the source image and the model that produced the result.
    """
    regions: list[LayoutRegion] = Field(
        default_factory=list,
        description="List of all detected layout regions.",
    )
    image_size: ImageSize = Field(
        description="Dimensions of the source image in pixels.",
    )
    model: str = Field(
        description="Name of the layout model that produced this result (e.g., 'yolov8_doclaynet').",
    )
    page_index: int = Field(
        default=0,
        description="Zero-based page index for multi-page documents.",
    )
