"""Schemas for the region mapping module output."""

from pydantic import BaseModel, Field

from docmind.models.common import ImageSize, ScriptDirection
from docmind.models.layout import LayoutRegion
from docmind.models.ocr import TextRegion


class MappedRegion(BaseModel):
    """
    A layout region enriched with its spatially matched OCR text regions.

    Combines the output of the layout analysis module (structural region)
    with the output of the OCR module (text content), linking them
    through spatial overlap.

    Text regions are sorted in reading order based on the dominant
    script direction of the region.
    """
    layout_region: LayoutRegion = Field(
        description="The source layout region this mapping is based on.",
    )
    text_regions: list[TextRegion] = Field(
        default_factory=list,
        description=(
            "OCR text regions that fall within this layout region, "
            "sorted in reading order."
        ),
    )
    dominant_script_direction: ScriptDirection = Field(
        default=ScriptDirection.LTR,
        description=(
            "The dominant reading direction of text in this region, "
            "determined by the majority script of the contained text regions."
        ),
    )

    @property
    def has_text(self) -> bool:
        """Whether this region contains any mapped text regions."""
        return len(self.text_regions) > 0

    @property
    def full_text(self) -> str:
        """
        Concatenate all text regions into a single string.

        Joins text in reading order with spaces. Useful for passing
        the region's content to the LLM extractor.
        """
        return " ".join(region.text for region in self.text_regions)


class MappingResult(BaseModel):
    """
    Complete output of the region mapping step.

    Contains layout regions enriched with their matched OCR text,
    plus any OCR text regions that could not be assigned to a
    layout region.
    """
    mapped_regions: list[MappedRegion] = Field(
        default_factory=list,
        description="Layout regions with their assigned OCR text regions.",
    )
    unassigned_text_regions: list[TextRegion] = Field(
        default_factory=list,
        description=(
            "OCR text regions that did not fall within any detected "
            "layout region. Passed to the LLM as supplementary context."
        ),
    )
    image_size: ImageSize = Field(
        description="Dimensions of the source image in pixels.",
    )
    page_index: int = Field(
        default=0,
        description="Zero-based page index for multi-page documents.",
    )

    @property
    def empty_regions(self) -> list[MappedRegion]:
        """
        Layout regions that have no text regions assigned to them.

        Useful for debugging — an empty TABLE region may indicate
        a layout detection issue or an overly strict mapping threshold.
        """
        return [r for r in self.mapped_regions if not r.has_text]
