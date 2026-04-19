"""Schemas for the preprocessing module output."""

from typing import Any

from pydantic import BaseModel, Field

from docmind.models.common import ImageSize


class PreprocessingOperation(BaseModel):
    """
    A record of a single preprocessing operation applied to an image.

    Stores the operation name and any parameters used, providing
    a traceable log of what transformations were performed.
    """
    name: str = Field(
        description=(
            "Name of the operation applied "
            "(e.g., 'deskew', 'binarize', 'denoise', 'grayscale')."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Operation-specific parameters. "
            "Examples: {'angle': -2.3} for deskew, "
            "{'method': 'adaptive', 'block_size': 11} for binarize."
        ),
    )


class PreprocessingMetadata(BaseModel):
    """
    Metadata output of the preprocessing module.

    Captures what operations were applied and how the image changed.
    The actual processed image (numpy array) is passed separately
    in memory — this schema carries only the metadata.
    """
    original_size: ImageSize = Field(
        description="Dimensions of the input image before preprocessing.",
    )
    processed_size: ImageSize = Field(
        description="Dimensions of the image after preprocessing.",
    )
    operations_applied: list[PreprocessingOperation] = Field(
        default_factory=list,
        description="Ordered list of operations that were applied to the image.",
    )
    was_modified: bool = Field(
        default=False,
        description="Whether the image was modified by any operation.",
    )
