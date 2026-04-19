"""Shared types and base schemas used across all DocMind modules."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, confloat


# --- Constrained Types ---

Confidence = Annotated[float, confloat(ge=0.0, le=1.0)] 
"""A confidence score between 0 and 1."""


# --- Enums ---

class ScriptDirection(str, Enum):
    """Text reading direction."""
    LTR = "ltr"
    RTL = "rtl"


class DocumentType(str, Enum):
    """Supported document types for structured extraction."""
    INVOICE = "invoice"
    RECEIPT = "receipt"


# --- Core Geometric Types ---

class Point(BaseModel):
    """A 2D coordinate point."""
    x: float
    y: float


class ImageSize(BaseModel):
    """Image dimensions in pixels."""
    width: int
    height: int


class BoundingBox(BaseModel):
    """
    A quadrilateral bounding box defined by four corner points.

    Points are ordered clockwise starting from the top-left:
        top_left → top_right → bottom_right → bottom_left

    This representation handles both axis-aligned and rotated text regions.
    """
    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        """
        Construct from axis-aligned (x, y, width, height) format.

        Args:
            x: Top-left x coordinate.
            y: Top-left y coordinate.
            w: Width.
            h: Height.
        """
        return cls(
            top_left=Point(x=x, y=y),
            top_right=Point(x=x + w, y=y),
            bottom_right=Point(x=x + w, y=y + h),
            bottom_left=Point(x=x, y=y + h),
        )

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        """
        Construct from (x1, y1, x2, y2) format — top-left and bottom-right corners.

        Args:
            x1: Top-left x coordinate.
            y1: Top-left y coordinate.
            x2: Bottom-right x coordinate.
            y2: Bottom-right y coordinate.
        """
        return cls(
            top_left=Point(x=x1, y=y1),
            top_right=Point(x=x2, y=y1),
            bottom_right=Point(x=x2, y=y2),
            bottom_left=Point(x=x1, y=y2),
        )

    @classmethod
    def from_points_list(cls, points: list[list[float]]) -> "BoundingBox":
        """
        Construct from a list of four [x, y] pairs.

        This matches PaddleOCR's native output format:
            [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

        Args:
            points: List of four [x, y] coordinate pairs, clockwise from top-left.
        """
        if len(points) != 4:
            raise ValueError(f"Expected 4 points, got {len(points)}.")

        return cls(
            top_left=Point(x=points[0][0], y=points[0][1]),
            top_right=Point(x=points[1][0], y=points[1][1]),
            bottom_right=Point(x=points[2][0], y=points[2][1]),
            bottom_left=Point(x=points[3][0], y=points[3][1]),
        )

    @property
    def center(self) -> Point:
        """Compute the center point of the bounding box."""
        cx = (self.top_left.x + self.top_right.x + self.bottom_right.x + self.bottom_left.x) / 4
        cy = (self.top_left.y + self.top_right.y + self.bottom_right.y + self.bottom_left.y) / 4
        return Point(x=cx, y=cy)

    @property
    def area(self) -> float:
        """
        Compute the area using the Shoelace formula.

        Works correctly for both axis-aligned and rotated quadrilaterals.
        """
        points = [self.top_left, self.top_right, self.bottom_right, self.bottom_left]
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y
        return abs(area) / 2.0

    @property
    def corners(self) -> list[Point]:
        """Return the four corners as a list, clockwise from top-left."""
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """
        Convert to axis-aligned (x1, y1, x2, y2) format.

        For rotated boxes, returns the axis-aligned bounding rectangle
        that fully contains the rotated box.
        """
        xs = [p.x for p in self.corners]
        ys = [p.y for p in self.corners]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_xywh(self) -> tuple[float, float, float, float]:
        """
        Convert to axis-aligned (x, y, width, height) format.

        For rotated boxes, returns the axis-aligned bounding rectangle
        that fully contains the rotated box.
        """
        x1, y1, x2, y2 = self.to_xyxy()
        return (x1, y1, x2 - x1, y2 - y1)
