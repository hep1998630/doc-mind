"""Region mapper that spatially matches OCR text regions to layout regions."""

import logging

import numpy as np

from docmind.config.settings import MappingSettings, get_settings
from docmind.models.common import BoundingBox, Point, ScriptDirection
from docmind.models.layout import LayoutResult
from docmind.models.mapping import MappedRegion, MappingResult
from docmind.models.ocr import OCRResult, TextRegion

logger = logging.getLogger(__name__)


class RegionMapper:
    """
    Maps OCR text regions to layout regions based on spatial overlap.

    Takes the independent outputs of the OCR and layout modules and
    combines them by determining which text regions fall within which
    layout regions. When a text region overlaps with multiple layout
    regions, it is assigned to the largest one (by area). Text regions
    within each layout region are sorted in reading order based on
    the dominant script direction.

    Args:
        settings: Mapping configuration. If None, loads from the
            application settings.
    """

    # Multiplier for average text height to determine line grouping.
    # Two text regions whose y-centers differ by less than this
    # fraction of the average text height are on the same line.
    LINE_GROUP_THRESHOLD = 0.5

    def __init__(self, settings: MappingSettings | None = None) -> None:
        self._settings = settings or get_settings().mapping

    def map(
        self, ocr_result: OCRResult, layout_result: LayoutResult
    ) -> MappingResult:
        """
        Combine OCR and layout results by spatial matching.

        For each text region, finds all overlapping layout regions and
        assigns the text to the largest one. Layout regions with no
        assigned text are preserved as empty mapped regions.

        Args:
            ocr_result: Output from the OCR module.
            layout_result: Output from the layout analysis module.

        Returns:
            MappingResult containing layout regions enriched with
            their matched text regions, plus any unassigned text.

        Raises:
            ValueError: If OCR and layout results have different
                image dimensions.
        """
        self._validate_dimensions(ocr_result, layout_result)

        # For each layout region, collect its assigned text regions
        region_texts: dict[str, list[TextRegion]] = {
            lr.region_id: [] for lr in layout_result.regions
        }
        unassigned: list[TextRegion] = []

        # Precompute layout region areas for conflict resolution
        region_areas: dict[str, float] = {
            lr.region_id: lr.bbox.area for lr in layout_result.regions
        }

        # For each text region, find the best matching layout region
        for text_region in ocr_result.text_regions:
            best_region_id = self._find_best_match(
                text_region, layout_result, region_areas
            )
            if best_region_id is not None:
                region_texts[best_region_id].append(text_region)
            else:
                unassigned.append(text_region)

        # Build mapped regions, preserving layout region order
        mapped_regions: list[MappedRegion] = []
        for layout_region in layout_result.regions:
            texts = region_texts[layout_region.region_id]

            dominant_direction = self._get_dominant_direction(texts)
            sorted_texts = self._sort_reading_order(texts, dominant_direction)

            mapped_region = MappedRegion(
                layout_region=layout_region,
                text_regions=sorted_texts,
                dominant_script_direction=dominant_direction,
            )
            mapped_regions.append(mapped_region)

        # Log summary
        filled = sum(1 for m in mapped_regions if m.has_text)
        empty = len(mapped_regions) - filled
        logger.info(
            "Mapping complete: %d regions (%d with text, %d empty), "
            "%d unassigned text regions",
            len(mapped_regions), filled, empty, len(unassigned),
        )

        return MappingResult(
            mapped_regions=mapped_regions,
            unassigned_text_regions=unassigned,
            image_size=ocr_result.image_size,
            page_index=ocr_result.page_index,
        )

    # --- Private: Matching Logic ---

    def _find_best_match(
        self,
        text_region: TextRegion,
        layout_result: LayoutResult,
        region_areas: dict[str, float],
    ) -> str | None:
        """
        Find the best matching layout region for a text region.

        If using center containment, returns the first region whose
        bounding box contains the text center. If using overlap,
        returns the largest region that exceeds the overlap threshold.

        Args:
            text_region: The text region to match.
            layout_result: All layout regions.
            region_areas: Precomputed areas for each layout region.

        Returns:
            The region_id of the best match, or None if no match.
        """
        if self._settings.use_center_containment:
            return self._find_match_by_center(text_region, layout_result)
        else:
            return self._find_match_by_overlap(
                text_region, layout_result, region_areas
            )

    def _find_match_by_center(
        self,
        text_region: TextRegion,
        layout_result: LayoutResult,
    ) -> str | None:
        """Find the first layout region containing the text center."""
        center = text_region.bbox.center
        for lr in layout_result.regions:
            if self._point_inside_bbox(center, lr.bbox):
                return lr.region_id
        return None

    def _find_match_by_overlap(
        self,
        text_region: TextRegion,
        layout_result: LayoutResult,
        region_areas: dict[str, float],
    ) -> str | None:
        """
        Find the largest layout region that overlaps sufficiently.

        Computes overlap ratio with all layout regions, filters by
        threshold, and returns the largest matching region.
        """
        candidates: list[tuple[str, float]] = []

        for lr in layout_result.regions:
            overlap = self._compute_overlap_ratio(text_region.bbox, lr.bbox)
            if overlap >= self._settings.overlap_threshold:
                candidates.append((lr.region_id, region_areas[lr.region_id]))

        if not candidates:
            return None

        # Resolve conflicts: pick the largest region
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    @staticmethod
    def _point_inside_bbox(point: Point, bbox: BoundingBox) -> bool:
        """Check if a point falls inside an axis-aligned bounding box."""
        x1, y1, x2, y2 = bbox.to_xyxy()
        return x1 <= point.x <= x2 and y1 <= point.y <= y2

    @staticmethod
    def _compute_overlap_ratio(
        text_bbox: BoundingBox, layout_bbox: BoundingBox
    ) -> float:
        """
        Compute the overlap ratio between a text region and a layout region.

        The ratio is: intersection_area / text_region_area.
        This measures what fraction of the text region is covered by
        the layout region.

        Args:
            text_bbox: Bounding box of the text region.
            layout_bbox: Bounding box of the layout region.

        Returns:
            Overlap ratio between 0 and 1.
        """
        tx1, ty1, tx2, ty2 = text_bbox.to_xyxy()
        lx1, ly1, lx2, ly2 = layout_bbox.to_xyxy()

        # Compute intersection
        ix1 = max(tx1, lx1)
        iy1 = max(ty1, ly1)
        ix2 = min(tx2, lx2)
        iy2 = min(ty2, ly2)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        text_area = text_bbox.area

        if text_area == 0:
            return 0.0

        return intersection_area / text_area

    # --- Private: Reading Order ---

    @staticmethod
    def _get_dominant_direction(
        text_regions: list[TextRegion],
    ) -> ScriptDirection:
        """
        Determine the dominant script direction from a list of text regions.

        Counts RTL vs LTR regions and returns the majority. Defaults
        to LTR if empty or tied.
        """
        if not text_regions:
            return ScriptDirection.LTR

        rtl_count = sum(
            1 for r in text_regions
            if r.script_direction == ScriptDirection.RTL
        )

        if rtl_count > len(text_regions) / 2:
            return ScriptDirection.RTL

        return ScriptDirection.LTR

    def _sort_reading_order(
        self,
        text_regions: list[TextRegion],
        direction: ScriptDirection,
    ) -> list[TextRegion]:
        """
        Sort text regions in reading order.

        Groups regions into lines based on vertical position (y-center),
        then sorts within each line by horizontal position — left-to-right
        for LTR, right-to-left for RTL.

        Lines are determined by grouping text regions whose y-centers
        are within LINE_GROUP_THRESHOLD * average_text_height of each
        other.

        Args:
            text_regions: List of text regions to sort.
            direction: Reading direction for horizontal sorting.

        Returns:
            Text regions sorted in reading order.
        """
        if len(text_regions) <= 1:
            return list(text_regions)

        # Compute average text height for line grouping threshold
        avg_height = np.mean([
            abs(r.bbox.bottom_left.y - r.bbox.top_left.y)
            for r in text_regions
        ])
        line_threshold = avg_height * self.LINE_GROUP_THRESHOLD

        # Sort by y-center first (top to bottom)
        regions_with_centers = [
            (r, r.bbox.center) for r in text_regions
        ]
        regions_with_centers.sort(key=lambda x: x[1].y)

        # Group into lines
        lines: list[list[tuple[TextRegion, Point]]] = []
        current_line: list[tuple[TextRegion, Point]] = [
            regions_with_centers[0]
        ]

        for region, center in regions_with_centers[1:]:
            # Compare against the average y of the current line,
            # not just the previous element, for more stable grouping
            current_line_avg_y = np.mean([c.y for _, c in current_line])
            if abs(center.y - current_line_avg_y) <= line_threshold:
                current_line.append((region, center))
            else:
                lines.append(current_line)
                current_line = [(region, center)]
        lines.append(current_line)

        # Sort within each line by x-position
        reverse = direction == ScriptDirection.RTL
        sorted_regions: list[TextRegion] = []
        for line in lines:
            line.sort(key=lambda x: x[1].x, reverse=reverse)
            sorted_regions.extend(r for r, _ in line)

        return sorted_regions

    # --- Private: Validation ---

    @staticmethod
    def _validate_dimensions(
        ocr_result: OCRResult, layout_result: LayoutResult
    ) -> None:
        """
        Verify that OCR and layout results refer to the same image.

        Raises:
            ValueError: If image dimensions don't match.
        """
        if ocr_result.image_size != layout_result.image_size:
            raise ValueError(
                f"Image size mismatch between OCR and layout results. "
                f"OCR: {ocr_result.image_size.width}x"
                f"{ocr_result.image_size.height}, "
                f"Layout: {layout_result.image_size.width}x"
                f"{layout_result.image_size.height}. "
                f"Both modules must receive the same preprocessed image."
            )
