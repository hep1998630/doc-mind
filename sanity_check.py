"""
Sanity check script for DocMind modules.

Usage:
    # OCR only (preprocessing + OCR)
    python sanity_check.py <image_path>

    # Full pipeline (preprocessing + OCR + layout + mapping)
    python sanity_check.py <image_path> --layout <model_path>

Examples:
    python sanity_check.py samples/invoices/sample_invoice.jpg
    python sanity_check.py samples/invoices/sample_invoice.jpg --layout weights/yolo_layout.pt

Requirements:
    - PaddleOCR installed with GPU support
    - For layout mode: ultralytics installed + YOLO model weights
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path so we can import docmind
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docmind.config.settings import (
    LayoutSettings,
    MappingSettings,
    OCRSettings,
    PreprocessingSettings,
)
from docmind.modules.preprocessing.processor import ImagePreprocessor
from docmind.modules.ocr.paddle_ocr import PaddleOCREngine


# --- Visualization Helpers ---

# Color palette for layout categories
CATEGORY_COLORS = {
    "caption": (128, 0, 128),
    "footnote": (128, 128, 0),
    "formula": (0, 128, 128),
    "list_item": (64, 128, 255),
    "page_footer": (128, 64, 0),
    "page_header": (0, 64, 128),
    "picture": (192, 192, 192),
    "section_header": (0, 165, 255),
    "table": (0, 0, 255),
    "text": (0, 200, 0),
    "title": (255, 0, 255),
}


def draw_ocr_results(image: np.ndarray, ocr_result) -> np.ndarray:
    """Draw OCR bounding boxes and text labels on the image."""
    vis_image = _ensure_bgr(image)

    for region in ocr_result.text_regions:
        points = [(int(p.x), int(p.y)) for p in region.bbox.corners]
        pts = np.array(points, dtype=np.int32)

        color = (0, 255, 0) if region.script_direction.value == "ltr" else (255, 0, 0)

        cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=2)

        label = f"{region.text[:30]} ({region.confidence:.2f})"
        label_pos = (int(region.bbox.top_left.x), int(region.bbox.top_left.y) - 5)
        cv2.putText(
            vis_image, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return vis_image


def draw_layout_results(image: np.ndarray, layout_result) -> np.ndarray:
    """Draw layout region bounding boxes with category labels."""
    vis_image = _ensure_bgr(image)

    for region in layout_result.regions:
        x1, y1, x2, y2 = region.bbox.to_xyxy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = CATEGORY_COLORS.get(region.category.value, (255, 255, 255))

        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

        label = f"{region.category.value} ({region.confidence:.2f})"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            vis_image, (x1, y1 - label_size[1] - 8), (x1 + label_size[0], y1),
            color, -1,
        )
        cv2.putText(
            vis_image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return vis_image


def draw_mapping_results(image: np.ndarray, mapping_result) -> np.ndarray:
    """Draw layout regions with their mapped OCR text inside."""
    vis_image = _ensure_bgr(image)

    for mapped_region in mapping_result.mapped_regions:
        lr = mapped_region.layout_region
        x1, y1, x2, y2 = lr.bbox.to_xyxy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = CATEGORY_COLORS.get(lr.category.value, (255, 255, 255))

        # Draw layout region
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

        # Label with category and text count
        label = f"{lr.category.value} [{len(mapped_region.text_regions)} texts]"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            vis_image, (x1, y1 - label_size[1] - 8), (x1 + label_size[0], y1),
            color, -1,
        )
        cv2.putText(
            vis_image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # Draw contained text regions in lighter color
        for text_region in mapped_region.text_regions:
            points = [(int(p.x), int(p.y)) for p in text_region.bbox.corners]
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=1)

    # Draw unassigned text regions in red
    for text_region in mapping_result.unassigned_text_regions:
        points = [(int(p.x), int(p.y)) for p in text_region.bbox.corners]
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    return vis_image


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


# --- Main ---

def run_preprocessing(image_path: Path, settings: PreprocessingSettings):
    """Run preprocessing and print results."""
    print("--- Preprocessing ---")
    preprocessor = ImagePreprocessor(settings=settings)
    processed_image, metadata = preprocessor.process(str(image_path))

    print(f"Original size:  {metadata.original_size.width} x {metadata.original_size.height}")
    print(f"Processed size: {metadata.processed_size.width} x {metadata.processed_size.height}")
    print(f"Was modified:   {metadata.was_modified}")
    print(f"Operations applied:")
    for op in metadata.operations_applied:
        print(f"  - {op.name}: {op.parameters}")
    print()

    return processed_image, metadata


def run_ocr(processed_image: np.ndarray, settings: OCRSettings):
    """Run OCR and print results."""
    print("--- OCR (PaddleOCR) ---")
    ocr_engine = PaddleOCREngine(settings=settings)

    print(f"Engine: {ocr_engine.engine_name}")
    print(f"Supported languages: {ocr_engine.supported_languages}")
    print("Running OCR...")
    print()

    ocr_result = ocr_engine.recognize(processed_image)

    print(f"Detected {len(ocr_result.text_regions)} text regions")
    print()

    print("--- Detected Text Regions ---")
    for i, region in enumerate(ocr_result.text_regions):
        center = region.bbox.center
        print(
            f"  [{i:3d}] "
            f"({region.script_direction.value.upper():>3s}) "
            f"conf={region.confidence:.3f}  "
            f"pos=({center.x:.0f}, {center.y:.0f})  "
            f'text="{region.text}"'
        )
    print()

    ltr_regions = [r for r in ocr_result.text_regions if r.script_direction.value == "ltr"]
    rtl_regions = [r for r in ocr_result.text_regions if r.script_direction.value == "rtl"]
    print("--- OCR Summary ---")
    print(f"Total regions: {len(ocr_result.text_regions)}")
    print(f"LTR (Latin):   {len(ltr_regions)}")
    print(f"RTL (Arabic):  {len(rtl_regions)}")
    if ocr_result.text_regions:
        avg_conf = sum(r.confidence for r in ocr_result.text_regions) / len(ocr_result.text_regions)
        print(f"Avg confidence: {avg_conf:.3f}")
    print()

    return ocr_result


def run_layout(processed_image: np.ndarray, settings: LayoutSettings):
    """Run layout analysis and print results."""
    from docmind.modules.layout.yolo_layout import YOLOLayoutAnalyzer

    print("--- Layout Analysis (YOLO) ---")
    layout_analyzer = YOLOLayoutAnalyzer(settings=settings)

    print(f"Model: {settings.model_path}")
    print("Running layout analysis...")
    print()

    layout_result = layout_analyzer.analyze(processed_image)

    print(f"Detected {len(layout_result.regions)} layout regions")
    print()

    print("--- Detected Layout Regions ---")
    for region in layout_result.regions:
        x1, y1, x2, y2 = region.bbox.to_xyxy()
        print(
            f"  [{region.region_id}] "
            f"{region.category.value:<16s} "
            f"conf={region.confidence:.3f}  "
            f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )
    print()

    # Category breakdown
    from collections import Counter
    categories = Counter(r.category.value for r in layout_result.regions)
    print("--- Layout Summary ---")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
    print()

    return layout_result


def run_mapping(ocr_result, layout_result, settings: MappingSettings):
    """Run region mapping and print results."""
    from docmind.modules.mapping.region_mapper import RegionMapper

    print("--- Region Mapping ---")
    mapper = RegionMapper(settings=settings)
    mapping_result = mapper.map(ocr_result, layout_result)

    print(f"Mapped regions:     {len(mapping_result.mapped_regions)}")
    print(f"  With text:        {len(mapping_result.mapped_regions) - len(mapping_result.empty_regions)}")
    print(f"  Empty:            {len(mapping_result.empty_regions)}")
    print(f"Unassigned texts:   {len(mapping_result.unassigned_text_regions)}")
    print()

    print("--- Mapped Regions Detail ---")
    for mapped in mapping_result.mapped_regions:
        lr = mapped.layout_region
        direction = mapped.dominant_script_direction.value.upper()
        status = "" if mapped.has_text else "  [EMPTY]"
        print(
            f"  [{lr.region_id}] {lr.category.value:<16s} "
            f"({direction})  "
            f"texts={len(mapped.text_regions)}{status}"
        )
        if mapped.has_text:
            # Show first 80 chars of combined text
            preview = mapped.full_text[:80]
            if len(mapped.full_text) > 80:
                preview += "..."
            print(f"           \"{preview}\"")
    print()

    if mapping_result.empty_regions:
        print("--- Empty Layout Regions (no text assigned) ---")
        for mapped in mapping_result.empty_regions:
            lr = mapped.layout_region
            x1, y1, x2, y2 = lr.bbox.to_xyxy()
            print(
                f"  [{lr.region_id}] {lr.category.value:<16s} "
                f"conf={lr.confidence:.3f}  "
                f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )
        print()

    if mapping_result.unassigned_text_regions:
        print("--- Unassigned Text Regions ---")
        for region in mapping_result.unassigned_text_regions:
            print(f'  "{region.text}" (conf={region.confidence:.3f})')
        print()

    return mapping_result


def save_outputs(image_path, processed_image, ocr_result,
                 layout_result=None, mapping_result=None):
    """Save visualization images and JSON results."""
    stem = image_path.stem
    parent = image_path.parent

    # OCR visualization
    ocr_vis = draw_ocr_results(processed_image, ocr_result)
    ocr_vis_path = parent / f"{stem}_ocr_result.jpg"
    cv2.imwrite(str(ocr_vis_path), ocr_vis)
    print(f"OCR visualization:     {ocr_vis_path}")

    # OCR JSON
    ocr_json_path = parent / f"{stem}_ocr_result.json"
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"OCR JSON result:       {ocr_json_path}")

    if layout_result:
        # Layout visualization
        layout_vis = draw_layout_results(processed_image, layout_result)
        layout_vis_path = parent / f"{stem}_layout_result.jpg"
        cv2.imwrite(str(layout_vis_path), layout_vis)
        print(f"Layout visualization:  {layout_vis_path}")

        # Layout JSON
        layout_json_path = parent / f"{stem}_layout_result.json"
        with open(layout_json_path, "w", encoding="utf-8") as f:
            json.dump(layout_result.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"Layout JSON result:    {layout_json_path}")

    if mapping_result:
        # Mapping visualization
        mapping_vis = draw_mapping_results(processed_image, mapping_result)
        mapping_vis_path = parent / f"{stem}_mapping_result.jpg"
        cv2.imwrite(str(mapping_vis_path), mapping_vis)
        print(f"Mapping visualization: {mapping_vis_path}")

        # Mapping JSON
        mapping_json_path = parent / f"{stem}_mapping_result.json"
        with open(mapping_json_path, "w", encoding="utf-8") as f:
            json.dump(mapping_result.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"Mapping JSON result:   {mapping_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DocMind sanity check — test preprocessing, OCR, layout, and mapping."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the document image to process.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        metavar="MODEL_PATH",
        help="Path to YOLO layout model weights. Enables layout + mapping steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu:0",
        help="Device for inference (default: gpu:0).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        nargs="+",
        default=["ar", "en"],
        help="OCR languages (default: ar en).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5).",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"DocMind Sanity Check")
    print(f"{'=' * 60}")
    print(f"Input:  {image_path}")
    print(f"Mode:   {'Full pipeline (OCR + Layout + Mapping)' if args.layout else 'OCR only'}")
    print(f"Device: {args.device}")
    print()

    # Step 1: Preprocessing
    preprocessing_settings = PreprocessingSettings(
        enable_deskew=True,
        enable_denoise=True,
        enable_binarize=False,
    )
    processed_image, metadata = run_preprocessing(image_path, preprocessing_settings)

    # Step 2: OCR
    ocr_settings = OCRSettings(
        languages=args.lang,
        confidence_threshold=args.conf,
        device=args.device,
    )
    ocr_result = run_ocr(processed_image, ocr_settings)

    # Step 3 & 4: Layout + Mapping (optional)
    layout_result = None
    mapping_result = None

    if args.layout:
        layout_settings = LayoutSettings(
            model_path=args.layout,
            confidence_threshold=args.conf,
        )
        layout_result = run_layout(processed_image, layout_settings)

        mapping_settings = MappingSettings()
        mapping_result = run_mapping(ocr_result, layout_result, mapping_settings)

    # Save outputs
    print("--- Saving Outputs ---")
    save_outputs(
        image_path, processed_image, ocr_result,
        layout_result, mapping_result,
    )

    print()
    print(f"{'=' * 60}")
    print("Sanity check complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
