"""
Sanity check script for the preprocessing and OCR modules.

Usage:
    python sanity_check.py <image_path>
    python sanity_check.py samples/invoices/sample_invoice.jpg

This script runs an image through the preprocessing and OCR pipeline,
prints detailed results, and optionally saves a visualization with
bounding boxes drawn on the image.

Requirements:
    - PaddleOCR installed (pip install paddleocr)
    - A sample document image (invoice, receipt, etc.)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path so we can import docmind
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docmind.config.settings import OCRSettings, PreprocessingSettings
from docmind.modules.preprocessing.processor import ImagePreprocessor
from docmind.modules.ocr.paddle_ocr import PaddleOCREngine


def draw_results(image: np.ndarray, ocr_result) -> np.ndarray:
    """Draw bounding boxes and text labels on the image."""
    # Convert grayscale to BGR for colored annotations
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    for region in ocr_result.text_regions:
        # Get corner points
        points = [
            (int(p.x), int(p.y)) for p in region.bbox.corners
        ]
        pts = np.array(points, dtype=np.int32)

        # Color by script direction: green for LTR, blue for RTL
        color = (0, 255, 0) if region.script_direction.value == "ltr" else (255, 0, 0)

        # Draw polygon
        cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=2)

        # Draw label
        label = f"{region.text[:30]} ({region.confidence:.2f})"
        label_pos = (int(region.bbox.top_left.x), int(region.bbox.top_left.y) - 5)
        cv2.putText(
            vis_image, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    return vis_image


def main():
    if len(sys.argv) < 2:
        print("Usage: python sanity_check.py <image_path>")
        print("Example: python sanity_check.py samples/invoices/sample_invoice.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"DocMind Sanity Check")
    print(f"{'=' * 60}")
    print(f"Input: {image_path}")
    print()

    # --- Step 1: Preprocessing ---
    print(f"--- Preprocessing ---")
    preprocessing_settings = PreprocessingSettings(
        enable_deskew=True,
        enable_denoise=True,
        enable_binarize=False,
    )
    preprocessor = ImagePreprocessor(settings=preprocessing_settings)
    processed_image, metadata = preprocessor.process(str(image_path))

    print(f"Original size:  {metadata.original_size.width} x {metadata.original_size.height}")
    print(f"Processed size: {metadata.processed_size.width} x {metadata.processed_size.height}")
    print(f"Was modified:   {metadata.was_modified}")
    print(f"Operations applied:")
    for op in metadata.operations_applied:
        print(f"  - {op.name}: {op.parameters}")
    print()

    # --- Step 2: OCR ---
    print(f"--- OCR (PaddleOCR) ---")
    ocr_settings = OCRSettings(
        languages=["ar", "en"],
        confidence_threshold=0.5,
        device="gpu:0",
    )
    ocr_engine = PaddleOCREngine(settings=ocr_settings)

    print(f"Engine: {ocr_engine.engine_name}")
    print(f"Supported languages: {ocr_engine.supported_languages}")
    print(f"Running OCR...")
    print()

    ocr_result = ocr_engine.recognize(processed_image)

    print(f"Detected {len(ocr_result.text_regions)} text regions")
    print()

    # --- Print Results ---
    print(f"--- Detected Text Regions ---")
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

    # --- Summary by Script Direction ---
    ltr_regions = [r for r in ocr_result.text_regions if r.script_direction.value == "ltr"]
    rtl_regions = [r for r in ocr_result.text_regions if r.script_direction.value == "rtl"]
    print(f"--- Summary ---")
    print(f"Total regions: {len(ocr_result.text_regions)}")
    print(f"LTR (Latin):   {len(ltr_regions)}")
    print(f"RTL (Arabic):  {len(rtl_regions)}")

    if ocr_result.text_regions:
        avg_confidence = sum(r.confidence for r in ocr_result.text_regions) / len(ocr_result.text_regions)
        print(f"Avg confidence: {avg_confidence:.3f}")
    print()

    # --- Save Visualization ---
    output_path = image_path.parent / f"{image_path.stem}_ocr_result.jpg"
    vis_image = draw_results(processed_image, ocr_result)
    cv2.imwrite(str(output_path), vis_image)
    print(f"Visualization saved to: {output_path}")

    # --- Save JSON ---
    json_output_path = image_path.parent / f"{image_path.stem}_ocr_result.json"
    result_dict = ocr_result.model_dump()
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f"JSON result saved to: {json_output_path}")

    print()
    print(f"{'=' * 60}")
    print("Sanity check complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
