"""
Sanity check script for DocMind modules.

Settings are read from the .env file by default. CLI arguments
override specific settings when provided.

Usage:
    # OCR only (using settings from .env)
    python sanity_check.py <image_path>

    # OCR + Extraction
    python sanity_check.py <image_path> --extract invoice
    python sanity_check.py <image_path> --extract receipt

    # Override specific settings via CLI
    python sanity_check.py <image_path> --extract invoice --device cpu --conf 0.7

    # Full pipeline with layout + mapping
    python sanity_check.py <image_path> --layout weights/yolo_layout.pt --extract invoice

Requirements:
    - PaddleOCR installed with GPU support
    - For extraction: LangChain + provider SDK + API key in .env
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

from docmind.config.settings import get_settings
from docmind.models.common import DocumentType
from docmind.modules.preprocessing.processor import ImagePreprocessor
from docmind.modules.ocr.paddle_ocr import PaddleOCREngine


# --- Visualization Helpers ---

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

        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)

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

        for text_region in mapped_region.text_regions:
            points = [(int(p.x), int(p.y)) for p in text_region.bbox.corners]
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=1)

    for text_region in mapping_result.unassigned_text_regions:
        points = [(int(p.x), int(p.y)) for p in text_region.bbox.corners]
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    return vis_image


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


# --- Pipeline Steps ---

def run_preprocessing(image_path: Path, settings):
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


def run_ocr(processed_image: np.ndarray, settings):
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


def run_layout(processed_image: np.ndarray, settings):
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

    from collections import Counter
    categories = Counter(r.category.value for r in layout_result.regions)
    print("--- Layout Summary ---")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
    print()

    return layout_result


def run_mapping(ocr_result, layout_result, settings):
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


def run_extraction(ocr_result, document_type: DocumentType, settings):
    """Run LLM extraction and print results."""
    from docmind.modules.extraction.langchain_extractor import LangChainExtractor

    print("--- LLM Extraction ---")
    extractor = LangChainExtractor(settings=settings)

    print(f"Provider:      {settings.provider}")
    print(f"Model:         {settings.model}")
    print(f"Document type: {document_type.value}")
    print("Running extraction...")
    print()

    extraction_result = extractor.extract(ocr_result, document_type)

    print("--- Extracted Fields ---")
    if extraction_result.fields:
        for field in extraction_result.fields:
            print(
                f"  {field.field_name:<20s} "
                f"conf={field.confidence:.2f}  "
                f"value={field.value}"
            )
    else:
        print("  (no fields extracted)")
    print()

    print("--- Extracted Line Items ---")
    if extraction_result.line_items:
        for i, item in enumerate(extraction_result.line_items):
            qty_str = f"qty={item.quantity}" if item.quantity is not None else "qty=n/a"
            price_str = f"price={item.unit_price}" if item.unit_price is not None else "price=n/a"
            code_str = f"code={item.item_code}" if item.item_code is not None else ""
            print(
                f"  [{i}] {item.description[:40]:<40s} "
                f"amount={item.amount:<10.2f} "
                f"{qty_str:<12s} {price_str:<14s} {code_str}"
            )
    else:
        print("  (no line items extracted)")
    print()

    print("--- Extraction Summary ---")
    print(f"Fields extracted:     {len(extraction_result.fields)}")
    print(f"Line items extracted: {len(extraction_result.line_items)}")
    print(f"LLM model used:      {extraction_result.model}")
    if extraction_result.fields:
        avg_conf = sum(f.confidence for f in extraction_result.fields) / len(extraction_result.fields)
        print(f"Avg field confidence: {avg_conf:.3f}")
    if extraction_result.line_items:
        avg_conf = sum(i.confidence for i in extraction_result.line_items) / len(extraction_result.line_items)
        print(f"Avg item confidence:  {avg_conf:.3f}")
    print()

    return extraction_result


def run_vlm_extraction(image_path: Path, document_type: DocumentType, settings):
    """Run VLM extraction directly from image and print results."""
    from docmind.modules.extraction.vlm_extractor import VLMExtractor

    print("--- VLM Extraction (Image -> Structured Data) ---")
    extractor = VLMExtractor(settings=settings)

    print(f"Provider:      {settings.provider}")
    print(f"Model:         {settings.model}")
    print(f"Max image edge: {settings.max_image_long_edge}px")
    print(f"Document type: {document_type.value}")
    print("Running VLM extraction...")
    print()

    extraction_result = extractor.extract_from_image(image_path, document_type)

    print("--- Extracted Fields ---")
    if extraction_result.fields:
        for field in extraction_result.fields:
            print(
                f"  {field.field_name:<20s} "
                f"conf={field.confidence:.2f}  "
                f"value={field.value}"
            )
    else:
        print("  (no fields extracted)")
    print()

    print("--- Extracted Line Items ---")
    if extraction_result.line_items:
        for i, item in enumerate(extraction_result.line_items):
            qty_str = f"qty={item.quantity}" if item.quantity is not None else "qty=n/a"
            price_str = f"price={item.unit_price}" if item.unit_price is not None else "price=n/a"
            code_str = f"code={item.item_code}" if item.item_code is not None else ""
            print(
                f"  [{i}] {item.description[:40]:<40s} "
                f"amount={item.amount:<10.2f} "
                f"{qty_str:<12s} {price_str:<14s} {code_str}"
            )
    else:
        print("  (no line items extracted)")
    print()

    print("--- Extraction Summary ---")
    print(f"Fields extracted:     {len(extraction_result.fields)}")
    print(f"Line items extracted: {len(extraction_result.line_items)}")
    print(f"LLM model used:      {extraction_result.model}")
    if extraction_result.fields:
        avg_conf = sum(f.confidence for f in extraction_result.fields) / len(extraction_result.fields)
        print(f"Avg field confidence: {avg_conf:.3f}")
    if extraction_result.line_items:
        avg_conf = sum(i.confidence for i in extraction_result.line_items) / len(extraction_result.line_items)
        print(f"Avg item confidence:  {avg_conf:.3f}")
    print()

    return extraction_result


# --- Output Saving ---

def save_outputs(image_path, processed_image, ocr_result,
                 layout_result=None, mapping_result=None,
                 extraction_result=None):
    """Save visualization images and JSON results."""
    stem = image_path.stem
    parent = image_path.parent

    ocr_vis = draw_ocr_results(processed_image, ocr_result)
    ocr_vis_path = parent / f"{stem}_ocr_result.jpg"
    cv2.imwrite(str(ocr_vis_path), ocr_vis)
    print(f"OCR visualization:     {ocr_vis_path}")

    ocr_json_path = parent / f"{stem}_ocr_result.json"
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_result.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"OCR JSON result:       {ocr_json_path}")

    if layout_result:
        layout_vis = draw_layout_results(processed_image, layout_result)
        layout_vis_path = parent / f"{stem}_layout_result.jpg"
        cv2.imwrite(str(layout_vis_path), layout_vis)
        print(f"Layout visualization:  {layout_vis_path}")

        layout_json_path = parent / f"{stem}_layout_result.json"
        with open(layout_json_path, "w", encoding="utf-8") as f:
            json.dump(layout_result.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"Layout JSON result:    {layout_json_path}")

    if mapping_result:
        mapping_vis = draw_mapping_results(processed_image, mapping_result)
        mapping_vis_path = parent / f"{stem}_mapping_result.jpg"
        cv2.imwrite(str(mapping_vis_path), mapping_vis)
        print(f"Mapping visualization: {mapping_vis_path}")

        mapping_json_path = parent / f"{stem}_mapping_result.json"
        with open(mapping_json_path, "w", encoding="utf-8") as f:
            json.dump(mapping_result.model_dump(), f, indent=2, ensure_ascii=False)
        print(f"Mapping JSON result:   {mapping_json_path}")

    if extraction_result:
        extraction_json_path = parent / f"{stem}_extraction_result.json"
        with open(extraction_json_path, "w", encoding="utf-8") as f:
            json.dump(
                extraction_result.model_dump(), f,
                indent=2, ensure_ascii=False,
            )
        print(f"Extraction JSON:       {extraction_json_path}")

        summary_path = parent / f"{stem}_extraction_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Document Type: {extraction_result.document_type.value}\n")
            f.write(f"Model: {extraction_result.model}\n")
            f.write(f"\n{'=' * 40}\n")
            f.write("FIELDS\n")
            f.write(f"{'=' * 40}\n")
            for field in extraction_result.fields:
                f.write(f"{field.field_name}: {field.value}\n")
            f.write(f"\n{'=' * 40}\n")
            f.write("LINE ITEMS\n")
            f.write(f"{'=' * 40}\n")
            for item in extraction_result.line_items:
                f.write(f"  {item.description} — {item.amount}\n")
        print(f"Extraction summary:    {summary_path}")


# --- Main ---

def apply_cli_overrides(settings, args):
    """
    Apply CLI argument overrides to settings loaded from .env.

    Only overrides values that were explicitly provided on the
    command line. Returns a new settings object with overrides applied.
    """
    overrides = {}

    # OCR overrides
    if args.device is not None:
        overrides["device"] = args.device
    if args.lang is not None:
        overrides["languages"] = args.lang
    if args.conf is not None:
        overrides["confidence_threshold"] = args.conf

    if overrides:
        return settings.model_copy(update=overrides)
    return settings


def main():
    parser = argparse.ArgumentParser(
        description="DocMind sanity check — test preprocessing, OCR, layout, mapping, and extraction.",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the document image to process.",
    )
    parser.add_argument(
        "--extract",
        type=str,
        default=None,
        choices=["invoice", "receipt"],
        metavar="TYPE",
        help="Document type for OCR+LLM extraction (invoice or receipt).",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default=None,
        choices=["invoice", "receipt"],
        metavar="TYPE",
        help="Document type for VLM extraction — sends image directly to a multimodal LLM, skipping OCR.",
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
        default=None,
        help="Override OCR device (e.g., gpu:0, cpu).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        nargs="+",
        default=None,
        help="Override OCR languages (e.g., ar en).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override confidence threshold.",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="paddleocr",
        choices=["paddleocr", "deepseek-ocr"],
        help="OCR engine to use (default: paddleocr).",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    # Load settings from .env
    settings = get_settings()

    # Determine mode
    is_vlm_mode = args.vlm is not None

    if is_vlm_mode:
        steps = ["VLM Extraction"]
    else:
        steps = ["Preprocessing", "OCR"]
        if args.layout:
            steps.extend(["Layout", "Mapping"])
        if args.extract:
            steps.append("Extraction")

    print(f"{'=' * 60}")
    print("DocMind Sanity Check")
    print(f"{'=' * 60}")
    print(f"Input:    {image_path}")
    print(f"Pipeline: {' -> '.join(steps)}")

    if is_vlm_mode:
        print(f"Mode:     VLM (image -> structured data)")
        print(f"Doc type: {args.vlm}")
        print(f"Provider: {settings.extraction.provider}")
        print(f"Model:    {settings.extraction.model}")
        print()

        # VLM pipeline: image directly to VLM
        document_type = DocumentType(args.vlm)
        extraction_result = run_vlm_extraction(
            image_path, document_type, settings.extraction
        )

        # Save outputs (no OCR results in VLM mode)
        print("--- Saving Outputs ---")
        stem = image_path.stem
        parent = image_path.parent

        extraction_json_path = parent / f"{stem}_vlm_extraction_result.json"
        with open(extraction_json_path, "w", encoding="utf-8") as f:
            json.dump(
                extraction_result.model_dump(), f,
                indent=2, ensure_ascii=False,
            )
        print(f"VLM extraction JSON:   {extraction_json_path}")

        summary_path = parent / f"{stem}_vlm_extraction_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Document Type: {extraction_result.document_type.value}\n")
            f.write(f"Model: {extraction_result.model}\n")
            f.write(f"Pipeline: VLM (direct image extraction)\n")
            f.write(f"\n{'=' * 40}\n")
            f.write("FIELDS\n")
            f.write(f"{'=' * 40}\n")
            for field in extraction_result.fields:
                f.write(f"{field.field_name}: {field.value}\n")
            f.write(f"\n{'=' * 40}\n")
            f.write("LINE ITEMS\n")
            f.write(f"{'=' * 40}\n")
            for item in extraction_result.line_items:
                f.write(f"  {item.description} — {item.amount}\n")
        print(f"VLM extraction summary: {summary_path}")

    else:
        # OCR pipeline
        ocr_engine_name = args.ocr_engine

        if ocr_engine_name == "paddleocr":
            ocr_settings = apply_cli_overrides(settings.ocr, args)
            print(f"OCR engine: PaddleOCR")
            print(f"Device:     {ocr_settings.device}")
            print(f"Languages:  {ocr_settings.languages}")
            print(f"Conf:       {ocr_settings.confidence_threshold}")
        else:
            print(f"OCR engine: DeepSeek-OCR (via Ollama)")

        if args.extract:
            print(f"Doc type:   {args.extract}")
            print(f"Provider:   {settings.extraction.provider}")
            print(f"Model:      {settings.extraction.model}")
        print()

        # Step 1: Preprocessing
        processed_image, metadata = run_preprocessing(
            image_path, settings.preprocessing
        )

        # Step 2: OCR
        if ocr_engine_name == "deepseek-ocr":
            from docmind.modules.ocr.deepseek_ocr import DeepSeekOCREngine

            print("--- OCR (DeepSeek-OCR) ---")
            ds_engine = DeepSeekOCREngine(prompt_mode="markdown")
            print("Running DeepSeek-OCR...")
            ocr_result = ds_engine.recognize(processed_image)
            print(f"Output length: {len(ocr_result.raw_text or '')} characters")
            if ocr_result.raw_text:
                preview = ocr_result.raw_text[:200]
                if len(ocr_result.raw_text) > 200:
                    preview += "..."
                print(f"Preview:\n{preview}")
            print()
        else:
            ocr_settings = apply_cli_overrides(settings.ocr, args)
            ocr_result = run_ocr(processed_image, ocr_settings)

        # Step 3 & 4: Layout + Mapping (optional)
        layout_result = None
        mapping_result = None

        if args.layout:
            layout_settings = settings.layout.model_copy(
                update={"model_path": args.layout}
            )
            if args.conf is not None:
                layout_settings = layout_settings.model_copy(
                    update={"confidence_threshold": args.conf}
                )
            layout_result = run_layout(processed_image, layout_settings)

            mapping_result = run_mapping(
                ocr_result, layout_result, settings.mapping
            )

        # Step 5: Extraction (optional)
        extraction_result = None

        if args.extract:
            document_type = DocumentType(args.extract)
            extraction_result = run_extraction(
                ocr_result, document_type, settings.extraction
            )

        # Save outputs
        print("--- Saving Outputs ---")
        save_outputs(
            image_path, processed_image, ocr_result,
            layout_result, mapping_result, extraction_result,
        )

    print()
    print(f"{'=' * 60}")
    print("Sanity check complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
