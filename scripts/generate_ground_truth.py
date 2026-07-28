"""
Ground Truth Generation Script for DocMind.

Selects invoice images with the most ground truth transcriptions,
fills gaps with OCR output, sends the combined text to a large LLM
for structured extraction, and saves the result for human review.

Usage:
    python scripts/generate_ground_truth.py <image_dir> <annotation_dir> <output_dir>
    python scripts/generate_ground_truth.py images/ annotations/ ground_truth/ --count 40
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docmind.config.settings import get_settings
from docmind.modules.preprocessing.processor import ImagePreprocessor
from docmind.modules.ocr.paddle_ocr import PaddleOCREngine

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Minimum number of text blocks required to use OCR+LLM approach.
# Below this, the script falls back to VLM (image-based extraction).
MIN_TEXT_BLOCKS = 4


@dataclass
class TextBlock:
    """A text block with its source and spatial position."""
    text: str
    x_center: float
    y_center: float
    bbox: tuple
    source: str  # "ground_truth" or "ocr"
    class_title: str = ""


def parse_annotation(annotation_path):
    """Parse annotation file, separate transcribed from untranscribed."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcribed = []
    untranscribed_bboxes = []

    for obj in data.get("objects", []):
        class_title = obj.get("classTitle", "")
        if class_title.lower() == "page":
            continue
        if obj.get("geometryType") != "rectangle":
            continue

        points = obj.get("points", {}).get("exterior", [])
        if len(points) != 2:
            continue

        x1, y1 = float(points[0][0]), float(points[0][1])
        x2, y2 = float(points[1][0]), float(points[1][1])

        transcription = None
        for tag in obj.get("tags", []):
            if tag.get("name") == "Transcription":
                transcription = tag.get("value", "")
                break

        if transcription:
            transcribed.append(TextBlock(
                text=transcription,
                x_center=(x1 + x2) / 2,
                y_center=(y1 + y2) / 2,
                bbox=(x1, y1, x2, y2),
                source="ground_truth",
                class_title=class_title,
            ))
        else:
            untranscribed_bboxes.append({
                "bbox": (x1, y1, x2, y2),
                "class_title": class_title,
            })

    return {
        "transcribed": transcribed,
        "untranscribed_bboxes": untranscribed_bboxes,
        "image_size": data.get("size", {}),
    }


def compute_containment(inner, outer):
    """What fraction of inner box is inside outer box."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])

    if inner_area == 0:
        return 0.0

    return intersection / inner_area


def fill_with_ocr(annotation_data, ocr_result, containment_threshold=0.5):
    """Combine GT transcriptions with OCR for untranscribed regions."""
    gt_transcribed = annotation_data["transcribed"]
    untranscribed_bboxes = annotation_data["untranscribed_bboxes"]

    all_blocks = list(gt_transcribed)
    ocr_used = set()

    ocr_regions = []
    for region in ocr_result.text_regions:
        ocr_bbox = region.bbox.to_xyxy()
        ocr_regions.append({
            "bbox": ocr_bbox,
            "text": region.text,
            "x_center": region.bbox.center.x,
            "y_center": region.bbox.center.y,
        })

    # Fill untranscribed GT regions with OCR text
    for gt_bbox_info in untranscribed_bboxes:
        gt_bbox = gt_bbox_info["bbox"]
        contained_ocr = []

        for i, ocr_reg in enumerate(ocr_regions):
            if i in ocr_used:
                continue
            containment = compute_containment(ocr_reg["bbox"], gt_bbox)
            if containment >= containment_threshold:
                contained_ocr.append((i, ocr_reg))

        if contained_ocr:
            contained_ocr.sort(key=lambda x: (x[1]["y_center"], x[1]["x_center"]))
            combined_text = " ".join(r["text"] for _, r in contained_ocr)
            x_center = (gt_bbox[0] + gt_bbox[2]) / 2
            y_center = (gt_bbox[1] + gt_bbox[3]) / 2

            all_blocks.append(TextBlock(
                text=combined_text,
                x_center=x_center,
                y_center=y_center,
                bbox=gt_bbox,
                source="ocr",
                class_title=gt_bbox_info["class_title"],
            ))

            for idx, _ in contained_ocr:
                ocr_used.add(idx)

    # Collect remaining unassigned OCR text
    for i, ocr_reg in enumerate(ocr_regions):
        if i not in ocr_used:
            overlaps_gt = False
            for gt_block in gt_transcribed:
                containment = compute_containment(ocr_reg["bbox"], gt_block.bbox)
                if containment >= containment_threshold:
                    overlaps_gt = True
                    break

            if not overlaps_gt:
                all_blocks.append(TextBlock(
                    text=ocr_reg["text"],
                    x_center=ocr_reg["x_center"],
                    y_center=ocr_reg["y_center"],
                    bbox=ocr_reg["bbox"],
                    source="ocr",
                    class_title="unknown",
                ))

    all_blocks.sort(key=lambda b: (b.y_center, b.x_center))
    return all_blocks


def format_text_for_llm(text_blocks):
    """Format text blocks with coordinates and source markers."""
    lines = []
    for block in text_blocks:
        source_tag = "GT" if block.source == "ground_truth" else "OCR"
        lines.append(
            f'[y={block.y_center:.0f}, x={block.x_center:.0f}] '
            f'({source_tag}) "{block.text}"'
        )
    return "\n".join(lines)


def structure_with_llm(formatted_text, document_type, settings):
    """Send combined text to LLM for structured extraction."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system_prompt = (
        "You are a document data extraction assistant. You receive "
        "OCR text from a document, where each line includes spatial "
        "coordinates [y, x] and a source tag (GT = verified ground truth, "
        "OCR = automated OCR which may have errors). "
        "The document may contain both Arabic and English text. "
        "Extract the requested fields accurately. Trust GT-tagged text "
        "more than OCR-tagged text when they conflict."
    )

    if document_type == "receipt":
        instructions = (
            "This is a receipt. Extract these fields (use null if not found):\n"
            "- store_name, receipt_number, receipt_date, receipt_time\n"
            "- subtotal, tax_amount, total_amount\n"
            "- payment_method, currency\n\n"
            "Also extract line items with: description, amount, "
            "quantity, unit_price, item_code\n\n"
        )
    else:
        instructions = (
            "This is an invoice. Extract these fields (use null if not found):\n"
            "- vendor_name, customer_name, invoice_number, invoice_date\n"
            "- due_date, subtotal, tax_amount, total_amount\n"
            "- currency\n\n"
            "Also extract line items with: description, amount, "
            "quantity, unit_price, item_code\n\n"
        )

    instructions += (
        "Respond with ONLY valid JSON in this format:\n"
        "{\n"
        '  "fields": [\n'
        '    {"field_name": "...", "value": "..."}\n'
        "  ],\n"
        '  "line_items": [\n'
        '    {"description": "...", "amount": 0.0, '
        '"quantity": null, "unit_price": null, "item_code": null}\n'
        "  ]\n"
        "}\n\n"
        "--- Document Text ---\n"
        f"{formatted_text}"
    )

    provider = settings.provider.lower()
    api_key = (
        settings.api_key.get_secret_value()
        if settings.api_key
        else None
    )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=settings.max_tokens,
        )
    else:
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": settings.model,
            "temperature": 0.0,
            "max_tokens": settings.max_tokens,
            "api_key": api_key or "not-needed",
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        llm = ChatOpenAI(**kwargs)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=instructions),
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response: %s", e)
        return {"fields": [], "line_items": []}


def structure_with_vlm(image_path, document_type, settings):
    """
    Send document image directly to a VLM for structured extraction.

    Used as fallback when OCR + GT text is insufficient.
    """
    import base64
    from langchain_core.messages import HumanMessage, SystemMessage

    # Load and encode image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("Could not load image: %s", image_path)
        return {"fields": [], "line_items": []}

    # Resize if needed
    h, w = image.shape[:2]
    max_edge = settings.max_image_long_edge
    long_edge = max(h, w)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        image = cv2.resize(
            image, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    system_prompt = (
        "You are a document data extraction assistant with vision capabilities. "
        "Read all text in the image carefully, including both Arabic and English. "
        "Pay close attention to numbers, dates, and amounts."
    )

    if document_type == "receipt":
        instructions = (
            "This is a receipt. Extract these fields (use null if not found):\n"
            "- store_name, receipt_number, receipt_date, receipt_time\n"
            "- subtotal, tax_amount, total_amount\n"
            "- payment_method, currency\n\n"
            "Also extract line items with: description, amount, "
            "quantity, unit_price, item_code\n\n"
        )
    else:
        instructions = (
            "This is an invoice. Extract these fields (use null if not found):\n"
            "- vendor_name, customer_name, invoice_number, invoice_date\n"
            "- due_date, subtotal, tax_amount, total_amount\n"
            "- currency\n\n"
            "Also extract line items with: description, amount, "
            "quantity, unit_price, item_code\n\n"
        )

    instructions += (
        "Respond with ONLY valid JSON in this format:\n"
        "{\n"
        '  "fields": [\n'
        '    {"field_name": "...", "value": "..."}\n'
        "  ],\n"
        '  "line_items": [\n'
        '    {"description": "...", "amount": 0.0, '
        '"quantity": null, "unit_price": null, "item_code": null}\n'
        "  ]\n"
        "}"
    )

    # Initialize LLM
    provider = settings.provider.lower()
    api_key = (
        settings.api_key.get_secret_value()
        if settings.api_key
        else None
    )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=settings.model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=settings.max_tokens,
        )
    else:
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": settings.model,
            "temperature": 0.0,
            "max_tokens": settings.max_tokens,
            "api_key": api_key or "not-needed",
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        llm = ChatOpenAI(**kwargs)

    human_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        },
        {
            "type": "text",
            "text": instructions,
        },
    ]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()

    if response_text.startswith("```"):
        lines = response_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse VLM response: %s", e)
        return {"fields": [], "line_items": []}


def build_ground_truth_entry(
    image_file, annotation_file, document_type,
    llm_result, model_used, text_blocks, raw_text,
):
    """Build a ground truth JSON entry with verification flags."""
    gt_count = sum(1 for b in text_blocks if b.source == "ground_truth")
    ocr_count = sum(1 for b in text_blocks if b.source == "ocr")

    fields = []
    for f in llm_result.get("fields", []):
        fields.append({
            "field_name": f.get("field_name", ""),
            "value": f.get("value"),
            "verified": False,
        })

    line_items = []
    for item in llm_result.get("line_items", []):
        line_items.append({
            "description": item.get("description", ""),
            "amount": item.get("amount"),
            "quantity": item.get("quantity"),
            "unit_price": item.get("unit_price"),
            "item_code": item.get("item_code"),
            "verified": False,
        })

    return {
        "image_file": image_file,
        "annotation_file": annotation_file,
        "document_type": document_type,
        "model_used": model_used,
        "fields": fields,
        "line_items": line_items,
        "raw_input": raw_text,
        "text_sources": {
            "ground_truth_regions": gt_count,
            "ocr_regions": ocr_count,
            "total_text_regions": gt_count + ocr_count,
        },
        "review_status": "pending",
    }


def select_images(annotation_dir, image_dir, count):
    """Select top N images by absolute transcription count."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    candidates = []
    for ann_file in sorted(annotation_dir.glob("*.json")):
        img_file = None
        for ext in image_extensions:
            candidate = image_dir / f"{ann_file.stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file is None:
            continue

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception):
            continue

        transcription_count = 0
        total_text_regions = 0
        for obj in data.get("objects", []):
            if obj.get("classTitle", "").lower() == "page":
                continue
            if obj.get("geometryType") != "rectangle":
                continue
            total_text_regions += 1
            for tag in obj.get("tags", []):
                if tag.get("name") == "Transcription":
                    transcription_count += 1
                    break

        if transcription_count > 0:
            candidates.append({
                "image": img_file,
                "annotation": ann_file,
                "transcription_count": transcription_count,
                "total_regions": total_text_regions,
            })

    candidates.sort(key=lambda x: x["transcription_count"], reverse=True)

    selected = candidates[:count]
    return [(c["image"], c["annotation"]) for c in selected]


def load_manual_selection(selection_path, image_dir, annotation_dir):
    """
    Load manually selected images from a selection JSON file
    (produced by select_images.py).

    Returns pairs of (image_path, annotation_path).
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    with open(selection_path, "r", encoding="utf-8") as f:
        selection = json.load(f)

    selected_names = selection.get("selected", [])
    pairs = []

    for name in selected_names:
        img_path = image_dir / name
        if not img_path.exists():
            print(f"  Warning: Image not found: {name}")
            continue

        # Find matching annotation
        stem = img_path.stem
        ann_path = annotation_dir / f"{stem}.json"
        if not ann_path.exists():
            # Image without annotation — still usable with OCR only
            pairs.append((img_path, None))
        else:
            pairs.append((img_path, ann_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth data for extraction evaluation.",
    )
    parser.add_argument("image_dir", type=str, help="Directory containing images.")
    parser.add_argument("annotation_dir", type=str, help="Directory containing annotations.")
    parser.add_argument("output_dir", type=str, help="Directory for output ground truth files.")
    parser.add_argument("--count", type=int, default=35, help="Number of images for auto-selection (default: 35).")
    parser.add_argument("--selection", type=str, default=None,
                        help="Path to manual selection JSON from select_images.py. Overrides auto-selection.")
    parser.add_argument("--doc-type", type=str, default="invoice",
                        choices=["invoice", "receipt"], help="Document type (default: invoice).")
    parser.add_argument("--min-blocks", type=int, default=None,
                        help=f"Minimum text blocks for OCR+LLM approach. Below this, falls back to VLM (default: {MIN_TEXT_BLOCKS}).")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    output_dir = Path(args.output_dir)

    if not image_dir.exists():
        print(f"Error: {image_dir} not found.")
        sys.exit(1)
    if not annotation_dir.exists():
        print(f"Error: {annotation_dir} not found.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    print(f"{'=' * 60}")
    print("Ground Truth Generation")
    print(f"{'=' * 60}")
    print(f"Image dir:      {image_dir}")
    print(f"Annotation dir: {annotation_dir}")
    print(f"Output dir:     {output_dir}")
    print(f"Document type:  {args.doc_type}")
    if args.selection:
        print(f"Selection:      Manual ({args.selection})")
    else:
        print(f"Selection:      Auto (top {args.count})")
    print(f"LLM provider:   {settings.extraction.provider}")
    print(f"LLM model:      {settings.extraction.model}")
    print()

    print("--- Selecting Images ---")
    if args.selection:
        selection_path = Path(args.selection)
        if not selection_path.exists():
            print(f"Error: Selection file not found: {selection_path}")
            sys.exit(1)
        pairs = load_manual_selection(selection_path, image_dir, annotation_dir)
        print(f"Loaded {len(pairs)} manually selected images")
    else:
        pairs = select_images(annotation_dir, image_dir, args.count)
        print(f"Auto-selected {len(pairs)} images by transcription count")
    print()

    if not pairs:
        print("No suitable images found.")
        sys.exit(1)

    min_blocks = args.min_blocks if args.min_blocks is not None else MIN_TEXT_BLOCKS

    print("--- Initializing Pipeline ---")
    preprocessor = ImagePreprocessor(settings=settings.preprocessing)
    ocr_engine = PaddleOCREngine(settings=settings.ocr)
    print("Pipeline ready")
    print(f"Min text blocks for OCR+LLM: {min_blocks} (below -> VLM fallback)")
    print()

    print("--- Processing Images ---")
    success_count = 0
    error_count = 0
    skipped_count = 0
    vlm_count = 0
    ocr_count = 0

    for i, (img_path, ann_path) in enumerate(pairs):
        print(f"  [{i + 1}/{len(pairs)}] {img_path.name}...", end=" ", flush=True)

        # Skip already reviewed files
        output_path = output_dir / f"{img_path.stem}_gt.json"
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("review_status") == "reviewed":
                    print("SKIPPED (already reviewed)")
                    skipped_count += 1
                    continue
            except (json.JSONDecodeError, Exception):
                pass

        try:
            # Parse annotation if available
            if ann_path is not None and ann_path.exists():
                annotation_data = parse_annotation(ann_path)
                ann_file_name = ann_path.name
            else:
                annotation_data = {
                    "transcribed": [],
                    "untranscribed_bboxes": [],
                    "image_size": {},
                }
                ann_file_name = "none"

            gt_count = len(annotation_data["transcribed"])
            gap_count = len(annotation_data["untranscribed_bboxes"])

            # Run OCR
            processed_image, _ = preprocessor.process(str(img_path))
            ocr_result = ocr_engine.recognize(processed_image)

            # Merge GT + OCR
            text_blocks = fill_with_ocr(annotation_data, ocr_result)

            # Decide: OCR+LLM or VLM fallback
            if len(text_blocks) >= min_blocks:
                # Sufficient text — use OCR+LLM approach
                formatted_text = format_text_for_llm(text_blocks)
                llm_result = structure_with_llm(
                    formatted_text, args.doc_type, settings.extraction
                )
                method = "OCR+LLM"
                raw_text = formatted_text
                ocr_count += 1
            else:
                # Insufficient text — fall back to VLM
                llm_result = structure_with_vlm(
                    img_path, args.doc_type, settings.extraction
                )
                method = "VLM"
                raw_text = f"[VLM extraction — only {len(text_blocks)} text blocks found, below threshold of {min_blocks}]"
                vlm_count += 1

            entry = build_ground_truth_entry(
                image_file=img_path.name,
                annotation_file=ann_file_name,
                document_type=args.doc_type,
                llm_result=llm_result,
                model_used=settings.extraction.model,
                text_blocks=text_blocks,
                raw_text=raw_text,
            )
            entry["extraction_method"] = method

            output_path = output_dir / f"{img_path.stem}_gt.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            fields_count = len(entry["fields"])
            items_count = len(entry["line_items"])
            print(f"[{method}] GT={gt_count} blocks={len(text_blocks)} -> {fields_count} fields, {items_count} items")
            success_count += 1

            time.sleep(1)

        except Exception as e:
            print(f"ERROR: {e}")
            error_count += 1

    print()
    print(f"{'=' * 60}")
    print("Generation Complete")
    print(f"{'=' * 60}")
    print(f"Total images:   {len(pairs)}")
    print(f"Succeeded:      {success_count}")
    print(f"  via OCR+LLM:  {ocr_count}")
    print(f"  via VLM:      {vlm_count}")
    print(f"Skipped:        {skipped_count} (already reviewed)")
    print(f"Failed:         {error_count}")
    print(f"Output dir:     {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run the review app to validate:")
    print(f"     streamlit run scripts/review_app.py -- {output_dir} {image_dir}")
    print("  2. Review and correct each entry")
    print("  3. Use verified entries for extraction evaluation")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
