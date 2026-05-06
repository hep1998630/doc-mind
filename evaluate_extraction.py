"""
Extraction Evaluation Script for DocMind.

Runs a single pipeline configuration against verified ground truth
and generates a detailed report. Run multiple times with different
models/settings to accumulate reports for comparison.

Usage:
    # Evaluate OCR+LLM pipeline
    python evaluate_extraction.py <ground_truth_dir> <image_dir> --mode ocr

    # Evaluate VLM pipeline
    python evaluate_extraction.py <ground_truth_dir> <image_dir> --mode vlm

    # Custom report name
    python evaluate_extraction.py gt/ images/ --mode vlm --report-name qwen_vl_72b

Examples:
    python evaluate_extraction.py ground_truth/ samples/invoices/ --mode ocr
    python evaluate_extraction.py ground_truth/ samples/invoices/ --mode vlm --report-name gpt4o_vlm
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docmind.config.settings import get_settings


# --- Text Comparison Utilities ---

def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def normalized_similarity(s1, s2):
    """
    Compute normalized string similarity between 0 and 1.
    1.0 = identical, 0.0 = completely different.
    """
    s1 = str(s1 or "").strip()
    s2 = str(s2 or "").strip()

    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    distance = levenshtein_distance(s1, s2)
    return 1.0 - (distance / max_len)


def is_exact_match(gt_value, pred_value):
    """Check if two values match exactly after stripping."""
    gt = str(gt_value or "").strip()
    pred = str(pred_value or "").strip()
    return gt == pred


def is_fuzzy_match(gt_value, pred_value, threshold=0.8):
    """Check if two values match within a similarity threshold."""
    return normalized_similarity(gt_value, pred_value) >= threshold


def numeric_match(gt_value, pred_value, tolerance=0.01):
    """
    Compare two values as numbers.
    Returns True if both are parseable and within tolerance.
    """
    try:
        gt_num = float(str(gt_value).replace(",", "").strip())
        pred_num = float(str(pred_value).replace(",", "").strip())
        return abs(gt_num - pred_num) <= tolerance
    except (ValueError, TypeError):
        return False


# --- Field Evaluation ---

NUMERIC_FIELDS = {
    "subtotal", "tax_amount", "total_amount",
    "amount", "quantity", "unit_price",
}


def evaluate_fields(gt_fields, pred_fields):
    """
    Evaluate predicted fields against ground truth fields.

    Returns per-field results and aggregate metrics.
    """
    # Build lookup from predicted fields
    pred_lookup = {}
    for f in pred_fields:
        name = f.get("field_name", "")
        pred_lookup[name] = f.get("value")

    results = []
    exact_matches = 0
    fuzzy_matches = 0
    numeric_matches = 0
    detected = 0
    total = 0

    for gt_field in gt_fields:
        name = gt_field.get("field_name", "")
        gt_value = gt_field.get("value")

        # Skip null ground truth values
        if gt_value is None or str(gt_value).strip() == "":
            continue

        total += 1
        pred_value = pred_lookup.get(name)

        field_result = {
            "field_name": name,
            "gt_value": gt_value,
            "pred_value": pred_value,
            "detected": pred_value is not None,
            "exact_match": False,
            "fuzzy_match": False,
            "numeric_match": False,
            "similarity": 0.0,
        }

        if pred_value is not None:
            detected += 1
            field_result["similarity"] = normalized_similarity(
                gt_value, pred_value
            )
            field_result["exact_match"] = is_exact_match(gt_value, pred_value)
            field_result["fuzzy_match"] = is_fuzzy_match(gt_value, pred_value)

            if name in NUMERIC_FIELDS:
                field_result["numeric_match"] = numeric_match(
                    gt_value, pred_value
                )

            if field_result["exact_match"]:
                exact_matches += 1
            if field_result["fuzzy_match"]:
                fuzzy_matches += 1
            if field_result["numeric_match"]:
                numeric_matches += 1

        results.append(field_result)

    # Count numeric fields for separate accuracy
    numeric_total = sum(
        1 for r in results if r["field_name"] in NUMERIC_FIELDS
    )
    numeric_correct = sum(
        1 for r in results
        if r["field_name"] in NUMERIC_FIELDS and r["numeric_match"]
    )

    return {
        "per_field": results,
        "total_fields": total,
        "detected": detected,
        "exact_matches": exact_matches,
        "fuzzy_matches": fuzzy_matches,
        "detection_rate": detected / total if total > 0 else 0,
        "exact_match_rate": exact_matches / total if total > 0 else 0,
        "fuzzy_match_rate": fuzzy_matches / total if total > 0 else 0,
        "avg_similarity": (
            sum(r["similarity"] for r in results) / len(results)
            if results else 0
        ),
        "numeric_total": numeric_total,
        "numeric_correct": numeric_correct,
        "numeric_accuracy": (
            numeric_correct / numeric_total if numeric_total > 0 else 0
        ),
    }


# --- Line Item Evaluation ---

def match_line_items(gt_items, pred_items, match_threshold=0.5):
    """
    Match predicted line items to ground truth using fuzzy
    description matching. Uses greedy best-match pairing.

    Returns matched pairs, unmatched GT, and unmatched predictions.
    """
    if not gt_items or not pred_items:
        return [], gt_items or [], pred_items or []

    # Compute similarity matrix
    used_pred = set()
    matches = []
    unmatched_gt = []

    for gt_item in gt_items:
        gt_desc = gt_item.get("description", "")
        best_sim = 0
        best_idx = -1

        for j, pred_item in enumerate(pred_items):
            if j in used_pred:
                continue
            pred_desc = pred_item.get("description", "")
            sim = normalized_similarity(gt_desc, pred_desc)
            if sim > best_sim:
                best_sim = sim
                best_idx = j

        if best_sim >= match_threshold and best_idx >= 0:
            matches.append((gt_item, pred_items[best_idx], best_sim))
            used_pred.add(best_idx)
        else:
            unmatched_gt.append(gt_item)

    unmatched_pred = [
        pred_items[j] for j in range(len(pred_items))
        if j not in used_pred
    ]

    return matches, unmatched_gt, unmatched_pred


def evaluate_line_items(gt_items, pred_items):
    """
    Evaluate predicted line items against ground truth.

    Returns matching metrics and per-item accuracy.
    """
    # Filter to verified items only
    gt_verified = [
        item for item in gt_items
        if item.get("verified", False)
    ]

    if not gt_verified:
        return {
            "gt_count": 0,
            "pred_count": len(pred_items),
            "matched": 0,
            "unmatched_gt": 0,
            "false_positives": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "per_item": [],
            "amount_accuracy": 0,
        }

    matches, unmatched_gt, unmatched_pred = match_line_items(
        gt_verified, pred_items
    )

    matched_count = len(matches)
    precision = (
        matched_count / len(pred_items) if pred_items else 0
    )
    recall = (
        matched_count / len(gt_verified) if gt_verified else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    # Per-item field accuracy
    per_item_results = []
    amount_correct = 0

    for gt_item, pred_item, desc_sim in matches:
        item_result = {
            "gt_description": gt_item.get("description", ""),
            "pred_description": pred_item.get("description", ""),
            "description_similarity": desc_sim,
            "amount_match": numeric_match(
                gt_item.get("amount"), pred_item.get("amount")
            ),
            "quantity_match": False,
            "unit_price_match": False,
        }

        if gt_item.get("quantity") is not None:
            item_result["quantity_match"] = numeric_match(
                gt_item.get("quantity"), pred_item.get("quantity")
            )

        if gt_item.get("unit_price") is not None:
            item_result["unit_price_match"] = numeric_match(
                gt_item.get("unit_price"), pred_item.get("unit_price")
            )

        if item_result["amount_match"]:
            amount_correct += 1

        per_item_results.append(item_result)

    return {
        "gt_count": len(gt_verified),
        "pred_count": len(pred_items),
        "matched": matched_count,
        "unmatched_gt": len(unmatched_gt),
        "false_positives": len(unmatched_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_item": per_item_results,
        "amount_accuracy": (
            amount_correct / matched_count if matched_count > 0 else 0
        ),
    }


# --- Document-Level Evaluation ---

def evaluate_document(gt_entry, pred_result):
    """Evaluate a single document's extraction against ground truth."""
    field_eval = evaluate_fields(
        gt_entry.get("fields", []),
        pred_result.get("fields", []),
    )

    item_eval = evaluate_line_items(
        gt_entry.get("line_items", []),
        pred_result.get("line_items", []),
    )

    return {
        "image_file": gt_entry.get("image_file", ""),
        "fields": field_eval,
        "line_items": item_eval,
    }


# --- Pipeline Runners ---

def run_ocr_pipeline(image_path, document_type, preprocessor, ocr_engine, extractor):
    """Run OCR+LLM pipeline and return extraction result as dict."""
    from docmind.models.common import DocumentType

    processed_image, _ = preprocessor.process(str(image_path))
    ocr_result = ocr_engine.recognize(processed_image)
    extraction = extractor.extract(ocr_result, DocumentType(document_type))

    return extraction.model_dump()


def run_vlm_pipeline(image_path, document_type, extractor):
    """Run VLM pipeline and return extraction result as dict."""
    from docmind.models.common import DocumentType

    extraction = extractor.extract_from_image(
        str(image_path), DocumentType(document_type)
    )

    return extraction.model_dump()


# --- Reporting ---

def print_report(report):
    """Print a human-readable evaluation report."""
    print()
    print(f"{'=' * 60}")
    print("EXTRACTION EVALUATION REPORT")
    print(f"{'=' * 60}")
    print()

    meta = report.get("metadata", {})
    print(f"Pipeline:       {meta.get('mode', 'unknown')}")
    if meta.get("ocr_engine") and meta.get("ocr_engine") != "none":
        print(f"OCR engine:     {meta.get('ocr_engine')}")
    print(f"Model:          {meta.get('model', 'unknown')}")
    print(f"Provider:       {meta.get('provider', 'unknown')}")
    print(f"Document type:  {meta.get('document_type', 'unknown')}")
    print(f"Images total:   {meta.get('images_total', 0)}")
    print(f"Succeeded:      {meta.get('images_evaluated', 0)}")
    print(f"Failed:         {meta.get('images_failed', 0)}")
    print(f"Success rate:   {meta.get('success_rate', 0):.1%}")
    print()

    # Failure breakdown
    failure_data = report.get("failures", {})
    if failure_data:
        print("--- Failures ---")
        for error_type, images in failure_data.items():
            print(f"  {error_type:<25s} {len(images):>3d} images")
            for img in images[:5]:
                print(f"    - {img}")
            if len(images) > 5:
                print(f"    ... and {len(images) - 5} more")
        print()

    print("--- Timing ---")
    print(f"Total time:         {meta.get('total_time_seconds', 0):.1f}s")
    print(f"Avg per image:      {meta.get('avg_time_per_image_seconds', 0):.1f}s")
    print(f"Min per image:      {meta.get('min_time_seconds', 0):.1f}s")
    print(f"Max per image:      {meta.get('max_time_seconds', 0):.1f}s")
    print()

    agg = report.get("aggregate", {})
    fields = agg.get("fields", {})
    items = agg.get("line_items", {})

    print("--- Field Metrics ---")
    print(f"Detection rate:     {fields.get('detection_rate', 0):.1%}")
    print(f"Exact match rate:   {fields.get('exact_match_rate', 0):.1%}")
    print(f"Fuzzy match rate:   {fields.get('fuzzy_match_rate', 0):.1%}")
    print(f"Avg similarity:     {fields.get('avg_similarity', 0):.3f}")
    print(f"Numeric accuracy:   {fields.get('numeric_accuracy', 0):.1%}")
    print()

    print("--- Line Item Metrics ---")
    print(f"Precision:          {items.get('precision', 0):.1%}")
    print(f"Recall:             {items.get('recall', 0):.1%}")
    print(f"F1 Score:           {items.get('f1', 0):.1%}")
    print(f"Amount accuracy:    {items.get('amount_accuracy', 0):.1%}")
    print()

    # Per-field breakdown
    field_breakdown = agg.get("per_field_breakdown", {})
    if field_breakdown:
        print("--- Per-Field Breakdown ---")
        print(f"  {'Field':<22s} {'Exact':>6s} {'Fuzzy':>6s} {'AvgSim':>7s} {'Count':>5s}")
        print(f"  {'-' * 48}")
        for name, data in sorted(field_breakdown.items()):
            print(
                f"  {name:<22s} "
                f"{data['exact_match_rate']:>5.0%} "
                f"{data['fuzzy_match_rate']:>5.0%} "
                f"{data['avg_similarity']:>7.3f} "
                f"{data['count']:>5d}"
            )
        print()

    # Per-document summary
    print("--- Per-Document Results ---")
    print(
        f"  {'Image':<30s} "
        f"{'F-Exact':>7s} {'F-Fuzzy':>7s} "
        f"{'LI-F1':>6s} {'LI-Amt':>6s} "
        f"{'Time':>6s}"
    )
    print(f"  {'-' * 65}")
    for doc in report.get("per_document", []):
        img = Path(doc["image_file"]).name[:30]
        f_exact = doc["fields"]["exact_match_rate"]
        f_fuzzy = doc["fields"]["fuzzy_match_rate"]
        li_f1 = doc["line_items"]["f1"]
        li_amt = doc["line_items"]["amount_accuracy"]
        doc_time = doc.get("processing_time_seconds", 0)
        print(
            f"  {img:<30s} "
            f"{f_exact:>6.0%} {f_fuzzy:>6.0%} "
            f"{li_f1:>5.0%} {li_amt:>5.0%} "
            f"{doc_time:>5.1f}s"
        )
    print()


def aggregate_results(doc_results):
    """Compute aggregate metrics across all documents."""
    if not doc_results:
        return {}

    # Aggregate field metrics
    total_fields = sum(d["fields"]["total_fields"] for d in doc_results)
    total_exact = sum(d["fields"]["exact_matches"] for d in doc_results)
    total_fuzzy = sum(d["fields"]["fuzzy_matches"] for d in doc_results)
    total_detected = sum(d["fields"]["detected"] for d in doc_results)
    total_numeric = sum(d["fields"]["numeric_total"] for d in doc_results)
    total_numeric_correct = sum(
        d["fields"]["numeric_correct"] for d in doc_results
    )

    all_similarities = []
    for d in doc_results:
        for f in d["fields"]["per_field"]:
            if f["detected"]:
                all_similarities.append(f["similarity"])

    # Per-field breakdown
    field_breakdown = {}
    for d in doc_results:
        for f in d["fields"]["per_field"]:
            name = f["field_name"]
            if name not in field_breakdown:
                field_breakdown[name] = {
                    "count": 0,
                    "exact": 0,
                    "fuzzy": 0,
                    "similarities": [],
                }
            field_breakdown[name]["count"] += 1
            if f["exact_match"]:
                field_breakdown[name]["exact"] += 1
            if f["fuzzy_match"]:
                field_breakdown[name]["fuzzy"] += 1
            field_breakdown[name]["similarities"].append(f["similarity"])

    field_breakdown_summary = {}
    for name, data in field_breakdown.items():
        count = data["count"]
        field_breakdown_summary[name] = {
            "count": count,
            "exact_match_rate": data["exact"] / count if count > 0 else 0,
            "fuzzy_match_rate": data["fuzzy"] / count if count > 0 else 0,
            "avg_similarity": (
                sum(data["similarities"]) / len(data["similarities"])
                if data["similarities"] else 0
            ),
        }

    # Aggregate line item metrics
    total_gt_items = sum(d["line_items"]["gt_count"] for d in doc_results)
    total_matched_items = sum(d["line_items"]["matched"] for d in doc_results)
    total_pred_items = sum(d["line_items"]["pred_count"] for d in doc_results)
    total_amount_correct = sum(
        sum(1 for it in d["line_items"]["per_item"] if it["amount_match"])
        for d in doc_results
    )

    precision = (
        total_matched_items / total_pred_items
        if total_pred_items > 0 else 0
    )
    recall = (
        total_matched_items / total_gt_items
        if total_gt_items > 0 else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "fields": {
            "total_fields": total_fields,
            "detected": total_detected,
            "exact_matches": total_exact,
            "fuzzy_matches": total_fuzzy,
            "detection_rate": total_detected / total_fields if total_fields > 0 else 0,
            "exact_match_rate": total_exact / total_fields if total_fields > 0 else 0,
            "fuzzy_match_rate": total_fuzzy / total_fields if total_fields > 0 else 0,
            "avg_similarity": (
                sum(all_similarities) / len(all_similarities)
                if all_similarities else 0
            ),
            "numeric_total": total_numeric,
            "numeric_correct": total_numeric_correct,
            "numeric_accuracy": (
                total_numeric_correct / total_numeric
                if total_numeric > 0 else 0
            ),
        },
        "line_items": {
            "gt_count": total_gt_items,
            "pred_count": total_pred_items,
            "matched": total_matched_items,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "amount_accuracy": (
                total_amount_correct / total_matched_items
                if total_matched_items > 0 else 0
            ),
        },
        "per_field_breakdown": field_breakdown_summary,
    }


# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extraction accuracy against verified ground truth.",
    )
    parser.add_argument(
        "ground_truth_dir", type=str,
        help="Directory containing verified *_gt.json files.",
    )
    parser.add_argument(
        "image_dir", type=str,
        help="Directory containing invoice images.",
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["ocr", "vlm"],
        help="Pipeline mode: 'ocr' for OCR+LLM, 'vlm' for vision LLM.",
    )
    parser.add_argument(
        "--report-name", type=str, default=None,
        help="Custom name for the report file. Defaults to mode_model.",
    )
    parser.add_argument(
        "--doc-type", type=str, default="invoice",
        choices=["invoice", "receipt"],
        help="Document type (default: invoice).",
    )
    parser.add_argument(
        "--ocr-engine", type=str, default="paddleocr",
        choices=["paddleocr", "deepseek-ocr"],
        help="OCR engine for ocr mode (default: paddleocr).",
    )
    args = parser.parse_args()

    gt_dir = Path(args.ground_truth_dir)
    image_dir = Path(args.image_dir)

    if not gt_dir.exists():
        print(f"Error: {gt_dir} not found.")
        sys.exit(1)
    if not image_dir.exists():
        print(f"Error: {image_dir} not found.")
        sys.exit(1)

    settings = get_settings()

    # Find verified ground truth files
    gt_files = []
    for f in sorted(gt_dir.glob("*_gt.json")):
        with open(f, "r", encoding="utf-8") as fh:
            entry = json.load(fh)
        if entry.get("review_status") == "reviewed":
            gt_files.append((f, entry))

    if not gt_files:
        print("No verified ground truth files found.")
        print("Review and mark entries as 'reviewed' in the review app first.")
        sys.exit(1)

    # Build report name
    model_name = settings.extraction.model.replace("/", "_")
    ocr_suffix = f"_{args.ocr_engine}" if args.mode == "ocr" else ""
    report_name = args.report_name or f"{args.mode}{ocr_suffix}_{model_name}"
    report_path = gt_dir / f"report_{report_name}.json"

    print(f"{'=' * 60}")
    print("Extraction Evaluation")
    print(f"{'=' * 60}")
    print(f"Mode:           {args.mode.upper()}")
    if args.mode == "ocr":
        print(f"OCR engine:     {args.ocr_engine}")
    print(f"LLM Model:      {settings.extraction.model}")
    print(f"Provider:       {settings.extraction.provider}")
    print(f"Document type:  {args.doc_type}")
    print(f"GT files:       {len(gt_files)} verified")
    print(f"Report:         {report_path}")
    print()

    # Run evaluation
    print("--- Initializing Pipeline ---")
    if args.mode == "ocr":
        from docmind.modules.preprocessing.processor import ImagePreprocessor
        from docmind.modules.extraction.langchain_extractor import LangChainExtractor

        preprocessor = ImagePreprocessor(settings=settings.preprocessing)
        extractor = LangChainExtractor(settings=settings.extraction)

        if args.ocr_engine == "deepseek-ocr":
            from docmind.modules.ocr.deepseek_ocr import DeepSeekOCREngine
            ocr_engine = DeepSeekOCREngine(prompt_mode="markdown")
            print("DeepSeek-OCR + LLM pipeline ready")
        else:
            from docmind.modules.ocr.paddle_ocr import PaddleOCREngine
            ocr_engine = PaddleOCREngine(settings=settings.ocr)
            print("PaddleOCR + LLM pipeline ready")
    else:
        from docmind.modules.extraction.vlm_extractor import VLMExtractor

        vlm_extractor = VLMExtractor(settings=settings.extraction)
        print("VLM pipeline ready")
    print()

    print("--- Running Pipeline ---")
    doc_results = []
    processing_times = []
    success = 0
    failures: dict[str, list[str]] = {
        "image_not_found": [],
        "empty_response": [],
        "no_json_found": [],
        "truncated_json": [],
        "invalid_json": [],
        "schema_validation_error": [],
        "api_error": [],
        "unknown_error": [],
    }
    total_start = time.time()

    from docmind.models.extraction import ExtractionError

    for i, (gt_path, gt_entry) in enumerate(gt_files):
        image_file = gt_entry.get("image_file", "")
        image_path = image_dir / image_file

        print(f"  [{i + 1}/{len(gt_files)}] {image_file}...", end=" ", flush=True)

        if not image_path.exists():
            print(f"IMAGE NOT FOUND")
            failures["image_not_found"].append(image_file)
            continue

        try:
            # Run the selected pipeline
            doc_start = time.time()

            if args.mode == "ocr":
                pred_result = run_ocr_pipeline(
                    image_path, args.doc_type,
                    preprocessor, ocr_engine, extractor,
                )
            else:
                pred_result = run_vlm_pipeline(
                    image_path, args.doc_type, vlm_extractor,
                )

            doc_time = time.time() - doc_start
            processing_times.append(doc_time)

            # Evaluate against ground truth
            doc_eval = evaluate_document(gt_entry, pred_result)
            doc_eval["processing_time_seconds"] = round(doc_time, 2)
            doc_results.append(doc_eval)

            f_exact = doc_eval["fields"]["exact_match_rate"]
            li_f1 = doc_eval["line_items"]["f1"]
            print(f"fields={f_exact:.0%} items_f1={li_f1:.0%} time={doc_time:.1f}s")

            success += 1
            time.sleep(1)

        except ExtractionError as e:
            print(f"EXTRACTION FAILED [{e.error_type}]: {e}")
            failures[e.error_type].append(image_file)

        except Exception as e:
            error_type = "api_error" if "api" in str(e).lower() or "rate" in str(e).lower() else "unknown_error"
            print(f"ERROR [{error_type}]: {e}")
            failures[error_type].append(image_file)

    total_time = time.time() - total_start
    total_failures = sum(len(v) for v in failures.values())

    # Aggregate results
    aggregate = aggregate_results(doc_results)

    # Build failure summary (only include non-empty categories)
    failure_summary = {
        k: v for k, v in failures.items() if v
    }

    # Build report
    report = {
        "metadata": {
            "mode": args.mode,
            "model": settings.extraction.model,
            "provider": settings.extraction.provider,
            "ocr_engine": args.ocr_engine if args.mode == "ocr" else "none",
            "document_type": args.doc_type,
            "images_evaluated": success,
            "images_failed": total_failures,
            "images_total": len(gt_files),
            "success_rate": success / len(gt_files) if gt_files else 0,
            "report_name": report_name,
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_image_seconds": (
                round(sum(processing_times) / len(processing_times), 2)
                if processing_times else 0
            ),
            "min_time_seconds": (
                round(min(processing_times), 2)
                if processing_times else 0
            ),
            "max_time_seconds": (
                round(max(processing_times), 2)
                if processing_times else 0
            ),
        },
        "failures": failure_summary,
        "aggregate": aggregate,
        "per_document": doc_results,
    }

    # Save report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print report
    print_report(report)

    print(f"Report saved to: {report_path}")
    print()
    print(f"{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
