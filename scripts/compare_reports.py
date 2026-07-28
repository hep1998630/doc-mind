"""
Report Comparison Script for DocMind.

Loads all evaluation report JSON files and generates a side-by-side
comparison of different model/pipeline configurations.

Usage:
    python scripts/compare_reports.py <ground_truth_dir>
    python scripts/compare_reports.py ground_truth/ --output comparison.json

The script automatically finds all report_*.json files in the directory.
"""

import argparse
import json
import sys
from pathlib import Path


def load_reports(gt_dir):
    """Load all report JSON files from the directory."""
    reports = []
    for f in sorted(gt_dir.glob("report_*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                report = json.load(fh)
                report["_file"] = f.name
                reports.append(report)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Could not load {f.name}: {e}")
    return reports


def build_comparison_row(report):
    """Extract key metrics from a single report into a flat dict."""
    meta = report.get("metadata", {})
    agg = report.get("aggregate", {})
    fields = agg.get("fields", {})
    items = agg.get("line_items", {})
    failures = report.get("failures", {})

    total_failures = sum(len(v) for v in failures.items()) if isinstance(failures, dict) else 0

    return {
        "report_name": meta.get("report_name", "unknown"),
        "mode": meta.get("mode", "?"),
        "ocr_engine": meta.get("ocr_engine", "n/a"),
        "model": meta.get("model", "unknown"),
        "provider": meta.get("provider", "unknown"),

        # Success
        "images_total": meta.get("images_total", 0),
        "images_ok": meta.get("images_evaluated", 0),
        "images_failed": meta.get("images_failed", total_failures),
        "success_rate": meta.get("success_rate", 0),

        # Fields
        "field_detection": fields.get("detection_rate", 0),
        "field_exact": fields.get("exact_match_rate", 0),
        "field_fuzzy": fields.get("fuzzy_match_rate", 0),
        "field_similarity": fields.get("avg_similarity", 0),
        "numeric_accuracy": fields.get("numeric_accuracy", 0),

        # Line items
        "item_precision": items.get("precision", 0),
        "item_recall": items.get("recall", 0),
        "item_f1": items.get("f1", 0),
        "item_amount_acc": items.get("amount_accuracy", 0),

        # Timing
        "avg_time": meta.get("avg_time_per_image_seconds", 0),
        "total_time": meta.get("total_time_seconds", 0),

        # Failures breakdown
        "failures": failures if isinstance(failures, dict) else {},
    }


def print_comparison(rows):
    """Print a formatted comparison table."""
    if not rows:
        print("No reports found.")
        return

    print()
    print(f"{'=' * 120}")
    print("MODEL COMPARISON REPORT")
    print(f"{'=' * 120}")
    print()

    # --- Overview Table ---
    print("--- Overview ---")
    print(
        f"  {'#':<3s} {'Name':<30s} {'Mode':<6s} "
        f"{'OCR Engine':<14s} {'Model':<28s} {'Success':>7s} {'Avg Time':>8s}"
    )
    print(f"  {'-' * 98}")
    for i, row in enumerate(rows):
        print(
            f"  {i + 1:<3d} {row['report_name']:<30s} {row['mode']:<6s} "
            f"{row['ocr_engine']:<14s} {row['model'][:28]:<28s} "
            f"{row['success_rate']:>6.0%} {row['avg_time']:>7.1f}s"
        )
    print()

    # --- Field Accuracy Table ---
    print("--- Field Accuracy ---")
    print(
        f"  {'#':<3s} {'Name':<30s} "
        f"{'Detect':>7s} {'Exact':>7s} {'Fuzzy':>7s} "
        f"{'AvgSim':>7s} {'NumAcc':>7s}"
    )
    print(f"  {'-' * 70}")
    for i, row in enumerate(rows):
        print(
            f"  {i + 1:<3d} {row['report_name']:<30s} "
            f"{row['field_detection']:>6.0%} "
            f"{row['field_exact']:>6.0%} "
            f"{row['field_fuzzy']:>6.0%} "
            f"{row['field_similarity']:>7.3f} "
            f"{row['numeric_accuracy']:>6.0%}"
        )
    print()

    # --- Line Item Accuracy Table ---
    print("--- Line Item Accuracy ---")
    print(
        f"  {'#':<3s} {'Name':<30s} "
        f"{'Precision':>9s} {'Recall':>7s} {'F1':>7s} {'AmtAcc':>7s}"
    )
    print(f"  {'-' * 65}")
    for i, row in enumerate(rows):
        print(
            f"  {i + 1:<3d} {row['report_name']:<30s} "
            f"{row['item_precision']:>8.0%} "
            f"{row['item_recall']:>6.0%} "
            f"{row['item_f1']:>6.0%} "
            f"{row['item_amount_acc']:>6.0%}"
        )
    print()

    # --- Timing Comparison ---
    print("--- Speed ---")
    print(
        f"  {'#':<3s} {'Name':<30s} "
        f"{'Avg/Image':>10s} {'Total':>10s}"
    )
    print(f"  {'-' * 55}")
    for i, row in enumerate(rows):
        print(
            f"  {i + 1:<3d} {row['report_name']:<30s} "
            f"{row['avg_time']:>9.1f}s "
            f"{row['total_time']:>9.1f}s"
        )
    print()

    # --- Failure Comparison ---
    any_failures = any(row["images_failed"] > 0 for row in rows)
    if any_failures:
        print("--- Failures ---")
        print(
            f"  {'#':<3s} {'Name':<30s} "
            f"{'Failed':>6s} {'Types':<40s}"
        )
        print(f"  {'-' * 80}")
        for i, row in enumerate(rows):
            if row["images_failed"] > 0:
                failure_types = ", ".join(
                    f"{k}({len(v)})"
                    for k, v in row["failures"].items()
                    if v
                )
                print(
                    f"  {i + 1:<3d} {row['report_name']:<30s} "
                    f"{row['images_failed']:>6d} "
                    f"{failure_types:<40s}"
                )
        print()

    # --- Rankings ---
    print("--- Rankings (Best to Worst) ---")
    print()

    rankings = [
        ("Field Exact Match", "field_exact", True),
        ("Field Fuzzy Match", "field_fuzzy", True),
        ("Numeric Accuracy", "numeric_accuracy", True),
        ("Line Item F1", "item_f1", True),
        ("Line Item Amount Accuracy", "item_amount_acc", True),
        ("Speed (avg per image)", "avg_time", False),
        ("Success Rate", "success_rate", True),
    ]

    for label, key, higher_is_better in rankings:
        sorted_rows = sorted(
            rows,
            key=lambda r: r[key],
            reverse=higher_is_better,
        )
        top = sorted_rows[0]
        bottom = sorted_rows[-1]

        if key == "avg_time":
            print(
                f"  {label + ':':<32s} "
                f"Best = {top['report_name']} ({top[key]:.1f}s)  |  "
                f"Worst = {bottom['report_name']} ({bottom[key]:.1f}s)"
            )
        else:
            print(
                f"  {label + ':':<32s} "
                f"Best = {top['report_name']} ({top[key]:.0%})  |  "
                f"Worst = {bottom['report_name']} ({bottom[key]:.0%})"
            )
    print()

    # --- Per-Field Breakdown Across Models ---
    print("--- Per-Field Comparison (Fuzzy Match Rate) ---")

    # Collect all field names across all reports
    all_field_names = set()
    for row in rows:
        report = row.get("_report", {})
        breakdown = report.get("aggregate", {}).get("per_field_breakdown", {})
        all_field_names.update(breakdown.keys())

    if all_field_names:
        sorted_fields = sorted(all_field_names)

        header = f"  {'Field':<22s}"
        for i, row in enumerate(rows):
            name_short = row["report_name"][:12]
            header += f" {name_short:>12s}"
        print(header)
        print(f"  {'-' * (22 + 13 * len(rows))}")

        for field_name in sorted_fields:
            line = f"  {field_name:<22s}"
            for row in rows:
                report = row.get("_report", {})
                breakdown = report.get("aggregate", {}).get(
                    "per_field_breakdown", {}
                )
                field_data = breakdown.get(field_name, {})
                fuzzy_rate = field_data.get("fuzzy_match_rate", -1)
                if fuzzy_rate >= 0:
                    line += f" {fuzzy_rate:>11.0%}"
                else:
                    line += f" {'n/a':>12s}"
            print(line)
        print()


def save_comparison(rows, output_path):
    """Save comparison data as JSON."""
    # Remove non-serializable _report reference
    clean_rows = []
    for row in rows:
        clean = {k: v for k, v in row.items() if k != "_report"}
        clean_rows.append(clean)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_rows, f, indent=2, ensure_ascii=False)
    print(f"Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare extraction evaluation reports.",
    )
    parser.add_argument(
        "report_dir",
        type=str,
        help="Directory containing report_*.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save comparison as JSON to this path.",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"Error: {report_dir} not found.")
        sys.exit(1)

    print(f"Loading reports from: {report_dir}")
    reports = load_reports(report_dir)

    if not reports:
        print("No report_*.json files found.")
        sys.exit(1)

    print(f"Found {len(reports)} reports")

    # Build comparison rows, keeping reference to full report
    rows = []
    for report in reports:
        row = build_comparison_row(report)
        row["_report"] = report
        rows.append(row)

    # Sort by field fuzzy match rate descending (best first)
    rows.sort(key=lambda r: r["field_fuzzy"], reverse=True)

    print_comparison(rows)

    # Save if requested
    output_path = args.output or report_dir / "comparison.json"
    save_comparison(rows, Path(output_path))

    print()
    print(f"{'=' * 120}")
    print("Comparison complete!")
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
