"""
Annotation Quality Assessment Script.

Scans annotation files to assess data quality and coverage
before running OCR evaluation.

Usage:
    python assess_annotations.py <annotation_dir>
    python assess_annotations.py <single_annotation.json>

Examples:
    python assess_annotations.py samples/annotations/
    python assess_annotations.py samples/annotations/001.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def assess_single_file(annotation_path: Path) -> dict:
    """Assess a single annotation file and return stats."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_size = data.get("size", {})
    objects = data.get("objects", [])

    stats = {
        "file": annotation_path.name,
        "image_width": image_size.get("width", 0),
        "image_height": image_size.get("height", 0),
        "total_objects": len(objects),
        "classes": Counter(),
        "geometry_types": Counter(),
        "with_transcription": 0,
        "without_transcription": 0,
        "transcriptions": [],
        "tag_names": Counter(),
    }

    for obj in objects:
        class_title = obj.get("classTitle", "Unknown")
        geometry = obj.get("geometryType", "unknown")

        stats["classes"][class_title] += 1
        stats["geometry_types"][geometry] += 1

        # Check all tags
        tags = obj.get("tags", [])
        has_transcription = False
        for tag in tags:
            tag_name = tag.get("name", "")
            stats["tag_names"][tag_name] += 1
            if tag_name == "Transcription":
                has_transcription = True
                value = tag.get("value", "")
                stats["transcriptions"].append({
                    "class": class_title,
                    "text": value,
                    "text_length": len(value),
                })

        if class_title.lower() != "page":
            if has_transcription:
                stats["with_transcription"] += 1
            else:
                stats["without_transcription"] += 1

    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python assess_annotations.py <annotation_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: {path} not found.")
        sys.exit(1)

    # Collect annotation files
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*.json"))

    if not files:
        print("No JSON annotation files found.")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("Annotation Quality Assessment")
    print(f"{'=' * 60}")
    print(f"Files found: {len(files)}")
    print()

    # Aggregate stats
    total_objects = 0
    total_text_regions = 0
    total_with_transcription = 0
    total_without_transcription = 0
    all_classes = Counter()
    all_geometry_types = Counter()
    all_tag_names = Counter()
    all_transcriptions = []
    files_with_any_transcription = 0
    files_without_any_transcription = 0
    image_sizes = Counter()
    transcription_languages = {"arabic": 0, "latin": 0, "mixed": 0, "numeric": 0}

    # Per-image coverage tracking
    per_image_coverage: list[dict] = []

    for annotation_file in files:
        try:
            stats = assess_single_file(annotation_file)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Failed to parse {annotation_file.name}: {e}")
            continue

        total_objects += stats["total_objects"]
        text_region_count = (
            stats["with_transcription"] + stats["without_transcription"]
        )
        total_text_regions += text_region_count
        total_with_transcription += stats["with_transcription"]
        total_without_transcription += stats["without_transcription"]
        all_classes += stats["classes"]
        all_geometry_types += stats["geometry_types"]
        all_tag_names += stats["tag_names"]
        all_transcriptions.extend(stats["transcriptions"])

        size_key = f"{stats['image_width']}x{stats['image_height']}"
        image_sizes[size_key] += 1

        if stats["with_transcription"] > 0:
            files_with_any_transcription += 1
        else:
            files_without_any_transcription += 1

        # Track per-image coverage
        coverage_pct = (
            stats["with_transcription"] / text_region_count * 100
            if text_region_count > 0 else 0.0
        )
        per_image_coverage.append({
            "file": stats["file"],
            "text_regions": text_region_count,
            "with_transcription": stats["with_transcription"],
            "without_transcription": stats["without_transcription"],
            "coverage_pct": coverage_pct,
            "transcriptions": stats["transcriptions"],
        })

    # Analyze transcription languages
    arabic_range = range(0x0600, 0x0700)
    for t in all_transcriptions:
        text = t["text"]
        has_arabic = any(ord(c) in arabic_range for c in text)
        has_latin = any(c.isalpha() and ord(c) < 0x0600 for c in text)
        is_numeric = all(c.isdigit() or c in " .,/-:" for c in text)

        if is_numeric:
            transcription_languages["numeric"] += 1
        elif has_arabic and has_latin:
            transcription_languages["mixed"] += 1
        elif has_arabic:
            transcription_languages["arabic"] += 1
        else:
            transcription_languages["latin"] += 1

    # Print report
    print("--- Overall Stats ---")
    print(f"Total annotation files:     {len(files)}")
    print(f"Total objects (all types):  {total_objects}")
    print(f"Total text regions:         {total_text_regions}")
    print()

    print("--- Transcription Coverage ---")
    print(f"Regions WITH transcription:    {total_with_transcription}")
    print(f"Regions WITHOUT transcription: {total_without_transcription}")
    if total_text_regions > 0:
        coverage = total_with_transcription / total_text_regions
        print(f"Transcription coverage:        {coverage:.1%}")
    print()
    print(f"Files with any transcription:     {files_with_any_transcription}")
    print(f"Files without any transcription:  {files_without_any_transcription}")
    print()

    print("--- Class Distribution ---")
    for cls, count in all_classes.most_common():
        print(f"  {cls:<20s} {count:>5d}")
    print()

    print("--- Geometry Types ---")
    for geo, count in all_geometry_types.most_common():
        print(f"  {geo:<20s} {count:>5d}")
    print()

    print("--- Tag Names Found ---")
    if all_tag_names:
        for tag, count in all_tag_names.most_common():
            print(f"  {tag:<20s} {count:>5d}")
    else:
        print("  (no tags found)")
    print()

    print("--- Image Sizes ---")
    for size, count in image_sizes.most_common():
        print(f"  {size:<15s} {count:>5d} images")
    print()

    if all_transcriptions:
        print("--- Transcription Analysis ---")
        print(f"Total transcriptions:  {len(all_transcriptions)}")
        lengths = [t["text_length"] for t in all_transcriptions]
        print(f"Avg text length:       {sum(lengths) / len(lengths):.1f} chars")
        print(f"Min text length:       {min(lengths)} chars")
        print(f"Max text length:       {max(lengths)} chars")
        print()

        print("--- Transcription Language Breakdown ---")
        print(f"  Arabic only:   {transcription_languages['arabic']:>5d}")
        print(f"  Latin only:    {transcription_languages['latin']:>5d}")
        print(f"  Mixed:         {transcription_languages['mixed']:>5d}")
        print(f"  Numeric only:  {transcription_languages['numeric']:>5d}")
        print()

        print("--- Transcription by Class ---")
        class_transcriptions = Counter(t["class"] for t in all_transcriptions)
        for cls, count in class_transcriptions.most_common():
            print(f"  {cls:<20s} {count:>5d}")
        print()

        print("--- Sample Transcriptions (first 15) ---")
        for t in all_transcriptions[:15]:
            text_preview = t["text"][:50]
            if len(t["text"]) > 50:
                text_preview += "..."
            print(f"  [{t['class']:<12s}] \"{text_preview}\"")
        print()
    else:
        print("--- Transcription Analysis ---")
        print("No transcriptions found in any annotation file.")
        print()
        print("This dataset only provides bounding box annotations")
        print("without ground truth text. You can still evaluate")
        print("detection coverage (did OCR find text in the right")
        print("places?) but not text accuracy (did OCR read the")
        print("text correctly?).")
        print()
        print("To measure text accuracy, you would need to manually")
        print("transcribe at least 20-30 regions across your test set.")
        print()

    # Per-image transcription coverage analysis
    print("--- Per-Image Transcription Coverage ---")
    if per_image_coverage:
        # Coverage distribution buckets
        buckets = {
            "100%": 0,
            "80-99%": 0,
            "50-79%": 0,
            "20-49%": 0,
            "1-19%": 0,
            "0%": 0,
        }
        for img in per_image_coverage:
            pct = img["coverage_pct"]
            if pct == 100:
                buckets["100%"] += 1
            elif pct >= 80:
                buckets["80-99%"] += 1
            elif pct >= 50:
                buckets["50-79%"] += 1
            elif pct >= 20:
                buckets["20-49%"] += 1
            elif pct > 0:
                buckets["1-19%"] += 1
            else:
                buckets["0%"] += 1

        print("  Coverage distribution:")
        for bucket, count in buckets.items():
            bar = "#" * min(count, 50)
            print(f"    {bucket:>8s}: {count:>5d}  {bar}")
        print()

        # Top candidates sorted by coverage then by region count
        candidates = [
            img for img in per_image_coverage
            if img["coverage_pct"] > 0
        ]
        candidates.sort(
            key=lambda x: (x["coverage_pct"], x["with_transcription"]),
            reverse=True,
        )

        print(f"  Images with any transcription: {len(candidates)}")
        print()

        if candidates:
            print("  Top 30 candidates (highest coverage):")
            print(
                f"    {'File':<35s} "
                f"{'Regions':>7s} {'Trans':>5s} {'Cover%':>6s}"
            )
            print(f"    {'-' * 55}")
            for img in candidates[:30]:
                print(
                    f"    {img['file']:<35s} "
                    f"{img['text_regions']:>7d} "
                    f"{img['with_transcription']:>5d} "
                    f"{img['coverage_pct']:>5.1f}%"
                )
            print()

            # Thresholds summary
            for threshold in [80, 50, 30]:
                qualifying = [
                    img for img in candidates
                    if img["coverage_pct"] >= threshold
                ]
                print(
                    f"  Images with >= {threshold}% coverage: "
                    f"{len(qualifying)}"
                )
            print()

        # Save candidate list to JSON
        output_dir = path if path.is_dir() else path.parent
        candidates_path = output_dir / "annotation_candidates.json"
        candidates_export = [
            {
                "file": img["file"],
                "text_regions": img["text_regions"],
                "with_transcription": img["with_transcription"],
                "coverage_pct": round(img["coverage_pct"], 1),
            }
            for img in candidates
        ]
        with open(candidates_path, "w", encoding="utf-8") as f:
            json.dump(candidates_export, f, indent=2, ensure_ascii=False)
        print(f"  Candidate list saved to: {candidates_path}")
        print()

    # Usability assessment
    print("--- Data Usability Assessment ---")
    if total_text_regions == 0:
        print("  No text regions found. Check annotation format.")
    elif total_with_transcription == 0:
        print("  DETECTION ONLY: Can measure if OCR finds text regions,")
        print("  but cannot measure text reading accuracy.")
        print("  Recommendation: Manually annotate 20-30 transcriptions")
        print("  for a representative subset.")
    else:
        high_coverage = [
            img for img in per_image_coverage
            if img["coverage_pct"] >= 80
        ]
        medium_coverage = [
            img for img in per_image_coverage
            if img["coverage_pct"] >= 50
        ]

        if len(high_coverage) >= 30:
            print(f"  GOOD: {len(high_coverage)} images with >= 80% transcription coverage.")
            print("  Sufficient for LLM-assisted ground truth generation.")
            print("  Recommended approach: feed these to a large LLM for")
            print("  structured extraction, then review manually.")
        elif len(medium_coverage) >= 30:
            print(f"  MODERATE: {len(medium_coverage)} images with >= 50% coverage.")
            print("  Can use for LLM-assisted annotation, but expect gaps.")
            print("  Consider supplementing with OCR output for missing regions.")
        elif len(high_coverage) > 0 or len(medium_coverage) > 0:
            print(f"  LIMITED: Only {len(high_coverage)} images with >= 80% coverage,")
            print(f"  {len(medium_coverage)} with >= 50% coverage.")
            print("  Consider using VLM (image-based) annotation instead of")
            print("  text-based LLM annotation for better coverage.")
        else:
            print("  INSUFFICIENT: No images with meaningful transcription coverage.")
            print("  Recommend using VLM-based annotation directly from images.")

    print()
    print(f"{'=' * 60}")
    print("Assessment complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
