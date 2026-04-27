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

    for annotation_file in files:
        try:
            stats = assess_single_file(annotation_file)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  Warning: Failed to parse {annotation_file.name}: {e}")
            continue

        total_objects += stats["total_objects"]
        total_text_regions += (
            stats["with_transcription"] + stats["without_transcription"]
        )
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

    # Usability assessment
    print("--- Data Usability Assessment ---")
    if total_text_regions == 0:
        print("  No text regions found. Check annotation format.")
    elif total_with_transcription == 0:
        print("  DETECTION ONLY: Can measure if OCR finds text regions,")
        print("  but cannot measure text reading accuracy.")
        print("  Recommendation: Manually annotate 20-30 transcriptions")
        print("  for a representative subset.")
    elif total_with_transcription < 20:
        print(f"  LIMITED: Only {total_with_transcription} transcriptions available.")
        print("  Enough for a rough CER/WER estimate, but not statistically")
        print("  reliable. Consider adding more transcriptions.")
    elif total_with_transcription < 100:
        print(f"  MODERATE: {total_with_transcription} transcriptions available.")
        print("  Sufficient for meaningful CER/WER evaluation.")
    else:
        print(f"  GOOD: {total_with_transcription} transcriptions available.")
        print("  Strong evaluation dataset.")

    print()
    print(f"{'=' * 60}")
    print("Assessment complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
