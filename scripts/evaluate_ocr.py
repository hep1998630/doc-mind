"""
OCR Evaluation Script for DocMind.

Evaluates OCR accuracy against ground truth annotations. Supports
the Supervisely-style annotation format with optional transcriptions.

Metrics calculated:
    Detection:
        - Detection coverage: % of ground truth regions matched by OCR
        - False positive rate: OCR regions not matching any ground truth
        - Containment: how well OCR regions sit within GT regions

    Text accuracy (when transcriptions are available):
        - Character Error Rate (CER): edit distance / reference length
        - Word Error Rate (WER): word-level edit distance / reference words
        - Exact match rate: % of regions with perfect transcription

Usage:
    # Evaluate a single image
    python scripts/evaluate_ocr.py <image_path> <annotation_json_path>

    # Evaluate a directory of images
    python scripts/evaluate_ocr.py <image_dir> <annotation_dir>

    # With custom settings
    python scripts/evaluate_ocr.py <image_path> <annotation_json> --containment-threshold 0.5

Examples:
    python scripts/evaluate_ocr.py samples/invoices/001.jpg samples/annotations/001.json
    python scripts/evaluate_ocr.py samples/invoices/ samples/annotations/
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docmind.config.settings import get_settings
from docmind.models.common import BoundingBox
from docmind.modules.preprocessing.processor import ImagePreprocessor
from docmind.modules.ocr.paddle_ocr import PaddleOCREngine

logging.basicConfig(level=logging.WARNING)


# --- Data Classes ---

@dataclass
class GroundTruthRegion:
    """A single ground truth annotation region."""
    class_title: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    transcription: str | None = None


@dataclass
class MatchResult:
    """Result of matching a ground truth region to an OCR region."""
    gt_region: GroundTruthRegion
    ocr_text: str | None = None
    iou: float = 0.0
    matched: bool = False


@dataclass
class ImageEvaluation:
    """Evaluation results for a single image."""
    image_path: str
    gt_region_count: int = 0
    ocr_region_count: int = 0
    matched_count: int = 0
    unmatched_gt_count: int = 0
    false_positive_count: int = 0
    iou_scores: list[float] = field(default_factory=list)
    cer_scores: list[float] = field(default_factory=list)
    wer_scores: list[float] = field(default_factory=list)
    exact_matches: int = 0
    transcription_count: int = 0
    match_details: list[MatchResult] = field(default_factory=list)


@dataclass
class DatasetEvaluation:
    """Aggregated evaluation results across all images."""
    image_results: list[ImageEvaluation] = field(default_factory=list)

    @property
    def total_gt_regions(self) -> int:
        return sum(r.gt_region_count for r in self.image_results)

    @property
    def total_ocr_regions(self) -> int:
        return sum(r.ocr_region_count for r in self.image_results)

    @property
    def total_matched(self) -> int:
        return sum(r.matched_count for r in self.image_results)

    @property
    def total_false_positives(self) -> int:
        return sum(r.false_positive_count for r in self.image_results)

    @property
    def detection_coverage(self) -> float:
        if self.total_gt_regions == 0:
            return 0.0
        return self.total_matched / self.total_gt_regions

    @property
    def false_positive_rate(self) -> float:
        if self.total_ocr_regions == 0:
            return 0.0
        return self.total_false_positives / self.total_ocr_regions

    @property
    def mean_iou(self) -> float:
        all_ious = [s for r in self.image_results for s in r.iou_scores]
        if not all_ious:
            return 0.0
        return float(np.mean(all_ious))

    @property
    def mean_cer(self) -> float:
        all_cers = [s for r in self.image_results for s in r.cer_scores]
        if not all_cers:
            return 0.0
        return float(np.mean(all_cers))

    @property
    def mean_wer(self) -> float:
        all_wers = [s for r in self.image_results for s in r.wer_scores]
        if not all_wers:
            return 0.0
        return float(np.mean(all_wers))

    @property
    def total_transcriptions(self) -> int:
        return sum(r.transcription_count for r in self.image_results)

    @property
    def total_exact_matches(self) -> int:
        return sum(r.exact_matches for r in self.image_results)

    @property
    def exact_match_rate(self) -> float:
        if self.total_transcriptions == 0:
            return 0.0
        return self.total_exact_matches / self.total_transcriptions


# --- Text Metrics ---

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

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


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate.

    CER = edit_distance(ref, hyp) / len(ref)

    Returns 1.0 if reference is empty.
    """
    reference = reference.strip()
    hypothesis = hypothesis.strip()

    if not reference:
        return 1.0 if hypothesis else 0.0

    distance = _levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate.

    WER = edit_distance(ref_words, hyp_words) / len(ref_words)

    Returns 1.0 if reference is empty.
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 1.0 if hyp_words else 0.0

    distance = _levenshtein_distance(
        " ".join(ref_words), " ".join(hyp_words)
    )
    ref_length = sum(len(w) for w in ref_words) + len(ref_words) - 1
    return distance / ref_length if ref_length > 0 else 0.0


# --- Spatial Metrics ---

def compute_containment(
    inner_box: tuple[float, float, float, float],
    outer_box: tuple[float, float, float, float],
) -> float:
    """
    Compute what fraction of inner_box is contained within outer_box.

    containment = intersection_area / inner_box_area

    This handles granularity mismatch: a small OCR word box inside
    a large GT line box will have high containment even though IoU
    would be low.
    """
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    inner_area = (
        (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    )

    if inner_area == 0:
        return 0.0

    return intersection / inner_area


# --- Annotation Parsing ---

def parse_annotation(annotation_path: Path) -> list[GroundTruthRegion]:
    """
    Parse a Supervisely-style annotation JSON file.

    Extracts bounding boxes and any available transcriptions.
    Skips non-text regions (e.g., "Page" class).
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions: list[GroundTruthRegion] = []

    for obj in data.get("objects", []):
        class_title = obj.get("classTitle", "")

        # Skip page-level annotations — they're not text regions
        if class_title.lower() == "page":
            continue

        # Skip non-rectangle geometries (like page polygons)
        if obj.get("geometryType") != "rectangle":
            continue

        points = obj.get("points", {}).get("exterior", [])
        if len(points) != 2:
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        # Extract transcription if available
        transcription = None
        for tag in obj.get("tags", []):
            if tag.get("name") == "Transcription":
                transcription = tag.get("value", "")
                break

        regions.append(GroundTruthRegion(
            class_title=class_title,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            transcription=transcription,
        ))

    return regions


# --- Evaluation ---

def evaluate_image(
    image_path: Path,
    annotation_path: Path,
    preprocessor: ImagePreprocessor,
    ocr_engine: PaddleOCREngine,
    containment_threshold: float = 0.5,
) -> ImageEvaluation:
    """
    Evaluate OCR performance on a single image against ground truth.

    Uses containment-based matching: an OCR region is assigned to a
    GT region if most of the OCR box falls within the GT box. This
    handles the common case where GT annotations are line-level but
    OCR detections are word-level.

    For each GT region, all contained OCR texts are combined (sorted
    in reading order) and compared against the transcription.
    """
    evaluation = ImageEvaluation(image_path=str(image_path))

    # Parse ground truth
    gt_regions = parse_annotation(annotation_path)
    evaluation.gt_region_count = len(gt_regions)

    if not gt_regions:
        return evaluation

    # Run OCR pipeline
    processed_image, _ = preprocessor.process(str(image_path))
    ocr_result = ocr_engine.recognize(processed_image)
    evaluation.ocr_region_count = len(ocr_result.text_regions)

    # Build OCR bounding boxes in (x1, y1, x2, y2) format
    ocr_boxes = []
    for region in ocr_result.text_regions:
        ocr_boxes.append({
            "bbox": region.bbox.to_xyxy(),
            "text": region.text,
            "used": False,
            "direction": region.script_direction.value,
        })

    # For each GT region, find all OCR regions contained within it
    for gt_region in gt_regions:
        contained_texts: list[tuple[float, int, str]] = []

        for i, ocr_box in enumerate(ocr_boxes):
            containment = compute_containment(
                ocr_box["bbox"], gt_region.bbox
            )
            if containment >= containment_threshold:
                contained_texts.append((containment, i, ocr_box["text"]))

        match = MatchResult(gt_region=gt_region)

        if contained_texts:
            match.matched = True
            evaluation.matched_count += 1

            # Average containment as quality score
            avg_containment = sum(c for c, _, _ in contained_texts) / len(contained_texts)
            match.iou = avg_containment
            evaluation.iou_scores.append(avg_containment)

            # Sort OCR texts in reading order within this GT region.
            # Use y-center first (top to bottom), then x-center
            # (direction-aware).
            def sort_key(item):
                idx = item[1]
                box = ocr_boxes[idx]["bbox"]
                y_center = (box[1] + box[3]) / 2
                x_center = (box[0] + box[2]) / 2
                return (y_center, x_center)

            contained_texts.sort(key=sort_key)
            combined_text = " ".join(t for _, _, t in contained_texts)
            match.ocr_text = combined_text

            # Mark OCR regions as used
            for _, idx, _ in contained_texts:
                ocr_boxes[idx]["used"] = True

            # Calculate text metrics if transcription is available
            if gt_region.transcription is not None:
                evaluation.transcription_count += 1

                cer = compute_cer(gt_region.transcription, combined_text)
                wer = compute_wer(gt_region.transcription, combined_text)
                evaluation.cer_scores.append(cer)
                evaluation.wer_scores.append(wer)

                if gt_region.transcription.strip() == combined_text.strip():
                    evaluation.exact_matches += 1

        evaluation.match_details.append(match)

    # Count unmatched and false positives
    evaluation.unmatched_gt_count = (
        evaluation.gt_region_count - evaluation.matched_count
    )
    evaluation.false_positive_count = sum(
        1 for ocr_box in ocr_boxes if not ocr_box["used"]
    )

    return evaluation


def evaluate_dataset(
    image_paths: list[Path],
    annotation_paths: list[Path],
    preprocessor: ImagePreprocessor,
    ocr_engine: PaddleOCREngine,
    containment_threshold: float = 0.5,
) -> DatasetEvaluation:
    """Evaluate OCR performance across a dataset of images."""
    dataset_eval = DatasetEvaluation()

    for i, (img_path, ann_path) in enumerate(
        zip(image_paths, annotation_paths)
    ):
        print(f"  [{i + 1}/{len(image_paths)}] {img_path.name}...", end=" ")
        try:
            result = evaluate_image(
                img_path, ann_path, preprocessor, ocr_engine,
                containment_threshold,
            )
            dataset_eval.image_results.append(result)
            print(
                f"GT={result.gt_region_count} "
                f"OCR={result.ocr_region_count} "
                f"Matched={result.matched_count}"
            )
        except Exception as e:
            print(f"ERROR: {e}")

    return dataset_eval


# --- Reporting ---

def print_report(evaluation: DatasetEvaluation) -> None:
    """Print a comprehensive evaluation report."""
    print()
    print(f"{'=' * 60}")
    print("OCR EVALUATION REPORT")
    print(f"{'=' * 60}")
    print()

    print(f"Images evaluated: {len(evaluation.image_results)}")
    print()

    # Detection metrics
    print("--- Detection Metrics ---")
    print(f"Ground truth regions:  {evaluation.total_gt_regions}")
    print(f"OCR detected regions:  {evaluation.total_ocr_regions}")
    print(f"Matched regions:       {evaluation.total_matched}")
    print(f"Unmatched GT regions:  {evaluation.total_gt_regions - evaluation.total_matched}")
    print(f"False positive OCR:    {evaluation.total_false_positives}")
    print()
    print(f"Detection coverage:    {evaluation.detection_coverage:.1%}")
    print(f"False positive rate:   {evaluation.false_positive_rate:.1%}")
    print(f"Mean containment:      {evaluation.mean_iou:.3f}")
    print()

    # Text accuracy metrics
    if evaluation.total_transcriptions > 0:
        print("--- Text Accuracy Metrics ---")
        print(f"Regions with transcription: {evaluation.total_transcriptions}")
        print(f"Mean CER:              {evaluation.mean_cer:.3f} ({evaluation.mean_cer:.1%})")
        print(f"Mean WER:              {evaluation.mean_wer:.3f} ({evaluation.mean_wer:.1%})")
        print(f"Exact match rate:      {evaluation.exact_match_rate:.1%}")
        print()
    else:
        print("--- Text Accuracy Metrics ---")
        print("No transcriptions available in ground truth.")
        print("Text accuracy metrics cannot be calculated.")
        print("Consider annotating a subset with transcriptions")
        print("for proper accuracy measurement.")
        print()

    # Per-image breakdown
    print("--- Per-Image Breakdown ---")
    print(
        f"  {'Image':<30s} "
        f"{'GT':>4s} {'OCR':>4s} {'Match':>5s} "
        f"{'Cov%':>5s} {'Cont':>5s} "
        f"{'CER':>5s} {'WER':>5s}"
    )
    print(f"  {'-' * 78}")

    for result in evaluation.image_results:
        img_name = Path(result.image_path).name[:30]
        coverage = (
            result.matched_count / result.gt_region_count * 100
            if result.gt_region_count > 0 else 0
        )
        mean_iou = (
            float(np.mean(result.iou_scores))
            if result.iou_scores else 0
        )
        mean_cer = (
            float(np.mean(result.cer_scores))
            if result.cer_scores else -1
        )
        mean_wer = (
            float(np.mean(result.wer_scores))
            if result.wer_scores else -1
        )

        cer_str = f"{mean_cer:.3f}" if mean_cer >= 0 else "  n/a"
        wer_str = f"{mean_wer:.3f}" if mean_wer >= 0 else "  n/a"

        print(
            f"  {img_name:<30s} "
            f"{result.gt_region_count:>4d} {result.ocr_region_count:>4d} "
            f"{result.matched_count:>5d} "
            f"{coverage:>4.0f}% "
            f"{mean_iou:>5.3f} "
            f"{cer_str:>5s} {wer_str:>5s}"
        )

    print()

    # Detailed match report for last image (most recent)
    if evaluation.image_results:
        last = evaluation.image_results[-1]
        print(f"--- Detailed Matches: {Path(last.image_path).name} ---")
        for match in last.match_details:
            gt = match.gt_region
            status = "MATCH" if match.matched else "MISS "
            x1, y1, x2, y2 = gt.bbox
            print(
                f"  [{status}] {gt.class_title:<12s} "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  "
                f"IoU={match.iou:.3f}"
            )
            if gt.transcription:
                print(f"           GT:  \"{gt.transcription}\"")
            if match.ocr_text:
                print(f"           OCR: \"{match.ocr_text}\"")
        print()


def save_report(evaluation: DatasetEvaluation, output_path: Path) -> None:
    """Save evaluation results as JSON."""
    report = {
        "summary": {
            "images_evaluated": len(evaluation.image_results),
            "total_gt_regions": evaluation.total_gt_regions,
            "total_ocr_regions": evaluation.total_ocr_regions,
            "total_matched": evaluation.total_matched,
            "detection_coverage": evaluation.detection_coverage,
            "false_positive_rate": evaluation.false_positive_rate,
            "mean_containment": evaluation.mean_iou,
            "transcriptions_available": evaluation.total_transcriptions,
            "mean_cer": evaluation.mean_cer if evaluation.total_transcriptions > 0 else None,
            "mean_wer": evaluation.mean_wer if evaluation.total_transcriptions > 0 else None,
            "exact_match_rate": evaluation.exact_match_rate if evaluation.total_transcriptions > 0 else None,
        },
        "per_image": [
            {
                "image": r.image_path,
                "gt_regions": r.gt_region_count,
                "ocr_regions": r.ocr_region_count,
                "matched": r.matched_count,
                "false_positives": r.false_positive_count,
                "mean_containment": float(np.mean(r.iou_scores)) if r.iou_scores else 0,
                "mean_cer": float(np.mean(r.cer_scores)) if r.cer_scores else None,
                "mean_wer": float(np.mean(r.wer_scores)) if r.wer_scores else None,
            }
            for r in evaluation.image_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {output_path}")


# --- Main ---

def find_image_annotation_pairs(
    image_path: Path, annotation_path: Path
) -> list[tuple[Path, Path]]:
    """
    Find matching image-annotation pairs.

    Supports both single file and directory inputs. For directories,
    matches files by stem name (e.g., 001.jpg <-> 001.json).
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    if image_path.is_file():
        return [(image_path, annotation_path)]

    if image_path.is_dir() and annotation_path.is_dir():
        pairs = []
        for img_file in sorted(image_path.iterdir()):
            if img_file.suffix.lower() not in image_extensions:
                continue
            ann_file = annotation_path / f"{img_file.stem}.json"
            if ann_file.exists():
                pairs.append((img_file, ann_file))
            else:
                print(f"  Warning: No annotation for {img_file.name}")
        return pairs

    print("Error: Both paths must be files or both must be directories.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR accuracy against ground truth annotations.",
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file or directory of images.",
    )
    parser.add_argument(
        "annotation_path",
        type=str,
        help="Path to annotation JSON file or directory of annotations.",
    )
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=0.5,
        help="Containment threshold for matching OCR to GT regions (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON evaluation report.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)
    annotation_path = Path(args.annotation_path)

    if not image_path.exists():
        print(f"Error: {image_path} not found.")
        sys.exit(1)
    if not annotation_path.exists():
        print(f"Error: {annotation_path} not found.")
        sys.exit(1)

    # Find image-annotation pairs
    pairs = find_image_annotation_pairs(image_path, annotation_path)
    if not pairs:
        print("No image-annotation pairs found.")
        sys.exit(1)

    print(f"{'=' * 60}")
    print("DocMind OCR Evaluation")
    print(f"{'=' * 60}")
    print(f"Found {len(pairs)} image-annotation pair(s)")
    print(f"Containment threshold: {args.containment_threshold}")
    print()

    # Initialize pipeline from settings
    settings = get_settings()
    preprocessor = ImagePreprocessor(settings=settings.preprocessing)
    ocr_engine = PaddleOCREngine(settings=settings.ocr)

    print("--- Processing Images ---")
    image_paths = [p[0] for p in pairs]
    annotation_paths = [p[1] for p in pairs]

    evaluation = evaluate_dataset(
        image_paths, annotation_paths,
        preprocessor, ocr_engine,
        containment_threshold=args.containment_threshold,
    )

    # Print report
    print_report(evaluation)

    # Save report if requested
    if args.output:
        save_report(evaluation, Path(args.output))
    else:
        # Default output next to the input
        if image_path.is_dir():
            output_path = image_path / "ocr_evaluation_report.json"
        else:
            output_path = image_path.parent / "ocr_evaluation_report.json"
        save_report(evaluation, output_path)

    print()
    print(f"{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
