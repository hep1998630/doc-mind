"""
Ground Truth Review App for DocMind.

A Streamlit-based interface for reviewing, editing, and validating
LLM-generated ground truth extraction data.

Usage:
    streamlit run scripts/review_app.py -- <ground_truth_dir> <image_dir>

Example:
    streamlit run scripts/review_app.py -- ground_truth/ samples/invoices/
"""

import json
import sys
from pathlib import Path

import streamlit as st


# --- Configuration ---

def get_dirs():
    """
    Get directories from command line args or session state.

    Streamlit passes args after '--' in sys.argv. Tries multiple
    parsing strategies since Streamlit's arg handling varies by version.
    """
    # Check session state first (set by UI input)
    if "gt_dir" in st.session_state and "image_dir" in st.session_state:
        gt = st.session_state.get("gt_dir")
        img = st.session_state.get("image_dir")
        if gt and img:
            return Path(gt), Path(img)

    args = sys.argv[1:]

    # Strategy 1: args after '--' separator
    if "--" in args:
        idx = args.index("--")
        remaining = args[idx + 1:]
        if len(remaining) >= 2:
            return Path(remaining[0]), Path(remaining[1])

    # Strategy 2: last two args that look like paths (not streamlit flags)
    path_args = [
        a for a in args
        if not a.startswith("-") and not a.endswith(".py")
    ]
    if len(path_args) >= 2:
        return Path(path_args[-2]), Path(path_args[-1])

    return None, None


# --- Data Loading ---

@st.cache_data
def load_gt_files(gt_dir_str):
    """Load all ground truth JSON files."""
    gt_dir = Path(gt_dir_str)
    files = sorted(gt_dir.glob("*_gt.json"))
    return [str(f) for f in files]


def load_entry(file_path):
    """Load a single ground truth entry."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_entry(file_path, entry):
    """Save a ground truth entry back to disk."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)


def get_progress(gt_files):
    """Calculate review progress across all files."""
    total = len(gt_files)
    reviewed = 0
    for fp in gt_files:
        try:
            entry = load_entry(fp)
            if entry.get("review_status") == "reviewed":
                reviewed += 1
        except Exception:
            pass
    return reviewed, total


# --- UI Components ---

def render_header(gt_dir, image_dir, gt_files, current_idx):
    """Render the app header with progress and navigation."""
    reviewed, total = get_progress(gt_files)

    st.title("DocMind Ground Truth Review")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.caption(f"GT dir: `{gt_dir}`")
    with col2:
        st.caption(f"Image dir: `{image_dir}`")
    with col3:
        st.caption(f"Progress: {reviewed}/{total}")

    st.progress(reviewed / total if total > 0 else 0)


def render_navigation(gt_files, current_idx):
    """Render navigation controls."""
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        if st.button("← Previous", disabled=current_idx <= 0, use_container_width=True):
            st.session_state.current_idx = current_idx - 1
            st.rerun()

    with col2:
        if st.button("Next →", disabled=current_idx >= len(gt_files) - 1, use_container_width=True):
            st.session_state.current_idx = current_idx + 1
            st.rerun()

    with col3:
        selected = st.selectbox(
            "Jump to",
            options=range(len(gt_files)),
            index=current_idx,
            format_func=lambda i: f"{i + 1}. {Path(gt_files[i]).stem}",
            label_visibility="collapsed",
        )
        if selected != current_idx:
            st.session_state.current_idx = selected
            st.rerun()

    with col4:
        entry = load_entry(gt_files[current_idx])
        status = entry.get("review_status", "pending")
        if status == "reviewed":
            st.success("Reviewed ✓")
        else:
            st.warning("Pending")


def render_image(image_dir, image_file):
    """Render the invoice image."""
    image_path = image_dir / image_file

    if image_path.exists():
        st.image(str(image_path), use_container_width=True)
    else:
        st.error(f"Image not found: {image_path}")
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            alt_path = image_dir / f"{Path(image_file).stem}{ext}"
            if alt_path.exists():
                st.image(str(alt_path), use_container_width=True)
                break


def render_metadata(entry):
    """Render source metadata."""
    sources = entry.get("text_sources", {})
    cols = st.columns(3)
    with cols[0]:
        st.metric("GT Regions", sources.get("ground_truth_regions", 0))
    with cols[1]:
        st.metric("OCR Regions", sources.get("ocr_regions", 0))
    with cols[2]:
        st.metric("Model", entry.get("model_used", "unknown"))


def render_fields_editor(entry, image_key, file_path):
    """Render editable fields and return updated values."""
    st.subheader("Fields")

    fields = entry.get("fields", [])
    updated_fields = []

    for i, field in enumerate(fields):
        col1, col2, col3 = st.columns([2, 3, 1])

        with col1:
            field_name = st.text_input(
                "Field name",
                value=field.get("field_name", ""),
                key=f"{image_key}_field_name_{i}",
                label_visibility="collapsed",
                disabled=True,
            )

        with col2:
            value = st.text_input(
                "Value",
                value=str(field.get("value", "") or ""),
                key=f"{image_key}_field_value_{i}",
                label_visibility="collapsed",
            )

        with col3:
            verified = st.checkbox(
                "✓",
                value=field.get("verified", False),
                key=f"{image_key}_field_verified_{i}",
            )

        updated_fields.append({
            "field_name": field.get("field_name", ""),
            "value": value if value else None,
            "verified": verified,
        })

    # Add new field button — saves directly and reruns
    if st.button("+ Add field", key=f"{image_key}_add_field"):
        entry["fields"].append({
            "field_name": "new_field",
            "value": "",
            "verified": False,
        })
        save_entry(file_path, entry)
        st.rerun()

    return updated_fields


def render_line_items_editor(entry, image_key, file_path):
    """Render editable line items and return updated values."""
    st.subheader("Line Items")

    line_items = entry.get("line_items", [])
    updated_items = []
    items_to_delete = []

    for i, item in enumerate(line_items):
        with st.expander(
            f"Item {i + 1}: {item.get('description', '')[:40]}",
            expanded=not item.get("verified", False),
        ):
            col1, col2 = st.columns(2)

            with col1:
                description = st.text_input(
                    "Description",
                    value=item.get("description", ""),
                    key=f"{image_key}_item_desc_{i}",
                )
                quantity = st.text_input(
                    "Quantity",
                    value=str(item.get("quantity", "") or ""),
                    key=f"{image_key}_item_qty_{i}",
                )
                item_code = st.text_input(
                    "Item Code",
                    value=str(item.get("item_code", "") or ""),
                    key=f"{image_key}_item_code_{i}",
                )

            with col2:
                amount = st.text_input(
                    "Amount",
                    value=str(item.get("amount", "") or ""),
                    key=f"{image_key}_item_amount_{i}",
                )
                unit_price = st.text_input(
                    "Unit Price",
                    value=str(item.get("unit_price", "") or ""),
                    key=f"{image_key}_item_price_{i}",
                )
                verified = st.checkbox(
                    "Verified ✓",
                    value=item.get("verified", False),
                    key=f"{image_key}_item_verified_{i}",
                )

            delete = st.button(
                "Delete this item",
                key=f"{image_key}_delete_item_{i}",
                type="secondary",
            )
            if delete:
                # Remove from entry directly and save
                entry["line_items"].pop(i)
                save_entry(file_path, entry)
                st.rerun()

            # Parse numeric values safely
            def safe_float(val):
                if not val or val.strip() == "":
                    return None
                try:
                    return float(val)
                except ValueError:
                    return val

            updated_items.append({
                "description": description,
                "amount": safe_float(amount),
                "quantity": safe_float(quantity),
                "unit_price": safe_float(unit_price),
                "item_code": item_code if item_code.strip() else None,
                "verified": verified,
            })

    # Add new item button — saves directly and reruns
    if st.button("+ Add line item", key=f"{image_key}_add_item"):
        entry["line_items"].append({
            "description": "",
            "amount": None,
            "quantity": None,
            "unit_price": None,
            "item_code": None,
            "verified": False,
        })
        save_entry(file_path, entry)
        st.rerun()

    return updated_items


def render_raw_input(entry):
    """Render the raw text that was sent to the LLM."""
    with st.expander("Raw LLM Input (for reference)"):
        raw = entry.get("raw_input", "")
        st.code(raw, language=None)


# --- Main App ---

def main():
    st.set_page_config(
        page_title="DocMind Ground Truth Review",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    gt_dir, image_dir = get_dirs()

    if gt_dir is None or image_dir is None:
        st.title("DocMind Ground Truth Review")
        st.info("Please provide the directory paths to get started.")

        gt_input = st.text_input(
            "Ground truth directory",
            placeholder="e.g., ground_truth/",
        )
        img_input = st.text_input(
            "Image directory",
            placeholder="e.g., samples/invoices/",
        )

        if st.button("Start Review", type="primary"):
            if gt_input and img_input:
                st.session_state.gt_dir = gt_input
                st.session_state.image_dir = img_input
                st.rerun()
            else:
                st.error("Please fill in both directories.")

        st.divider()
        st.caption(
            "Or run from command line: "
            "`streamlit run scripts/review_app.py -- ground_truth/ samples/invoices/`"
        )
        return

    if not gt_dir.exists():
        st.error(f"Ground truth directory not found: {gt_dir}")
        return
    if not image_dir.exists():
        st.error(f"Image directory not found: {image_dir}")
        return

    # Load files
    gt_files = load_gt_files(str(gt_dir))

    if not gt_files:
        st.error(f"No *_gt.json files found in {gt_dir}")
        return

    # Initialize session state
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    current_idx = st.session_state.current_idx
    current_file = gt_files[current_idx]
    entry = load_entry(current_file)

    # Render header and navigation
    render_header(gt_dir, image_dir, gt_files, current_idx)
    render_navigation(gt_files, current_idx)
    st.divider()

    # Two-column layout: image on left, fields on right
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader(f"Image: {entry.get('image_file', 'unknown')}")
        render_image(image_dir, entry.get("image_file", ""))
        render_metadata(entry)

    with right_col:
        image_key = Path(current_file).stem
        updated_fields = render_fields_editor(entry, image_key, current_file)
        updated_items = render_line_items_editor(entry, image_key, current_file)
        render_raw_input(entry)

    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("💾 Save Changes", type="primary", use_container_width=True):
            entry["fields"] = updated_fields
            entry["line_items"] = updated_items
            save_entry(current_file, entry)
            st.success("Saved!")
            st.rerun()

    with col2:
        if st.button("✅ Mark Reviewed & Save", use_container_width=True):
            entry["fields"] = updated_fields
            entry["line_items"] = updated_items
            entry["review_status"] = "reviewed"
            save_entry(current_file, entry)
            st.success("Marked as reviewed!")
            # Auto-advance to next
            if current_idx < len(gt_files) - 1:
                st.session_state.current_idx = current_idx + 1
            st.rerun()

    with col3:
        if st.button("↩ Reset to Pending", use_container_width=True):
            entry["review_status"] = "pending"
            save_entry(current_file, entry)
            st.rerun()

    with col4:
        # Show verification stats for current entry
        total_fields = len(updated_fields)
        verified_fields = sum(1 for f in updated_fields if f.get("verified"))
        total_items = len(updated_items)
        verified_items = sum(1 for i in updated_items if i.get("verified"))
        st.caption(
            f"Fields: {verified_fields}/{total_fields} verified | "
            f"Items: {verified_items}/{total_items} verified"
        )


if __name__ == "__main__":
    main()
