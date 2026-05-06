"""
Image Selection App for DocMind.

Browse invoice images and manually select which ones are suitable
for ground truth annotation. Saves the selection to a JSON file
that can be fed to generate_ground_truth.py.

Usage:
    streamlit run select_images.py
"""

import json
import sys
from pathlib import Path

import streamlit as st


# --- Data Loading ---

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_images(image_dir):
    """Find all image files in a directory."""
    images = []
    for f in sorted(image_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def load_selection(selection_path):
    """Load existing selection file if it exists."""
    if selection_path.exists():
        with open(selection_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"selected": [], "skipped": []}


def save_selection(selection_path, selection):
    """Save selection to JSON file."""
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False)


# --- Main App ---

def main():
    st.set_page_config(
        page_title="DocMind Image Selection",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("DocMind Image Selection")
    st.caption("Browse images and select ones suitable for ground truth annotation.")

    # Directory input
    if "image_dir" not in st.session_state:
        st.session_state.image_dir = ""
    if "output_file" not in st.session_state:
        st.session_state.output_file = ""

    col1, col2 = st.columns(2)
    with col1:
        image_dir_input = st.text_input(
            "Image directory",
            value=st.session_state.image_dir,
            placeholder="e.g., samples/invoices/",
        )
    with col2:
        output_file_input = st.text_input(
            "Selection output file",
            value=st.session_state.output_file,
            placeholder="e.g., selected_images.json",
        )

    if not image_dir_input or not output_file_input:
        st.info("Enter the image directory and output file path to begin.")
        return

    image_dir = Path(image_dir_input)
    output_path = Path(output_file_input)

    if not image_dir.exists():
        st.error(f"Directory not found: {image_dir}")
        return

    st.session_state.image_dir = image_dir_input
    st.session_state.output_file = output_file_input

    # Load images and existing selection
    all_images = find_images(image_dir)
    if not all_images:
        st.error("No image files found in the directory.")
        return

    selection = load_selection(output_path)
    selected_set = set(selection.get("selected", []))
    skipped_set = set(selection.get("skipped", []))

    # Initialize navigation index
    if "select_idx" not in st.session_state:
        st.session_state.select_idx = 0

    # Filter options
    st.divider()
    filter_mode = st.radio(
        "Show",
        ["All images", "Unreviewed only", "Selected only", "Skipped only"],
        horizontal=True,
    )

    if filter_mode == "Unreviewed only":
        filtered = [
            img for img in all_images
            if img.name not in selected_set and img.name not in skipped_set
        ]
    elif filter_mode == "Selected only":
        filtered = [img for img in all_images if img.name in selected_set]
    elif filter_mode == "Skipped only":
        filtered = [img for img in all_images if img.name in skipped_set]
    else:
        filtered = all_images

    if not filtered:
        st.info("No images match the current filter.")
        st.metric("Selected", len(selected_set))
        return

    # Clamp index
    if st.session_state.select_idx >= len(filtered):
        st.session_state.select_idx = len(filtered) - 1
    current_idx = st.session_state.select_idx
    current_image = filtered[current_idx]

    # Progress bar
    progress_cols = st.columns([1, 1, 1, 1])
    with progress_cols[0]:
        st.metric("Total", len(all_images))
    with progress_cols[1]:
        st.metric("Selected", len(selected_set))
    with progress_cols[2]:
        st.metric("Skipped", len(skipped_set))
    with progress_cols[3]:
        remaining = len(all_images) - len(selected_set) - len(skipped_set)
        st.metric("Remaining", remaining)

    reviewed = len(selected_set) + len(skipped_set)
    st.progress(reviewed / len(all_images) if all_images else 0)

    # Navigation
    nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])

    with nav_col1:
        if st.button("← Previous", disabled=current_idx <= 0, use_container_width=True):
            st.session_state.select_idx = current_idx - 1
            st.rerun()
    with nav_col2:
        selected_idx = st.selectbox(
            "Jump to",
            options=range(len(filtered)),
            index=current_idx,
            format_func=lambda i: f"{i + 1}/{len(filtered)}: {filtered[i].name}",
            label_visibility="collapsed",
        )
        if selected_idx != current_idx:
            st.session_state.select_idx = selected_idx
            st.rerun()
    with nav_col3:
        if st.button("Next →", disabled=current_idx >= len(filtered) - 1, use_container_width=True):
            st.session_state.select_idx = current_idx + 1
            st.rerun()

    st.divider()

    # Image display and status
    img_col, action_col = st.columns([3, 1])

    with img_col:
        # Show current status
        if current_image.name in selected_set:
            st.success(f"✓ SELECTED: {current_image.name}")
        elif current_image.name in skipped_set:
            st.warning(f"✗ SKIPPED: {current_image.name}")
        else:
            st.info(f"⊘ UNREVIEWED: {current_image.name}")

        st.image(str(current_image), use_container_width=True)

    with action_col:
        st.subheader("Actions")

        # Select button
        if st.button(
            "✅ Select",
            type="primary",
            use_container_width=True,
            key="select_btn",
        ):
            selected_set.add(current_image.name)
            skipped_set.discard(current_image.name)
            selection["selected"] = sorted(list(selected_set))
            selection["skipped"] = sorted(list(skipped_set))
            save_selection(output_path, selection)
            # Auto-advance
            if current_idx < len(filtered) - 1:
                st.session_state.select_idx = current_idx + 1
            st.rerun()

        # Skip button
        if st.button(
            "⏭ Skip",
            use_container_width=True,
            key="skip_btn",
        ):
            skipped_set.add(current_image.name)
            selected_set.discard(current_image.name)
            selection["selected"] = sorted(list(selected_set))
            selection["skipped"] = sorted(list(skipped_set))
            save_selection(output_path, selection)
            # Auto-advance
            if current_idx < len(filtered) - 1:
                st.session_state.select_idx = current_idx + 1
            st.rerun()

        # Undo button
        if current_image.name in selected_set or current_image.name in skipped_set:
            if st.button(
                "↩ Undo",
                use_container_width=True,
                key="undo_btn",
            ):
                selected_set.discard(current_image.name)
                skipped_set.discard(current_image.name)
                selection["selected"] = sorted(list(selected_set))
                selection["skipped"] = sorted(list(skipped_set))
                save_selection(output_path, selection)
                st.rerun()

        st.divider()

        # Quick stats
        st.caption(f"Image: {current_idx + 1}/{len(filtered)}")
        st.caption(f"File: {current_image.name}")

        # File size
        size_kb = current_image.stat().st_size / 1024
        st.caption(f"Size: {size_kb:.0f} KB")

    # Footer with selection file info
    st.divider()
    st.caption(
        f"Selection saved to: `{output_path}` | "
        f"Use with: `python generate_ground_truth.py ... --selection {output_path}`"
    )


if __name__ == "__main__":
    main()
