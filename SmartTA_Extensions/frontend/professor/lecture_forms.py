"""Form UI components for adding YouTube lectures.

This module handles the user interface for lecture addition forms,
including validation, category selection, and input rendering.
"""
import re
from typing import Dict, Any

import streamlit as st

from SmartTA_Extensions.utils.helpers import (
    load_youtube_lectures,
    extract_youtube_id,
)

# VALIDATION HELPERS

def validate_youtube_url(url: str) -> tuple[bool, str]:
    """Validate YouTube URL and return (is_valid, message)."""
    if not url:
        return False, ""
    video_id = extract_youtube_id(url)
    if not video_id:
        return False, "⚠️ Invalid YouTube URL format. Please use a valid link (e.g., youtube.com/watch?v=... or youtu.be/...)"
    return True, ""


def normalize_lecture_id(text: str) -> str:
    """Normalize text to valid lecture ID format."""
    if not text:
        return ""
    normalized = re.sub(r'[^a-zA-Z0-9\s\-]', '', text.strip())
    return re.sub(r'\s+', '_', normalized).strip('_')


def check_lecture_conflicts(
    lecture_title: str,
    custom_id: str,
    yt_data: dict
) -> tuple[bool, bool, str]:
    """Check for ID and title conflicts. Returns (id_conflict, title_conflict, normalized_id)."""
    existing_ids = set(yt_data.get("lectures", {}).keys())
    existing_titles = set(
        info.get("title", "").lower().strip() 
        for info in yt_data.get("lectures", {}).values()
    )
    
    normalized_id = normalize_lecture_id(custom_id or lecture_title)
    id_conflict = normalized_id in existing_ids if normalized_id else False
    title_conflict = lecture_title.lower().strip() in existing_titles if lecture_title else False
    
    return id_conflict, title_conflict, normalized_id

# UI COMPONENT RENDERERS
def render_category_selector(lecture_categories: dict) -> str | None:
    """Render category selection UI and return selected category."""
    upload_categories = sorted(lecture_categories.keys()) if lecture_categories else []
    
    if upload_categories:
        category_options = ["— Select Category —"] + upload_categories + ["+ Create New Category"]
        category_choice = st.selectbox(
            "Category:",
            options=category_options,
            index=0,
            help="Choose which category for this lecture or create a new one"
        )
        
        if category_choice == "— Select Category —":
            return None
        elif category_choice == "+ Create New Category":
            new_category = st.text_input(
                "Enter new category name:",
                placeholder="e.g., Deep Learning"
            )
            new_category_stripped = new_category.strip() if new_category else ""
            
            # Check if category already exists (case-insensitive)
            if new_category_stripped:
                existing_categories_lower = {cat.lower() for cat in upload_categories}
                if new_category_stripped.lower() in existing_categories_lower:
                    st.warning(f"⚠️ Category '{new_category_stripped}' already exists. Please select it from the dropdown or choose a different name.")
                    return None
            
            return new_category_stripped if new_category_stripped else None
        else:
            return category_choice
    else:
        first_category = st.text_input(
            "Create first category:",
            placeholder="e.g., Supervised Learning"
        )
        return first_category.strip() if first_category else None


def render_lecture_details(
    lecture_title: str,
    youtube_url: str
) -> tuple[str, str, str, str]:
    """Render lecture detail inputs and return (title, custom_id, url, note)."""
    title = st.text_input(
        "Lecture Title:",
        value=lecture_title,
        placeholder="e.g., 01. Introduction to Machine Learning"
    )
    
    suggested_id = normalize_lecture_id(title)
    custom_id = st.text_input(
        "Optional: Custom Lecture ID (filename)",
        value=suggested_id,
        placeholder="e.g., 01_introduction_to_ml",
        help="Auto-suggested from the title. You can override. Used as filename under data/lectures.",
    )
    
    url = st.text_input(
        "YouTube URL:",
        value=youtube_url,
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Real-time URL validation
    is_valid, message = validate_youtube_url(url)
    if url and not is_valid:
        st.warning(message)
    
    note = st.text_area(
        "Professor's Note (optional):",
        placeholder="Key topics: gradient descent, learning rate...",
        help="This note will be shown to students when they view this lecture",
    )
    
    return title, custom_id, url, note


def render_submit_controls(
    can_submit: bool,
    id_conflict: bool,
    title_conflict: bool,
    normalized_id: str,
    lecture_title: str
) -> bool:
    """Render submit button and validation warnings. Returns True if submitted."""
    col_submit, col_warn = st.columns([1, 2])
    
    with col_submit:
        submit_help = None if can_submit else "Enter title, URL, choose category, ensure unique ID and title"
        trigger_submit = st.button(
            "Add & Transcribe Lecture",
            type="primary",
            disabled=not can_submit,
            help=submit_help
        )
    
    with col_warn:
        if id_conflict:
            st.warning(f"⚠️ Lecture ID '{normalized_id}' already exists. Change the Custom Lecture ID.")
        elif title_conflict:
            st.warning(f"⚠️ A lecture with title '{lecture_title}' already exists. Choose a different title.")
    
    return trigger_submit

# MAIN FORM COMPONENT

def render_add_lecture_form(
    lecture_categories: dict[str, list[str]],
    on_submit_callback
) -> None:
    """Render the complete 'Add YouTube Lecture' form.
    
    Args:
        lecture_categories: Dict of category names to lecture lists
        on_submit_callback: Function to call with (title, url, category, custom_id, note) when submitted
    """
    st.markdown("---")
    st.markdown('<div id="add_lecture"></div>', unsafe_allow_html=True)
    st.subheader("📺 Add YouTube Lecture")
    st.info("Upload your lecture to YouTube as Unlisted, then paste the link here.")

    # Get form inputs
    selected_category = render_category_selector(lecture_categories)
    lecture_title, custom_lecture_id, youtube_url, professor_note = render_lecture_details("", "")
    
    # Validate inputs
    yt_data = load_youtube_lectures()
    id_conflict, title_conflict, normalized_id = check_lecture_conflicts(
        lecture_title, custom_lecture_id, yt_data
    )
    
    # Submit controls
    can_submit = bool(
        lecture_title and youtube_url and selected_category 
        and not id_conflict and not title_conflict
    )
    trigger_submit = render_submit_controls(
        can_submit, id_conflict, title_conflict, normalized_id, lecture_title
    )

    if trigger_submit:
        on_submit_callback(
            lecture_title, youtube_url, selected_category, 
            custom_lecture_id, professor_note
        )
