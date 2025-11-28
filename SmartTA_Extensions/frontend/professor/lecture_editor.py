"""Lecture editing and management UI components.

This module handles the user interface for managing existing lectures,
including editing metadata, deleting lectures, and organizing by category.
"""
import os
import json
import time
from typing import Dict, Any

import streamlit as st

from SmartTA_Extensions.utils.helpers import (
    load_youtube_lectures,
    save_youtube_lectures,
    delete_lecture_file,
)

# LECTURE LIST UI

def render_manage_lectures(TRANSCRIPT_DIR: str, META_SEGMENTS: str) -> None:
    """Render the 'Manage Existing Lectures' section."""
    st.markdown("---")
    st.markdown('<div id="manage_lectures"></div>', unsafe_allow_html=True)
    st.subheader("🗂️ Manage Existing Lectures")

    yt_data = load_youtube_lectures()
    all_lecture_data = yt_data.get("lectures", {})
    if not all_lecture_data:
        st.info("No lectures added yet. Add your first lecture above!")
        return

    # Group by category
    categories = {}
    for lec_id, lec_info in all_lecture_data.items():
        cat = lec_info.get("category", "Uncategorized")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((lec_id, lec_info))
    
    for category in sorted(categories.keys()):
        with st.expander(f"📂 {category} ({len(categories[category])} lectures)"):
            for lec_id, lec_info in sorted(categories[category], key=lambda x: x[1].get("title", "")):
                render_single_lecture_item(lec_id, lec_info, category, categories, TRANSCRIPT_DIR, META_SEGMENTS)


def render_single_lecture_item(
    lec_id: str,
    lec_info: Dict[str, Any],
    category: str,
    categories: Dict[str, list],
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str,
) -> None:
    """Render a single lecture item with edit/delete controls."""
    title = lec_info.get("title", "Unknown")
    note = lec_info.get("professor_note", "")
    youtube_url = lec_info.get("youtube_url", "")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{title}**")
        if note:
            st.caption(f"📝 Note: {note}")
        st.caption(f"🔗 [{youtube_url}]({youtube_url})")
    with col2:
        if st.button("Edit", key=f"edit_{lec_id}"):
            st.session_state[f"editing_{lec_id}"] = True
        if st.button("Delete", key=f"delete_{lec_id}"):
            st.session_state[f"confirm_delete_{lec_id}"] = True
    
    # Edit form
    if st.session_state.get(f"editing_{lec_id}", False):
        render_edit_lecture_form(lec_id, title, note, category, categories, TRANSCRIPT_DIR, META_SEGMENTS)
    
    # Delete confirmation
    if st.session_state.get(f"confirm_delete_{lec_id}", False):
        render_delete_confirmation(lec_id, title, TRANSCRIPT_DIR, META_SEGMENTS)
    
    st.markdown("---")

# EDIT OPERATIONS

def render_edit_lecture_form(
    lec_id: str,
    title: str,
    note: str,
    category: str,
    categories: Dict[str, list],
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str,
) -> None:
    """Render edit form for a lecture."""
    with st.form(key=f"edit_form_{lec_id}"):
        st.markdown("#### Edit Lecture")
        new_title = st.text_input("Title:", value=title)
        new_note = st.text_area("Professor's Note:", value=note)
        new_category = st.selectbox(
            "Category:", options=sorted(categories.keys()) + ["+ New Category"],
            index=sorted(categories.keys()).index(category) if category in categories else 0,
        )
        if new_category == "+ New Category":
            new_category = st.text_input("New category name:")
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.form_submit_button("Save Changes", type="primary"):
                update_lecture_metadata(
                    lec_id, new_title, new_note, new_category,
                    title, TRANSCRIPT_DIR, META_SEGMENTS
                )
        with col_cancel:
            if st.form_submit_button("Cancel"):
                st.session_state[f"editing_{lec_id}"] = False
                st.rerun()


def update_lecture_metadata(
    lec_id: str,
    new_title: str,
    new_note: str,
    new_category: str,
    old_title: str,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str,
) -> None:
    """Update lecture metadata and related files."""
    yt_data = load_youtube_lectures()
    yt_data["lectures"][lec_id]["title"] = new_title
    yt_data["lectures"][lec_id]["professor_note"] = new_note
    yt_data["lectures"][lec_id]["category"] = new_category
    
    # Update title in segments if changed
    if new_title != old_title:
        if os.path.exists(META_SEGMENTS):
            with open(META_SEGMENTS, 'r', encoding='utf-8') as f:
                all_segments = json.load(f)
            if old_title in all_segments:
                all_segments[new_title] = all_segments.pop(old_title)
                with open(META_SEGMENTS, 'w', encoding='utf-8') as f:
                    json.dump(all_segments, f, indent=2)
        old_transcript = os.path.join(TRANSCRIPT_DIR, f"{old_title}.txt")
        new_transcript = os.path.join(TRANSCRIPT_DIR, f"{new_title}.txt")
        if os.path.exists(old_transcript):
            os.rename(old_transcript, new_transcript)
    
    save_youtube_lectures(yt_data)
    st.session_state[f"editing_{lec_id}"] = False
    st.success("Changes saved!")
    time.sleep(1)
    st.rerun()

# DELETE OPERATIONS

def render_delete_confirmation(
    lec_id: str,
    title: str,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str
) -> None:
    """Render delete confirmation dialog."""
    st.warning(f"Are you sure you want to delete **{title}**?")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Yes, Delete", key=f"confirm_yes_{lec_id}"):
            delete_lecture(lec_id, title, TRANSCRIPT_DIR, META_SEGMENTS)
    with col_no:
        if st.button("Cancel", key=f"confirm_no_{lec_id}"):
            st.session_state[f"confirm_delete_{lec_id}"] = False
            st.rerun()


def delete_lecture(
    lec_id: str,
    title: str,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str
) -> None:
    """Delete lecture and all associated files."""
    delete_lecture_file(lec_id)
    if os.path.exists(META_SEGMENTS):
        with open(META_SEGMENTS, 'r', encoding='utf-8') as f:
            all_segments = json.load(f)
        if title in all_segments:
            del all_segments[title]
            with open(META_SEGMENTS, 'w', encoding='utf-8') as f:
                json.dump(all_segments, f, indent=2)
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{title}.txt")
    if os.path.exists(transcript_path):
        os.remove(transcript_path)
    st.session_state[f"confirm_delete_{lec_id}"] = False
    st.success(f"Deleted '{title}'")
    time.sleep(1)
    st.rerun()
