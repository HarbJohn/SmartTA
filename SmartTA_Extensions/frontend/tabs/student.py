"""Student Assistant tab orchestrator.

Coordinates student-facing components from frontend/student/ package:
- Video player with timestamp navigation
- Semantic search (FAISS + BM25 + cross-encoder)
- AI-powered follow-up Q&A
- Course materials access
"""
import os
import json
from typing import Optional, Any

import streamlit as st

from SmartTA_Extensions.utils.helpers import load_youtube_lectures
from SmartTA_Extensions.frontend.components import render_course_materials
from SmartTA_Extensions.frontend.student.search_ui import (
    render_video_player,
    render_search_form,
    render_sidebar_stats,
    update_search_history,
)
from SmartTA_Extensions.frontend.student.search_pipeline import execute_search_pipeline, display_search_results
from SmartTA_Extensions.frontend.student.followup_ai import render_followup_question_section
from SmartTA_Extensions.backend.indexing import reindex_if_needed


def render_student_tab(
    DATA_DIR: str,
    META_SEGMENTS: str,
    META_PATH: str,
    INDEX_PATH: str,
    LOG_PATH: str,
    UPLOADS_DIR: str,
    all_lectures: list[str],
    lecture_categories: dict[str, list[str]],
    metadata: Optional[list[dict[str, Any]]] = None,
    index: Optional[Any] = None,
) -> None:
    """Render the Student Assistant tab.
    
    Args:
        DATA_DIR: Root data directory path
        META_SEGMENTS: Path to segments metadata JSON
        META_PATH: Path to search metadata JSON
        INDEX_PATH: Path to FAISS index file
        LOG_PATH: Path to query log JSON
        UPLOADS_DIR: Path to uploaded course materials
        all_lectures: List of all lecture titles
        lecture_categories: Map of category names to lecture IDs
        metadata: Optional preloaded metadata list
        index: Optional preloaded FAISS index
    """
    # Initialize session state
    if "last_search_results" not in st.session_state:
        st.session_state["last_search_results"] = []
    if "last_search_query" not in st.session_state:
        st.session_state["last_search_query"] = ""
    if "last_clicked_result" not in st.session_state:
        st.session_state["last_clicked_result"] = None

    # Handle timestamp jump from RAG tab
    if "jump_to_youtube" in st.session_state:
        jump_data = st.session_state["jump_to_youtube"]
        st.session_state["jump_to"] = {
            "lecture": jump_data.get("lecture"),
            "start": jump_data.get("timestamp", 0)
        }
        del st.session_state["jump_to_youtube"]
        st.info("⏩ Jumped to recommended segment! Playing below.")

    # Sidebar: lecture selection
    selected_lecture = render_sidebar_stats(all_lectures, "", lecture_categories)

    # Clear jump state on lecture change
    if "last_selected_lecture" not in st.session_state:
        st.session_state["last_selected_lecture"] = selected_lecture
    if st.session_state["last_selected_lecture"] != selected_lecture:
        if "jump_to" in st.session_state:
            del st.session_state["jump_to"]
        st.session_state["last_selected_lecture"] = selected_lecture

    # Main content: video and search
    col_video, col_ta = st.columns([2, 1])

    with col_video:
        render_video_player(selected_lecture, all_lectures)

    with col_ta:
        query, do_search, only_this_lecture = render_search_form(selected_lecture)

        # Execute search
        if do_search and query:
            _metadata = metadata
            _index = index
            if not _metadata or _index is None:
                if os.path.exists(META_SEGMENTS):
                    with open(META_SEGMENTS, "r", encoding="utf-8") as f:
                        _all_segments = json.load(f)
                else:
                    _all_segments = {}
                _metadata, _index, _ = reindex_if_needed(_all_segments, load_youtube_lectures)

            if _metadata and _index is not None:
                results = execute_search_pipeline(
                    query, only_this_lecture, selected_lecture, _metadata, _index, all_lectures
                )
                
                st.session_state["last_search_results"] = results
                st.session_state["last_search_query"] = query
                st.session_state["last_clicked_result"] = None
                update_search_history(query)
                
                display_search_results(results, query)

        # Show cached results
        if not do_search and st.session_state.get("last_search_results") and st.session_state.get("last_search_query"):
            st.markdown("---")
            from SmartTA_Extensions.frontend.components import render_search_results
            render_search_results(st.session_state["last_search_results"], st.session_state["last_search_query"])

        # AI follow-up questions
        render_followup_question_section(META_SEGMENTS)

    # Course materials
    st.markdown("---")
    st.markdown("## Course Materials")
    render_course_materials(UPLOADS_DIR)
