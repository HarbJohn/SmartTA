"""YouTube-related tabs (student + professor) for SmartTA."""

from __future__ import annotations

import json

import streamlit as st

from . import config
from .helpers import ext_cwd


def render_youtube_library() -> None:
    """Student view for the YouTube semantic search tab."""
    if not config.ENABLE_YOUTUBE:
        return
    deps = (
        config.render_student_tab,
        config.load_youtube_lectures,
        config.get_youtube_lectures_categorized,
        config.reindex_if_needed,
    )
    if not all(deps):
        st.info("YouTube library unavailable (extension import failed).")
        return

    st.markdown("---")
    st.markdown("## 📚 Lecture Library & Search (YouTube)")
    st.caption("Browse and search through youtube videos")

    render_tab, load_lectures, categorize, reindex = deps  # type: ignore

    with ext_cwd():
        if config.EXT_META_SEG.exists():
            try:
                segments = json.loads(config.EXT_META_SEG.read_text(encoding="utf-8"))
                all_segments = segments if isinstance(segments, dict) else {}
            except Exception:
                all_segments = {}
        else:
            all_segments = {}

        # Reindex silently to avoid page jumps on load
        metadata, index, new_indexes = reindex(all_segments, load_lectures)  # type: ignore[arg-type]

        lecture_categories = categorize()
        yt = load_lectures()
        all_lectures = [info.get("title") for info in yt.get("lectures", {}).values()]

        render_tab(
            DATA_DIR=str(config.EXT_DATA_DIR),
            META_SEGMENTS=str(config.EXT_META_SEG),
            META_PATH=str(config.EXT_META_PATH),
            INDEX_PATH=str(config.EXT_INDEX_PATH),
            LOG_PATH=str(config.EXT_LOG_PATH),
            UPLOADS_DIR=str(config.EXT_UPLOADS),
            all_lectures=all_lectures,
            lecture_categories=lecture_categories,
            metadata=metadata,
            index=index,
        )


def render_professor_dashboard() -> None:
    """Professor dashboard wiring from the SmartTA extensions package."""
    if not config.ENABLE_YOUTUBE:
        return
    if not (config.render_professor_tab and config.get_youtube_lectures_categorized):
        st.info("Professor dashboard not available (extension import failed).")
        return

    with ext_cwd():
        lecture_categories = config.get_youtube_lectures_categorized()
        config.render_professor_tab(  # type: ignore[arg-type]
            DATA_DIR=str(config.EXT_DATA_DIR),
            LECTURES_DIR=str(config.EXT_LECTURES),
            TRANSCRIPT_DIR=str(config.EXT_TRANS),
            META_SEGMENTS=str(config.EXT_META_SEG),
            META_PATH=str(config.EXT_META_PATH),
            INDEX_PATH=str(config.EXT_INDEX_PATH),
            LOG_PATH=str(config.EXT_LOG_PATH),
            UPLOADS_DIR=str(config.EXT_UPLOADS),
            lecture_categories=lecture_categories,
        )
