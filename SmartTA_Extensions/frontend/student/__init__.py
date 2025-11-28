"""Student Assistant modules.

This package contains all student-specific UI components:
- search_ui: Video player and search form UI
- search_pipeline: Search execution and ranking
- followup_ai: AI-powered follow-up Q&A
"""
from .search_ui import (
    render_video_player,
    render_search_form,
    render_sidebar_stats,
    update_search_history,
)
from .search_pipeline import execute_search_pipeline, display_search_results
from .followup_ai import render_followup_question_section

__all__ = [
    "render_video_player",
    "render_search_form",
    "render_sidebar_stats",
    "update_search_history",
    "execute_search_pipeline",
    "display_search_results",
    "render_followup_question_section",
]
