"""Professor Dashboard modules.

This package contains all professor-specific UI components:
- lecture_forms: Lecture addition form UI
- lecture_processor: Transcription and indexing
- lecture_editor: Lecture management (edit/delete)
- control_panel: System maintenance controls
- analytics_ui: Query analytics and confused topics
- material_mgmt: Course materials management
"""
from .lecture_forms import render_add_lecture_form
from .lecture_processor import process_new_lecture
from .lecture_editor import render_manage_lectures
from .control_panel import render_control_panel
from .analytics_ui import render_confused_topics, render_analytics_dashboard
from .material_mgmt import render_material_management

__all__ = [
    "render_add_lecture_form",
    "process_new_lecture",
    "render_manage_lectures",
    "render_control_panel",
    "render_confused_topics",
    "render_analytics_dashboard",
    "render_material_management",
]
