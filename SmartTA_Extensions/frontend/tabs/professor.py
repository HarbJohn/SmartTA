"""Professor Dashboard tab orchestrator.

Coordinates professor-facing components from frontend/professor/ package:
- Lecture CRUD (add, edit, delete lectures with auto-transcription)
- Analytics (query logs, confused topics, engagement metrics)
- Course materials management
- System maintenance controls
"""
import logging
import streamlit as st

from SmartTA_Extensions.frontend.professor.control_panel import render_control_panel
from SmartTA_Extensions.frontend.professor.analytics_ui import render_confused_topics, render_analytics_dashboard
from SmartTA_Extensions.frontend.professor import render_add_lecture_form, render_manage_lectures
from SmartTA_Extensions.frontend.professor.material_mgmt import render_material_management

logger = logging.getLogger(__name__)

try:
    import sys
    from pathlib import Path
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    _smartta_rag = _project_root / "SmartTA_RAG"
    if str(_smartta_rag) not in sys.path:
        sys.path.insert(0, str(_smartta_rag))
    
    from smartta_unified import config
    config.APP_DATA_DIR = _project_root / "data"
    
    from smartta_unified.feedback import render_feedback_summary
    FEEDBACK_AVAILABLE = True
except Exception as e:
    FEEDBACK_AVAILABLE = False
    logger.warning(f"Feedback module unavailable: {e}")


def render_professor_tab(
    DATA_DIR: str,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str,
    META_PATH: str,
    INDEX_PATH: str,
    LOG_PATH: str,
    UPLOADS_DIR: str,
    lecture_categories: dict[str, list[str]],
) -> None:
    """Render the Professor Dashboard tab.
    
    Args:
        DATA_DIR: Root data directory
        TRANSCRIPT_DIR: Directory for transcript text files
        META_SEGMENTS: Path to segments metadata JSON
        META_PATH: Path to FAISS metadata JSON
        INDEX_PATH: Path to FAISS vector index
        LOG_PATH: Path to query analytics log
        UPLOADS_DIR: Directory for uploaded course materials
        lecture_categories: Category to lecture IDs mapping
    """

    # Dashboard header
    st.markdown("""
    <h2 style="
        text-align:center; 
        font-size:2.2rem; 
        font-weight:900;
        margin:20px 0 30px 0; 
        background:linear-gradient(90deg, #00C6FF, #2e7cf6);
        -webkit-background-clip:text; 
        -webkit-text-fill-color:transparent;
        text-shadow:0 0 20px rgba(0,198,255,0.3);
    ">📺 Professor Dashboard</h2>
    """, unsafe_allow_html=True)

    # Course feedback from RAG module
    if FEEDBACK_AVAILABLE:
        st.markdown("---")
        render_feedback_summary(title="### 📝 Course Feedback Summary")
        st.markdown("---")

    # Dashboard sections
    render_control_panel(LOG_PATH, META_PATH, INDEX_PATH, TRANSCRIPT_DIR)
    render_confused_topics(LOG_PATH)
    
    # Lecture management
    def submit_callback(title, url, category, custom_id, note):
        from SmartTA_Extensions.frontend.professor.lecture_processor import process_new_lecture
        process_new_lecture(
            title, url, category, custom_id, note,
            DATA_DIR, TRANSCRIPT_DIR, META_SEGMENTS, META_PATH, INDEX_PATH
        )
    render_add_lecture_form(lecture_categories, submit_callback)
    
    render_material_management(UPLOADS_DIR)
    render_manage_lectures(TRANSCRIPT_DIR, META_SEGMENTS)
    render_analytics_dashboard(LOG_PATH)
