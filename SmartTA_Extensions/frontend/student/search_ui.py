"""Student search and video player UI components.
Extracted from student.py to improve maintainability.
"""
import os
import datetime
from typing import Optional, Any
import streamlit as st

from SmartTA_Extensions.utils.helpers import (
    load_youtube_lectures,
    get_video_id_from_title,
)


def render_video_player(selected_lecture: str, all_lectures: list[str]) -> None:
    """Render the video player section with YouTube iframe."""
    st.markdown("### 🎥 Lecture Preview")
    st.warning(
        "Copyright Notice: These lecture videos are for educational viewing only. Downloading, distributing, or sharing is prohibited."
    )
    try:
        lecture_title = None
        if selected_lecture == "All Lectures" and all_lectures:
            lecture_title = all_lectures[0]
        elif selected_lecture != "All Lectures":
            lecture_title = selected_lecture

        if lecture_title:
            jump = st.session_state.get("jump_to")
            jump_start_param = ""
            if jump and isinstance(jump, dict):
                if jump.get("lecture"):
                    lecture_title = jump["lecture"]
                if isinstance(jump.get("start"), (int, float)):
                    try:
                        jump_start_param = f"?start={int(jump['start'])}&autoplay=1"
                    except Exception:
                        jump_start_param = ""
            
            # Display professor's note if available
            yt_data = load_youtube_lectures()
            professor_note = None
            for lec_id, lec_info in yt_data.get("lectures", {}).items():
                if lec_info.get("title") == lecture_title:
                    professor_note = lec_info.get("professor_note", "")
                    break
            if professor_note:
                st.info(f"Professor's Note: {professor_note}")

            video_id = get_video_id_from_title(lecture_title)
            if video_id:
                embed_url = f"https://www.youtube.com/embed/{video_id}{jump_start_param}"
                
                # Create stable container key to prevent video reload
                video_key = f"video_{video_id}_{jump_start_param}"
                
                # Only recreate iframe if URL actually changed
                if "current_video_key" not in st.session_state or st.session_state["current_video_key"] != video_key:
                    st.session_state["current_video_key"] = video_key
                
                # Use container to prevent unnecessary rerenders
                video_container = st.container()
                with video_container:
                    try:
                        import streamlit.components.v1 as components
                        components.iframe(embed_url, width=None, height=480, scrolling=False)
                    except Exception:
                        st.markdown(
                            f"""
                        <iframe width=\"100%\" height=\"480\" src=\"{embed_url}\" 
                        frameborder=\"0\" allow=\"accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture\" 
                        allowfullscreen></iframe>
                        """,
                            unsafe_allow_html=True,
                        )
                if jump_start_param:
                    if st.button("Back to selected lecture", key="clear_jump_btn"):
                        try:
                            del st.session_state["jump_to"]
                        except Exception:
                            st.session_state["jump_to"] = None
                        st.rerun()
            else:
                st.error(f"YouTube video not found for lecture: {lecture_title}")
        else:
            st.info("Please select a lecture to view.")
    except Exception:
        st.error("Error displaying lecture video. Please try again.")


def render_search_form(selected_lecture: str) -> tuple[str, bool, bool]:
    """Render the search form and return (query, do_search, only_this_lecture).
    
    Returns:
        tuple: (query text, whether to execute search, whether to filter by current lecture)
    """
    st.markdown("""
    <div style="text-align:center; margin:2rem 0 1.5rem 0;">
        <h2 style="
            font-size:2.2rem; 
            font-weight:900;
            margin:20px 0; 
            background:linear-gradient(90deg, #00C6FF, #2e7cf6);
            -webkit-background-clip:text; 
            -webkit-text-fill-color:transparent;
            text-shadow:0 0 20px rgba(0,198,255,0.3);
        ">💬 Ask the TA</h2>
        <p style="color:#9aa2b1; margin-top:-10px;">
            Ask questions about your course lectures and get instant answers with video timestamps
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize search history in session state
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []
    
    # Check if a history item was clicked
    history_search_triggered = False
    history_query = ""
    
    # Display search history suggestions before form
    if st.session_state["search_history"]:
        st.markdown('<div style="margin-bottom: 0.5rem;"><span style="color: #9aa2b1; font-size: 0.85rem;">📜 Recent searches:</span></div>', unsafe_allow_html=True)
        history_cols = st.columns(min(3, len(st.session_state["search_history"][:3])))
        for idx, hist_query in enumerate(st.session_state["search_history"][:3]):
            with history_cols[idx]:
                if st.button(f"🔍 {hist_query[:30]}{'...' if len(hist_query) > 30 else ''}", key=f"hist_{idx}", help=hist_query):
                    history_search_triggered = True
                    history_query = hist_query
    
    with st.form("ask_ta_form"):
        query = st.text_input(
            "Your question:",
            key="student_query",
            placeholder="e.g., What is gradient descent?",
            help="Press Enter or click Search",
            label_visibility="collapsed",
        )
        # Helpful search tips
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
                    padding: 0.6rem 1rem; border-radius: 10px; margin: 0.5rem 0 1rem 0;
                    border-left: 3px solid #8b5cf6; font-size: 0.85rem; color: #9aa2b1;">
            💡 <strong style="color: #a78bfa;">Tip:</strong> Use technical terms for best results 
            (e.g., <code style="background: rgba(139, 92, 246, 0.2); padding: 2px 6px; border-radius: 4px; color: #c4b5fd;">SVM</code>, 
            <code style="background: rgba(139, 92, 246, 0.2); padding: 2px 6px; border-radius: 4px; color: #c4b5fd;">gradient descent</code>, 
            <code style="background: rgba(139, 92, 246, 0.2); padding: 2px 6px; border-radius: 4px; color: #c4b5fd;">overfitting</code>, 
            <code style="background: rgba(139, 92, 246, 0.2); padding: 2px 6px; border-radius: 4px; color: #c4b5fd;">CNN</code>)
        </div>
        """, unsafe_allow_html=True)
        col_filter, col_btn = st.columns([3, 2])
        with col_filter:
            only_this_lecture = False
            if selected_lecture and selected_lecture != "All Lectures":
                only_this_lecture = st.checkbox("📌 Current lecture only", value=False)
        with col_btn:
            do_search = st.form_submit_button("🔎 Search", width='stretch', type="primary")

    if st.session_state.get("last_search_results"):
        if st.button("🗑️ Clear Results", width='stretch'):
            st.session_state["last_search_results"] = []
            st.session_state["last_search_query"] = ""
            st.session_state["last_clicked_result"] = None
            st.rerun()

    # Override query if history button was clicked
    if history_search_triggered:
        query = history_query
        do_search = True

    return query, do_search, only_this_lecture


def render_sidebar_stats(all_lectures: list[str], selected_category: str, lecture_categories: dict) -> str:
    """Render sidebar with category selection and stats.
    
    Returns:
        str: The selected lecture from the dropdown
    """
    if not all_lectures:
        st.warning("👋 Welcome! Please upload some lecture videos to get started.")

    st.sidebar.markdown("### 🎥 Lecture Selection")

    if len(lecture_categories) > 1:
        selected_cat = st.sidebar.selectbox(
            "📂 Category:",
            options=["All Categories"] + sorted(lecture_categories.keys()),
            index=0,
            help="Filter lectures by category",
        )
    else:
        selected_cat = "All Categories"

    if selected_cat == "All Categories":
        available_lectures = all_lectures
    else:
        from SmartTA_Extensions.utils.helpers import load_youtube_lectures
        yt_data = load_youtube_lectures()
        lecture_ids = lecture_categories.get(selected_cat, [])
        available_lectures = [yt_data["lectures"][lec_id]["title"] for lec_id in lecture_ids]
        available_lectures = sorted(available_lectures)

    selected_lecture = st.sidebar.selectbox(
        "Choose lecture:",
        options=["All Lectures"] + available_lectures,
        index=0,
        help="Select a specific lecture to view",
    )

    st.sidebar.markdown("""
    <div class="sidebar-title">
        <h3>📊 Quick Stats</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="stat-card-sidebar">
        <div style="color: #9aa2b1; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: 0.5px; text-transform: uppercase;">📚 Total Lectures</div>
        <div style="color: #00C6FF; font-size: 2rem; font-weight: 900;">{len(all_lectures)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get("last_search_results"):
        st.sidebar.markdown(f"""
        <div class="stat-card-sidebar">
            <div style="color: #9aa2b1; font-size: 0.8rem; font-weight: 600; margin-bottom: 0.5rem; letter-spacing: 0.5px; text-transform: uppercase;">🔍 Last Search</div>
            <div style="color: #6366f1; font-size: 2rem; font-weight: 900;">{len(st.session_state["last_search_results"])}</div>
            <div style="color: #7dd3fc; font-size: 0.75rem; margin-top: 0.25rem;">segments found</div>
        </div>
        """, unsafe_allow_html=True)

    return selected_lecture


def update_search_history(query: str) -> None:
    """Add query to search history (avoid duplicates, keep last 10)."""
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []
    
    if query not in st.session_state["search_history"]:
        st.session_state["search_history"].insert(0, query)
        # Keep only last 10 unique searches
        if len(st.session_state["search_history"]) > 10:
            st.session_state["search_history"] = st.session_state["search_history"][:10]
