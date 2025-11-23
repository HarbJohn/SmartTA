import os
import json
import re
import time
import datetime
from typing import Optional, Any
import streamlit as st

from SmartTA_Extensions.utils.helpers import (
    load_youtube_lectures,
    get_youtube_lectures_categorized,
)
from SmartTA_Extensions.frontend.components import render_search_results, render_course_materials
from SmartTA_Extensions.backend.indexing import (
    load_sentence_transformer,
    load_cross_encoder,
    reindex_if_needed,
    get_index_hash,
)
from SmartTA_Extensions.backend.search import (
    search_segments,
    deduplicate_segments,
    detect_question_type,
    expand_query_semantically,
    hybrid_rank,
)
from SmartTA_Extensions.utils.helpers import get_video_id_from_title
from SmartTA_Extensions.backend.analytics import log_query


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
    # Sidebar selections and quick stats
    if not all_lectures:
        st.warning("👋 Welcome! Please upload some lecture videos to get started.")

    st.sidebar.markdown("### 🎥 Lecture Selection")

    if len(lecture_categories) > 1:
        selected_category = st.sidebar.selectbox(
            "📂 Category:",
            options=["All Categories"] + sorted(lecture_categories.keys()),
            index=0,
            help="Filter lectures by category",
        )
    else:
        selected_category = "All Categories"

    if selected_category == "All Categories":
        available_lectures = all_lectures
    else:
        yt_data = load_youtube_lectures()
        lecture_ids = lecture_categories.get(selected_category, [])
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

    if "last_selected_lecture" not in st.session_state:
        st.session_state["last_selected_lecture"] = selected_lecture
    if st.session_state["last_selected_lecture"] != selected_lecture:
        if "jump_to" in st.session_state:
            del st.session_state["jump_to"]
        st.session_state["last_selected_lecture"] = selected_lecture

    col_video, col_ta = st.columns([2, 1])

    with col_video:
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

    with col_ta:
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

        if "last_search_results" not in st.session_state:
            st.session_state["last_search_results"] = []
        if "last_search_query" not in st.session_state:
            st.session_state["last_search_query"] = ""
        if "last_clicked_result" not in st.session_state:
            st.session_state["last_clicked_result"] = None

        # Override query if history button was clicked
        if history_search_triggered:
            query = history_query
            do_search = True

        # The search pipeline below mirrors app.py logic exactly
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

            if not _metadata or _index is None:
                st.warning("Search index not available yet. Add a lecture or re-run indexing from the Professor Dashboard.")
            else:
                embedder = load_sentence_transformer()
                if not embedder:
                    st.error("Embedding model failed to load. Please try again.")
                else:
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    skeleton_placeholder = st.empty()
                    skeleton_placeholder.markdown(
                        """
                    <div style="margin: 2rem 0;">
                        <div class="skeleton" style="height: 60px; margin-bottom: 1rem;"></div>
                        <div class="skeleton" style="height: 120px; margin-bottom: 0.75rem;"></div>
                        <div class="skeleton" style="height: 120px; margin-bottom: 0.75rem;"></div>
                        <div class="skeleton" style="height: 120px;"></div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    try:
                        idx_hash = get_index_hash()
                        t0 = time.time()
                        lecture_count_search = len(all_lectures) if all_lectures else len(set(m.get("lecture", "") for m in _metadata))
                        # Use balanced "quality" mode (FAISS + BM25 hybrid)
                        mode = "quality"

                        if "query_cache" not in st.session_state:
                            st.session_state["query_cache"] = {}

                        cache_key = f"{query.lower().strip()}|{mode}|{idx_hash}"
                        cache_hit = False
                        typo_corrections = []  # Initialize for cache hits
                        
                        if cache_key in st.session_state["query_cache"]:
                            cached_result = st.session_state["query_cache"][cache_key]
                            results = cached_result["results"]
                            typo_corrections = cached_result.get("typo_corrections", [])
                            progress_placeholder.progress(1.0)
                            status_placeholder.success(f"⚡ Loaded from cache: {len(results)} segment(s) (instant)")
                            time.sleep(0.5)
                            cache_hit = True

                        if not cache_hit:
                            question_type = detect_question_type(query)
                            expanded_query = expand_query_semantically(query, _metadata, embedder, top_k=3)
                            search_query = expanded_query if expanded_query != query else query

                            # Balanced search: FAISS + BM25 hybrid ranking
                            progress_placeholder.progress(0.2)
                            status_placeholder.info(f"🔍 Searching {lecture_count_search} lectures…")
                            raw_results, typo_corrections = search_segments(search_query, _metadata, _index, embedder, idx_hash, k=30, threshold=0.30)
                            progress_placeholder.progress(0.6)
                            status_placeholder.info("⚡ Hybrid reranking (BM25 + embeddings)…")
                            hybrid = hybrid_rank(search_query, raw_results, _metadata, embedder, use_bm25=True, max_candidates=36, question_type=question_type)
                            progress_placeholder.progress(0.85)
                            status_placeholder.info("🎯 Filtering duplicates…")
                            results = deduplicate_segments(hybrid, embedder, max_final=8, text_sim_threshold=0.85, time_overlap=0.8)

                            if len(st.session_state["query_cache"]) >= 50:
                                oldest_key = next(iter(st.session_state["query_cache"]))
                                del st.session_state["query_cache"][oldest_key]
                            st.session_state["query_cache"][cache_key] = {
                                "results": results,
                                "typo_corrections": typo_corrections
                            }

                        if "only_this_lecture" in locals() and only_this_lecture and selected_lecture and selected_lecture != "All Lectures":
                            results = [r for r in results if r.get("lecture") == selected_lecture]

                        progress_placeholder.progress(1.0)
                        elapsed = time.time() - t0
                        
                        # Calculate total watch time
                        total_watch_seconds = sum(r.get('end', r.get('start', 0) + 15) - r.get('start', 0) for r in results)
                        watch_mins = total_watch_seconds // 60
                        watch_secs = total_watch_seconds % 60
                        watch_time_str = f"{watch_mins}m {watch_secs}s" if watch_mins > 0 else f"{watch_secs}s"
                        
                        # Display typo corrections if any
                        if 'typo_corrections' in locals() and typo_corrections:
                            correction_text = ", ".join([f"<strong>{orig}</strong> → {corr}" for orig, corr in typo_corrections])
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
                                        padding: 0.75rem 1.25rem; border-radius: 10px; border-left: 3px solid #6366f1;
                                        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.2); margin-bottom: 1rem;">
                                <div style="display: flex; align-items: center; gap: 0.5rem;">
                                    <span style="font-size: 1.2rem;">✏️</span>
                                    <span style="color: #a5b4fc; font-size: 0.9rem;">Autocorrected: {correction_text}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        status_placeholder.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
                                    padding: 1rem 1.5rem; border-radius: 12px; border-left: 4px solid #10b981;
                                    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.2); animation: slideIn 0.4s ease-out;">
                            <div style="display: flex; align-items: center; gap: 0.75rem;">
                                <span style="font-size: 1.5rem;">✨</span>
                                <div>
                                    <strong style="color: #10b981; font-size: 1.1rem;">Search Complete!</strong>
                                    <div style="color: #9aa2b1; font-size: 0.9rem; margin-top: 0.25rem;">
                                        Found <span style="color: #10b981; font-weight: 700;">{len(results)}</span> segment{'s' if len(results) != 1 else ''} 
                                        in <span style="color: #10b981; font-weight: 700;">{elapsed:.2f}s</span>
                                        · ⏱ <span style="color: #10b981; font-weight: 700;">{watch_time_str}</span> total watch time
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(0.8)
                        st.session_state["last_search_results"] = results
                        st.session_state["last_search_query"] = query
                        st.session_state["last_clicked_result"] = None
                        
                        # Add to search history (avoid duplicates, keep last 10)
                        if query not in st.session_state["search_history"]:
                            st.session_state["search_history"].insert(0, query)
                            # Keep only last 10 unique searches
                            if len(st.session_state["search_history"]) > 10:
                                st.session_state["search_history"] = st.session_state["search_history"][:10]
                    finally:
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        skeleton_placeholder.empty()

                    if not results or len(results) < 2:
                        if not cache_hit and len(query.split()) <= 3:
                            try:
                                fallback_results, _ = search_segments(query, _metadata, _index, embedder, idx_hash, k=40, threshold=0.20)
                                if len(fallback_results) > len(results):
                                    results = fallback_results[:8]
                                    st.markdown(f"""
                                    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
                                                padding: 1rem 1.5rem; border-radius: 12px; border-left: 4px solid #6366f1;
                                                box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);">
                                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                                            <span style="font-size: 1.3rem;">🔍</span>
                                            <span style="color: #a5b4fc; font-weight: 600;">Found {len(results)} results with broader search!</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            except Exception:
                                pass

                    if not results:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.1) 100%);
                                    padding: 1.5rem 2rem; border-radius: 16px; border-left: 5px solid #f59e0b;
                                    box-shadow: 0 4px 20px rgba(245, 158, 11, 0.2);">
                            <div style="display: flex; gap: 1rem;">
                                <div style="font-size: 2rem;">🤔</div>
                                <div>
                                    <h4 style="color: #fbbf24; margin: 0 0 0.75rem 0; font-size: 1.15rem; font-weight: 700;">No Relevant Segments Found</h4>
                                    <div style="color: #9aa2b1; line-height: 1.8;">
                                        <div style="margin-bottom: 0.5rem;"><strong style="color: #fbbf24;">•</strong> Try using different keywords</div>
                                        <div style="margin-bottom: 0.5rem;"><strong style="color: #fbbf24;">•</strong> Rephrase your question</div>
                                        <div style="margin-bottom: 0.5rem;"><strong style="color: #fbbf24;">•</strong> Be more specific about the topic</div>
                                        <div><strong style="color: #fbbf24;">•</strong> Use full terms instead of acronyms</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        if len(results) >= 3:
                            st.markdown(
                                """
                            <script>
                            const colors = ['#667eea', '#764ba2', '#10b981', '#f59e0b', '#3b82f6', '#ec4899'];
                            for (let i = 0; i < 30; i++) {
                                setTimeout(() => {
                                    const confetti = document.createElement('div');
                                    confetti.className = 'confetti';
                                    confetti.style.left = Math.random() * 100 + '%';
                                    confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
                                    confetti.style.animationDelay = (Math.random() * 0.5) + 's';
                                    confetti.style.width = (Math.random() * 8 + 4) + 'px';
                                    confetti.style.height = confetti.style.width;
                                    document.body.appendChild(confetti);
                                    setTimeout(() => confetti.remove(), 3000);
                                }, i * 50);
                            }
                            </script>
                            """,
                                unsafe_allow_html=True,
                            )

                        toast_id = f"toast-{int(time.time()*1000)}"
                        st.markdown(
                            f"""
                        <style>
                        @keyframes toast-auto-dismiss {{
                            0% {{
                                transform: translateX(0);
                                opacity: 1;
                            }}
                            85% {{
                                transform: translateX(0);
                                opacity: 1;
                            }}
                            100% {{
                                transform: translateX(400px);
                                opacity: 0;
                                visibility: hidden;
                            }}
                        }}
                        #{toast_id} {{
                            animation: toast-auto-dismiss 3s ease forwards;
                        }}
                        </style>
                        <div class=\"toast success\" id=\"{toast_id}\">
                            <span style=\"font-size: 1.5rem;\">✅</span>
                            <div>
                                <strong>Search Complete!</strong>
                                <div style="font-size: 0.875rem; color: #9aa2b1;">Found {len(results)} relevant segment{'s' if len(results) != 1 else ''} in {elapsed:.2f}s</div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        render_search_results(results, query)

                    log_query(query, None, results)

        if not do_search and st.session_state.get("last_search_results") and st.session_state.get("last_search_query"):
            st.markdown("---")
            render_search_results(st.session_state["last_search_results"], st.session_state["last_search_query"])

        if st.session_state.get("last_search_results") and st.session_state.get("last_search_query"):
            st.markdown("")
            with st.expander("💬 Ask Follow-up Question (AI-powered)", expanded=False):
                st.caption("Ask a clarifying question based on the search results above. GPT will synthesize an answer from the retrieved segments.")
                
                followup_query = st.text_input(
                    "Follow-up question:",
                    placeholder="e.g., Why is the learning rate important?",
                    key="followup_query_input",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    ask_followup = st.button("🤖 Ask AI", key="ask_followup_btn", width='stretch', type="primary")
                with col2:
                    use_top_n = st.selectbox("Use top", [3, 5, 8], index=0, key="followup_top_n")
                
                if ask_followup and followup_query:
                    with st.spinner("🧠 AI is thinking..."):
                        try:
                            import openai
                            api_key = os.getenv("OPENAI_API_KEY")
                            
                            if not api_key:
                                st.error("⚠️ OPENAI_API_KEY not set. Please configure your API key.")
                            else:
                                # Load all segments for context expansion
                                all_segments_data = {}
                                if os.path.exists(META_SEGMENTS):
                                    with open(META_SEGMENTS, "r", encoding="utf-8") as f:
                                        all_segments_data = json.load(f)
                                
                                # Build context from search results with neighboring segments
                                rs = st.session_state["last_search_results"][:use_top_n]
                                context_parts = []
                                for idx, r in enumerate(rs, 1):
                                    lec = r.get("lecture", "Unknown")
                                    ts_start = r.get("start", 0)
                                    ts_end = r.get("end", 0)
                                    text = r.get("text", "")
                                    
                                    # Get neighboring segments for fuller context (±1 segment before/after)
                                    lecture_segments = all_segments_data.get(lec, [])
                                    expanded_text = text
                                    
                                    if lecture_segments:
                                        # Find current segment index
                                        current_idx = None
                                        for seg_idx, seg in enumerate(lecture_segments):
                                            if abs(seg.get("start", 0) - ts_start) < 1:  # Match by timestamp
                                                current_idx = seg_idx
                                                break
                                        
                                        if current_idx is not None:
                                            # Add context: 1 segment before + current + 1 segment after
                                            context_segments = []
                                            if current_idx > 0:
                                                context_segments.append(lecture_segments[current_idx - 1].get("text", ""))
                                            context_segments.append(text)
                                            if current_idx < len(lecture_segments) - 1:
                                                context_segments.append(lecture_segments[current_idx + 1].get("text", ""))
                                            
                                            expanded_text = " ".join(context_segments)
                                    
                                    context_parts.append(f"[{idx}] From '{lec}' at {datetime.timedelta(seconds=int(ts_start))}:\n{expanded_text}")
                                
                                context = "\n\n".join(context_parts)
                                original_q = st.session_state["last_search_query"]
                                
                                # Call OpenAI
                                client = openai.OpenAI(api_key=api_key)
                                response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are a helpful teaching assistant. Answer the student's follow-up question using this priority:\n1. PRIMARY: Use information from the provided lecture segments and cite them with [1], [2], etc.\n2. FALLBACK: If the segments don't contain enough information, provide a correct general explanation but clearly mark it as 'Note: This information is not from your lecture materials - it's general knowledge about [topic].'\nBe concise, accurate, and educational. Always distinguish between lecture content and general knowledge."
                                        },
                                        {
                                            "role": "user",
                                            "content": f"Original question: {original_q}\n\nLecture segments:\n{context}\n\nFollow-up question: {followup_query}\n\nAnswer using lecture segments if possible (cite [1], [2], etc.). If not, provide accurate general knowledge but label it clearly."
                                        }
                                    ],
                                    temperature=0.5,
                                    max_tokens=600
                                )
                                
                                answer = response.choices[0].message.content
                                
                                st.markdown("### 🎓 AI Answer")
                                st.markdown(answer)
                                
                                st.markdown("---")
                                st.markdown("**📚 Source Segments:**")
                                for idx, r in enumerate(rs, 1):
                                    lec = r.get("lecture")
                                    vid = get_video_id_from_title(lec)
                                    ts = int(r.get("start", 0))
                                    tstr = str(datetime.timedelta(seconds=ts))
                                    if vid:
                                        url = f"https://youtu.be/{vid}?t={ts}"
                                        st.markdown(f"**[{idx}]** [{lec} @ {tstr}]({url})")
                                    else:
                                        st.markdown(f"**[{idx}]** {lec} @ {tstr}")
                                
                        except ImportError:
                            st.error("❌ OpenAI package not installed. Run: pip install openai")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            st.info("Make sure your OPENAI_API_KEY is valid and you have API credits.")

    st.markdown("---")
    st.markdown("## Course Materials")
    render_course_materials(UPLOADS_DIR)
