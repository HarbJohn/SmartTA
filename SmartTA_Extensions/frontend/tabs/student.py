import os
import json
import re
import time
import datetime
from typing import Optional, Any
import streamlit as st

from utils.helpers import (
    load_youtube_lectures,
    get_youtube_lectures_categorized,
)
from frontend.styles import apply_search_ui_styles, render_search_header
from frontend.components import render_search_results, render_course_materials
from backend.indexing import (
    load_sentence_transformer,
    load_cross_encoder,
    reindex_if_needed,
    get_index_hash,
)
from backend.search import (
    search_segments,
    deduplicate_segments,
    detect_question_type,
    expand_query_semantically,
    hybrid_rank,
    precision_rerank,
    token_overlap_score,
)
from utils.helpers import get_video_id_from_title
from backend.analytics import log_query


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

    def _sort_lectures(titles: list[str]) -> list[str]:
        def _lecture_key(title: str):
            match = re.match(r"^(\d+)", title.strip())
            if match:
                return (int(match.group(1)), title)
            return (9999, title)
        return sorted(titles, key=_lecture_key)

    if selected_category == "All Categories":
        available_lectures = _sort_lectures(all_lectures)
    else:
        yt_data = load_youtube_lectures()
        lecture_ids = lecture_categories.get(selected_category, [])
        available_lectures = [yt_data["lectures"][lec_id]["title"] for lec_id in lecture_ids]
        available_lectures = _sort_lectures(available_lectures)

    selected_lecture = st.sidebar.selectbox(
        "Choose lecture:",
        options=["All Lectures"] + available_lectures,
        index=0,
        help="Select a specific lecture to view",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Total Slide Lectures", 6)
    st.sidebar.metric("Total Video Lectures", len(all_lectures))
    if st.session_state.get("last_search_results"):
        st.sidebar.metric("Last Search Results", len(st.session_state["last_search_results"]))

    if "last_selected_lecture" not in st.session_state:
        st.session_state["last_selected_lecture"] = selected_lecture
    if st.session_state["last_selected_lecture"] != selected_lecture:
        if "jump_to" in st.session_state:
            del st.session_state["jump_to"]
        st.session_state["last_selected_lecture"] = selected_lecture

    col_video, col_ta = st.columns([2, 1])

    with col_video:
        st.subheader("Lecture Preview")
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
        apply_search_ui_styles()
        render_search_header()

        with st.form("ask_ta_form"):
            query = st.text_input(
                "Your question:",
                key="student_query",
                placeholder="e.g., What is gradient descent?",
                help="Press Enter or click Search",
                label_visibility="collapsed",
            )
            col_filter, col_btn = st.columns([2, 2])
            with col_filter:
                only_this_lecture = False
                if selected_lecture and selected_lecture != "All Lectures":
                    only_this_lecture = st.checkbox("📌 Current lecture only", value=False)
            with col_btn:
                do_search = st.form_submit_button("🔎 Search", use_container_width=True, type="primary")
            
            # Default to Quality mode (use BM25 hybrid ranking)
            search_mode = "🎯 Quality"

        if st.session_state.get("last_search_results"):
            if st.button("🗑️ Clear Results", use_container_width=True):
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

        # The search pipeline below mirrors app.py logic exactly
        if do_search and query:
            try:
                _metadata = metadata
                _index = index
            except Exception:
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
                        mode = search_mode.split()[1].lower() if "search_mode" in locals() else "fast"

                        if "query_cache" not in st.session_state:
                            st.session_state["query_cache"] = {}

                        cache_key = f"{query.lower().strip()}|{mode}|{idx_hash}"
                        cache_hit = False
                        if cache_key in st.session_state["query_cache"]:
                            cached_result = st.session_state["query_cache"][cache_key]
                            results = cached_result["results"]
                            progress_placeholder.progress(1.0)
                            status_placeholder.success(f"⚡ Loaded from cache: {len(results)} segment(s) (instant)")
                            time.sleep(0.5)
                            cache_hit = True

                        if not cache_hit:
                            question_type = detect_question_type(query)
                            expanded_query = expand_query_semantically(query, _metadata, embedder, top_k=3)
                            search_query = expanded_query if expanded_query != query else query

                            if mode == "fast":
                                progress_placeholder.progress(0.3)
                                status_placeholder.info(f"🔍 Searching {lecture_count_search} lectures…")
                                raw_results = search_segments(search_query, _metadata, _index, embedder, idx_hash, k=24, threshold=0.30)
                                for r in raw_results:
                                    r["kw_overlap"] = round(token_overlap_score(query, r.get("text", "")), 4)
                                progress_placeholder.progress(0.7)
                                status_placeholder.info("🎯 Filtering duplicates…")
                                results = deduplicate_segments(raw_results, embedder, max_final=8, text_sim_threshold=0.85, time_overlap=0.8)
                            elif mode == "quality":
                                progress_placeholder.progress(0.2)
                                status_placeholder.info(f"🔍 Searching {lecture_count_search} lectures…")
                                raw_results = search_segments(search_query, _metadata, _index, embedder, idx_hash, k=30, threshold=0.30)
                                progress_placeholder.progress(0.6)
                                status_placeholder.info("⚡ Hybrid reranking (BM25 + embeddings)…")
                                hybrid = hybrid_rank(search_query, raw_results, _metadata, embedder, use_bm25=True, max_candidates=36, question_type=question_type)
                                progress_placeholder.progress(0.85)
                                status_placeholder.info("🎯 Filtering duplicates…")
                                results = deduplicate_segments(hybrid, embedder, max_final=8, text_sim_threshold=0.85, time_overlap=0.8)
                            else:
                                progress_placeholder.progress(0.2)
                                status_placeholder.info(f"🔍 Searching {lecture_count_search} lectures…")
                                raw_results = search_segments(search_query, _metadata, _index, embedder, idx_hash, k=30, threshold=0.30)
                                progress_placeholder.progress(0.4)
                                status_placeholder.info("⚡ Hybrid reranking (BM25 + embeddings)…")
                                hybrid = hybrid_rank(search_query, raw_results, _metadata, embedder, use_bm25=True, max_candidates=36, question_type=question_type)
                                progress_placeholder.progress(0.65)
                                status_placeholder.info("🔬 Precision reranking (cross-encoder)…")
                                ce = load_cross_encoder()
                                if ce:
                                    hybrid = precision_rerank(search_query, hybrid, ce, top_k=24)
                                else:
                                    st.info("Precision model unavailable; using quality mode.")
                                progress_placeholder.progress(0.85)
                                status_placeholder.info("🎯 Filtering duplicates…")
                                results = deduplicate_segments(hybrid, embedder, max_final=8, text_sim_threshold=0.85, time_overlap=0.8)

                            if len(st.session_state["query_cache"]) >= 50:
                                oldest_key = next(iter(st.session_state["query_cache"]))
                                del st.session_state["query_cache"][oldest_key]
                            st.session_state["query_cache"][cache_key] = {"results": results}

                        if "only_this_lecture" in locals() and only_this_lecture and selected_lecture and selected_lecture != "All Lectures":
                            results = [r for r in results if r.get("lecture") == selected_lecture]

                        progress_placeholder.progress(1.0)
                        elapsed = time.time() - t0
                        status_placeholder.success(f"✅ Found {len(results)} segment(s) in {elapsed:.2f}s")
                        time.sleep(0.8)
                        st.session_state["last_search_results"] = results
                        st.session_state["last_search_query"] = query
                        st.session_state["last_clicked_result"] = None
                    finally:
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        skeleton_placeholder.empty()

                    if not results or len(results) < 2:
                        if not cache_hit and len(query.split()) <= 3:
                            try:
                                fallback_results = search_segments(query, _metadata, _index, embedder, idx_hash, k=40, threshold=0.20)
                                if len(fallback_results) > len(results):
                                    results = fallback_results[:8]
                                    st.success(f"Found {len(results)} results with broader search!")
                            except Exception:
                                pass

                    if not results:
                        st.warning(
                            "🤔 No relevant segments found. Try:\n- Using different keywords\n- Rephrasing your question\n- Being more specific\n- Using related terms or full names instead of acronyms"
                        )
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
                                <div style=\"font-size: 0.875rem; color: #64748b;\">Found {len(results)} relevant segment{'s' if len(results) != 1 else ''} in {elapsed:.2f}s</div>
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
            with st.expander("🧠 Generate concise answer (with citations)", expanded=False):
                if st.button("Compose Answer", key="compose_answer_btn", use_container_width=True):
                    q = st.session_state["last_search_query"]
                    rs = st.session_state["last_search_results"][:3]
                    try:
                        sentences = []
                        for r in rs:
                            for s in re.split(r"(?<=[.!?])\s+", r.get("text", "")):
                                if 6 <= len(s.split()) <= 36:
                                    sentences.append((s, r))
                        scored = sorted(((token_overlap_score(q, s), s, r) for s, r in sentences), key=lambda x: x[0], reverse=True)
                        picked = []
                        seen = set()
                        for _, s, r in scored:
                            if len(picked) >= 4:
                                break
                            sig = s.strip().lower()
                            if sig in seen:
                                continue
                            seen.add(sig)
                            picked.append((s, r))
                        if not picked and rs:
                            picked = [(rs[0].get("text", "")[:180] + "…", rs[0])]
                        answer = " ".join(s for s, _ in picked)
                        st.markdown(f"**Answer:** {answer}")
                        st.markdown("**Citations:**")
                        cites = []
                        for _, r in picked:
                            lec = r.get("lecture")
                            vid = get_video_id_from_title(lec)
                            ts = int(r.get("start", 0))
                            tstr = str(datetime.timedelta(seconds=ts))
                            if vid:
                                url = f"https://youtu.be/{vid}?t={ts}"
                                cites.append(f"- [{lec} @ {tstr}]({url})")
                            else:
                                cites.append(f"- {lec} @ {tstr}")
                        st.markdown("\n".join(cites))
                    except Exception:
                        st.info("Could not compose an answer. Try refining your question.")

    st.markdown("---")
    st.markdown("## Course Materials")
    render_course_materials(UPLOADS_DIR)
