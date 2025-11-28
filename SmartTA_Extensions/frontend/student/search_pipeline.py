"""Search pipeline execution logic for student assistant.
Handles the complete search workflow including caching, query expansion, ranking, and result display.
Extracted from student.py to improve maintainability.
"""
import os
import json
import time
import datetime
from typing import Optional, Any, Tuple, List, Dict

import streamlit as st

from SmartTA_Extensions.backend.indexing import (
    load_sentence_transformer,
    load_cross_encoder,
    get_index_hash,
)
from SmartTA_Extensions.backend.search import (
    search_segments,
    deduplicate_segments,
    detect_question_type,
    expand_query_semantically,
    hybrid_rank,
)
from SmartTA_Extensions.backend.search_modules.ranking import precision_rerank
from SmartTA_Extensions.backend.analytics import log_query
from SmartTA_Extensions.frontend.components import render_search_results


def execute_search_pipeline(
    query: str,
    only_this_lecture: bool,
    selected_lecture: str,
    metadata: List[Dict[str, Any]],
    index: Any,
    all_lectures: list[str],
) -> List[Dict[str, Any]]:
    """Execute the complete search pipeline and return results.
    
    Args:
        query: The search query string
        only_this_lecture: Whether to filter by current lecture only
        selected_lecture: The currently selected lecture title
        metadata: Preloaded metadata list
        index: Preloaded FAISS index
        all_lectures: List of all lecture titles
        
    Returns:
        List of search result dictionaries
    """
    if not metadata or index is None:
        st.warning("Search index not available yet. Add a lecture or re-run indexing from the Professor Dashboard.")
        return []
    
    embedder = load_sentence_transformer()
    if not embedder:
        st.error("Embedding model failed to load. Please try again.")
        return []
    
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
        lecture_count_search = len(all_lectures) if all_lectures else len(set(m.get("lecture", "") for m in metadata))
        mode = "quality"  # Use balanced "quality" mode (FAISS + BM25 hybrid)

        if "query_cache" not in st.session_state:
            st.session_state["query_cache"] = {}

        cache_key = f"{query.lower().strip()}|{mode}|{idx_hash}"
        cache_hit = False
        typo_corrections = []
        
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
            expanded_query = expand_query_semantically(query, metadata, embedder, top_k=3)
            search_query = expanded_query if expanded_query != query else query

            # Balanced search: FAISS + BM25 hybrid ranking
            progress_placeholder.progress(0.2)
            status_placeholder.info(f"🔍 Searching {lecture_count_search} lectures…")
            raw_results, typo_corrections = search_segments(search_query, metadata, index, embedder, idx_hash, k=30, threshold=0.30)
            progress_placeholder.progress(0.6)
            status_placeholder.info("⚡ Hybrid reranking (BM25 + embeddings)…")
            hybrid = hybrid_rank(search_query, raw_results, metadata, embedder, use_bm25=True, max_candidates=36, question_type=question_type)
            
            # Optional cross-encoder precision rerank
            try:
                cross_encoder = load_cross_encoder()
            except Exception:
                cross_encoder = None
            if cross_encoder:
                progress_placeholder.progress(0.75)
                status_placeholder.info("🎯 Precision reranking…")
                hybrid = precision_rerank(search_query, hybrid, cross_encoder, top_k=24)
            else:
                # Show a one-time fading notice about missing cross-encoder (performance fallback)
                if not st.session_state.get("ce_warning_shown", False):
                    st.session_state["ce_warning_shown"] = True
                    ce_warn = st.empty()
                    ce_warn.markdown(
                        """
                        <style>
                        @keyframes fadeOutCE { 0% {opacity:1;} 70% {opacity:1;} 100% {opacity:0; height:0; margin:0; padding:0;} }
                        .ce-missing-banner { 
                            background: linear-gradient(90deg, rgba(255,179,71,0.18), rgba(255,140,0,0.18));
                            border-left: 4px solid #f59e0b;
                            padding: 0.6rem 0.9rem; 
                            border-radius: 10px; 
                            font-size: 0.75rem; 
                            color: #b45309; 
                            animation: fadeOutCE 4.5s ease forwards; 
                            box-shadow: 0 2px 8px rgba(245,158,11,0.25);
                        }
                        </style>
                        <div class="ce-missing-banner">⚠️ using fast ranking.</div>
                        """,
                        unsafe_allow_html=True,
                    )
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

        if only_this_lecture and selected_lecture and selected_lecture != "All Lectures":
            results = [r for r in results if r.get("lecture") == selected_lecture]

        progress_placeholder.progress(1.0)
        elapsed = time.time() - t0
        
        # Calculate total watch time
        total_watch_seconds = sum(r.get('end', r.get('start', 0) + 15) - r.get('start', 0) for r in results)
        watch_mins = total_watch_seconds // 60
        watch_secs = total_watch_seconds % 60
        watch_time_str = f"{watch_mins}m {watch_secs}s" if watch_mins > 0 else f"{watch_secs}s"
        
        # Display typo corrections if any
        if typo_corrections:
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
        
        # Fallback search if results are too few
        if not results or len(results) < 2:
            if not cache_hit and len(query.split()) <= 3:
                try:
                    fallback_results, _ = search_segments(query, metadata, index, embedder, idx_hash, k=40, threshold=0.20)
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
        
        return results
        
    finally:
        progress_placeholder.empty()
        status_placeholder.empty()
        skeleton_placeholder.empty()


def display_search_results(results: List[Dict[str, Any]], query: str) -> None:
    """Display search results with animations and fallback messages."""
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
                <div style="font-size: 0.875rem; color: #9aa2b1;">Found {len(results)} relevant segment{'s' if len(results) != 1 else ''}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        render_search_results(results, query)
    
    log_query(query, None, results)
