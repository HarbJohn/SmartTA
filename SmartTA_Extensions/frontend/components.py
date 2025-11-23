import os
import re
import time
import datetime
from typing import List, Dict, Any, Tuple

import streamlit as st

from utils.helpers import load_youtube_lectures


def render_app_header(lecture_count: int, total_queries: int) -> None:
    """Render the gradient header with stats.
    Mirrors the original inline HTML block from app.py without logic changes.
    """
    st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3); position: relative; overflow: hidden;">
    <div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; 
                background: rgba(255,255,255,0.1); border-radius: 50%; filter: blur(40px);"></div>
    <div style="position: absolute; bottom: -30px; left: -30px; width: 150px; height: 150px; 
                background: rgba(255,255,255,0.1); border-radius: 50%; filter: blur(30px);"></div>
    <div style="position: relative; z-index: 1; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 2rem;">
        <div>
            <h1 style="color: white; font-weight: 800; font-size: 2.5rem; margin: 0; 
                       text-shadow: 0 2px 20px rgba(0,0,0,0.2);">
                🎓 SmartTA
            </h1>
            <p style="color: rgba(255,255,255,0.95); font-size: 1.15rem; margin: 0.75rem 0 0 0; 
                      font-weight: 500; text-shadow: 0 1px 10px rgba(0,0,0,0.1);">
                AI-Powered Teaching Assistant | Semantic Search · Video Navigation · Real-time Analytics
            </p>
        </div>
        <div style="display: flex; gap: 1.5rem;">
            <div style="text-align: center; background: rgba(13,17,23,0.5); padding: 1rem 1.5rem; border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);">
                <div style="color: rgba(230,238,248,0.8); font-size: 0.875rem; font-weight: 600; margin-bottom: 0.25rem;">LECTURES</div>
                <div style="color: #e6eef8; font-size: 2rem; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">{lecture_count}</div>
            </div>
            <div style="text-align: center; background: rgba(13,17,23,0.5); padding: 1rem 1.5rem; border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);">
                <div style="color: rgba(230,238,248,0.8); font-size: 0.875rem; font-weight: 600; margin-bottom: 0.25rem;">SEARCHES</div>
                <div style="color: #e6eef8; font-size: 2rem; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">{total_queries}</div>
            </div>
            <div style="text-align: center; background: rgba(13,17,23,0.5); padding: 1rem 1.5rem; border-radius: 12px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);">
                <div style="color: rgba(230,238,248,0.8); font-size: 0.875rem; font-weight: 600; margin-bottom: 0.25rem;">AI MODE</div>
                <div style="color: #e6eef8; font-size: 1.25rem; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.2);">🚀 ACTIVE</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_search_results(results: List[Dict[str, Any]], query_text: str) -> None:
    """Render search results grouped by lecture.
    Copies original behavior, with a helper for per-lecture rendering.
    """
    if not results:
        st.info("💡 No results yet. Enter a question above to search.")
        return

    # Group results by lecture
    from collections import defaultdict
    grouped: Dict[str, list] = defaultdict(list)
    for idx, r in enumerate(results):
        grouped[r['lecture']].append((idx, r))

    # Enhanced results header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0;
                border-left: 4px solid #667eea;">
        <h3 style="color: #e6eef8; margin: 0 0 0.5rem 0; font-size: 1.25rem;">
            📚 Found {len(results)} segment{'s' if len(results) != 1 else ''}
        </h3>
        <p style="color: #b9c6d8; margin: 0; font-size: 0.95rem;">
            🔍 <strong>Query:</strong> <em>{query_text}</em>
        </p>
    """, unsafe_allow_html=True)

    # Show lecture distribution if multiple lectures
    if len(grouped) > 1:
        dist_items = [(lec, len(segs)) for lec, segs in list(grouped.items())[:3]]
        cols = st.columns(min(len(dist_items), 4))
        for idx, (lec, count) in enumerate(dist_items):
            lec_short = lec[:25] + '...' if len(lec) > 25 else lec
            with cols[idx]:
                st.info(f"📖 **{count}** from _{lec_short}_")

    st.markdown("</div>", unsafe_allow_html=True)

    # Prepare category mapping for badges
    _yt_map = load_youtube_lectures()
    _lec_info = _yt_map.get("lectures", {})

    def _cat_of(title: str):
        for _id, _info in _lec_info.items():
            if _info.get("title") == title:
                return _info.get("category", "Uncategorized"), _info
        return "Uncategorized", None

    # Deterministic color for category
    def _cat_color(name: str) -> str:
        palette = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#ec4899"]
        h = sum(ord(c) for c in name) if name else 0
        return palette[h % len(palette)]

    # Cache category lookups to avoid duplicate lookups
    cat_cache = {lec: _cat_of(lec) for lec in grouped.keys()}

    # Render grouped by lecture
    for lec_idx, (lecture, items) in enumerate(grouped.items()):
        if len(grouped) > 1:
            lec_display = lecture if len(lecture) <= 45 else lecture[:42] + "..."
            _cat, _info = cat_cache[lecture]
            cat_color = _cat_color(_cat)
            header_text = f"📖 {lec_display} ({len(items)} segment{'s' if len(items)!=1 else ''}) · {_cat}"
            with st.expander(label=header_text, expanded=(lec_idx == 0)):
                st.markdown(f"<div style='margin-bottom:1rem'><span style='background:{cat_color};color:white;padding:.25rem .75rem;border-radius:12px;font-size:.85rem;font-weight:600'>📂 {_cat}</span></div>", unsafe_allow_html=True)
                if _info and _info.get('professor_note'):
                    st.markdown(f"<div style='background:rgba(102,126,234,0.12);border-left:3px solid {cat_color};padding:.5rem .75rem;border-radius:8px;margin:.5rem 0 1rem 0;color:#b9c6d8'>🧑‍🏫 <em>{_info['professor_note']}</em></div>", unsafe_allow_html=True)
                _render_lecture_results(items, query_text)
        else:
            _cat, _info = cat_cache[lecture]
            cat_color = _cat_color(_cat)
            st.markdown(f"<div style='margin-bottom:1rem'><span style='background:{cat_color};color:white;padding:.25rem .75rem;border-radius:12px;font-size:.85rem;font-weight:600'>📂 {_cat}</span></div>", unsafe_allow_html=True)
            if _info and _info.get('professor_note'):
                st.markdown(f"<div style='background:rgba(102,126,234,0.12);border-left:3px solid {cat_color};padding:.5rem .75rem;border-radius:8px;margin:.5rem 0 1rem 0;color:#b9c6d8'>🧑‍🏫 <em>{_info['professor_note']}</em></div>", unsafe_allow_html=True)
            _render_lecture_results(items, query_text)


def _render_lecture_results(items: List[Tuple[int, Dict[str, Any]]], query_text: str) -> None:
    """Render results for a single lecture with enhanced visuals.
    Mirrors original behavior.
    """
    for global_idx, r in items:
        start_s = int(r.get('start', 0))
        end_s = int(r.get('end', start_s + 15))
        time_str = f"{datetime.timedelta(seconds=start_s)}"
        duration = end_s - start_s

        expanded_default = (global_idx == 0) or (st.session_state.get('last_clicked_result') == global_idx)

        # Color-coded score badge
        score_val = r.get('score', r.get('kw_overlap', 0))
        if score_val >= 0.7:
            badge_color = "#10b981"  # green
            badge_icon = "🟢"
        elif score_val >= 0.5:
            badge_color = "#f59e0b"  # amber
            badge_icon = "🟡"
        else:
            badge_color = "#6b7280"  # gray
            badge_icon = "⚪"

        display_header = f"{badge_icon} ⏱️ {time_str} · {duration}s"
        with st.expander(display_header, expanded=expanded_default):
            # Score badges at top
            if 'score' in r or 'kw_overlap' in r:
                badge_html = '<div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">'
                if 'score' in r:
                    score = r['score']
                    bg = "#10b981" if score >= 0.7 else "#f59e0b" if score >= 0.5 else "#6b7280"
                    badge_html += f'''
                    <span style="background: {bg}; color: white; padding: 0.25rem 0.75rem; 
                                 border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                        🎯 Score: {score:.2f}
                    </span>
                    '''
                if 'kw_overlap' in r:
                    kw = r['kw_overlap']
                    bg_kw = "#8b5cf6" if kw >= 0.3 else "#94a3b8"
                    badge_html += f'''
                    <span style="background: {bg_kw}; color: white; padding: 0.25rem 0.75rem; 
                                 border-radius: 12px; font-size: 0.75rem; font-weight: 600;">
                        🔑 Keywords: {kw:.2f}
                    </span>
                    '''
                badge_html += '</div>'
                st.markdown(badge_html, unsafe_allow_html=True)

            # Highlight query terms
            raw_text = r.get('text', '')[:600]
            q_tokens = sorted(set(re.findall(r"\w+", query_text.lower())), key=len, reverse=True)
            highlighted = raw_text
            for qt in q_tokens:
                pattern = re.compile(rf"\b{re.escape(qt)}\b", re.IGNORECASE)

                def _safe_mark(match: re.Match) -> str:
                    start = match.start()
                    snippet = highlighted[max(0, start - 30): start]
                    if "smartta-highlight" in snippet:
                        return match.group(0)
                    return f'<mark class="smartta-highlight">{match.group(0)}</mark>'

                highlighted = pattern.sub(_safe_mark, highlighted)

            st.markdown(f'<div style="line-height: 1.6; color: #b9c6d8;">{highlighted}</div>', unsafe_allow_html=True)
            st.markdown("")

            # Action button
            if st.button("▶️ Jump to Timestamp", key=f"jump_{global_idx}", use_container_width=True, type="primary"):
                st.session_state['jump_to'] = {
                    'lecture': r['lecture'],
                    'start': start_s,
                    'end': end_s,
                    'ts': time.time()
                }
                st.session_state['last_clicked_result'] = global_idx
                st.rerun()


def render_course_materials(uploads_dir: str) -> None:
    """Render the course materials section with download buttons.
    Accepts uploads_dir to avoid coupling to app.py constants.
    """
    yt_data = load_youtube_lectures()
    all_materials = yt_data.get("course_materials", [])

    if all_materials:
        for material in sorted(all_materials, key=lambda x: x.get('order', 999)):
            with st.expander(f"{material['title']}", expanded=False):
                st.markdown(f"*{material['description']}*")
                st.markdown("---")

                for file_info in material.get('files', []):
                    file_path = os.path.join(uploads_dir, file_info['filename'])
                    if os.path.exists(file_path):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{file_info['filename']}**")
                            if file_info.get('note'):
                                st.caption(f"Note: {file_info['note']}")
                        with col2:
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download",
                                    data=f,
                                    file_name=file_info['filename'],
                                    mime=file_info.get('type', 'application/octet-stream'),
                                    key=f"dl_material_{material['id']}_{file_info['filename']}"
                                )
    else:
        st.info("No course materials available yet. Professor can add materials from the Professor Dashboard.")
