import os
import re
import time
import datetime
from typing import List, Dict, Any, Tuple

import streamlit as st

from SmartTA_Extensions.utils.helpers import load_youtube_lectures

# Stopwords for query term highlighting (must match scoring.py for consistency)
_STOPWORDS = {
    'the', 'is', 'are', 'a', 'an', 'what', 'how', 'to', 'of', 'and', 'in', 'on',
    'for', 'with', 'by', 'from', 'at', 'as', 'that', 'this', 'it', 'does', 'do',
    'did', 'be', 'can', 'if', 'or', 'we', 'you', 'your', 'our', 'i', 'am', 'was',
    'were', 'been', 'have', 'has', 'had', 'will', 'would', 'should', 'could', 'may',
    'might', 'must', 'shall', 'about', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once', 'so',
    'but', 'not', 'than', 'there', 'here', 'where', 'when', 'why', 'who', 'which',
    'these', 'those', 'some', 'any', 'all', 'both', 'each', 'every', 'another', 'other',
    'such', 'no', 'nor', 'only', 'own', 'same', 'very', 'just', 'even', 'also', 'too',
    'more', 'most', 'much', 'many', 'few', 'less', 'least', 'use', 'uses', 'used', 'using',
    'get', 'gets', 'got', 'make', 'makes', 'made', 'take', 'takes', 'took', 'give', 'gives',
    'gave', 'put', 'puts', 'see', 'saw', 'seen', 'go', 'goes', 'went', 'come', 'comes', 'came'
}

# Important ML/AI acronyms and technical terms that should always be highlighted (even if short)
_IMPORTANT_TERMS = {
    # ML Algorithms
    'svm', 'knn', 'pca', 'lda', 'gbm', 'xgb', 'rf', 'dt', 'nb', 'svc', 'svr',
    'gbdt', 'cart', 'id3', 'c45', 'em', 'dbscan', 'optics', 'birch',
    # Neural Networks
    'cnn', 'rnn', 'gru', 'lstm', 'gan', 'vae', 'mlp', 'dnn', 'ann', 'ffnn',
    'bnn', 'snn', 'wgan', 'dcgan', 'unet', 'yolo', 'rcnn', 'fcn',
    # NLP & AI
    'nlp', 'llm', 'gpt', 'bert', 'ai', 'ml', 'dl', 'rl', 'elmo', 'xlnet',
    't5', 'bart', 'ner', 'pos', 'seq2seq', 'tfidf', 'bow',
    # Optimization & Training
    'sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamw', 'lars', 'lamb',
    'gd', 'bgd', 'mbgd', 'lr', 'wd', 'l1', 'l2', 'dropout', 'bn', 'ln',
    # Activation Functions
    'relu', 'elu', 'selu', 'gelu', 'prelu', 'swish', 'mish', 'tanh', 'silu',
    # Metrics & Loss
    'auc', 'roc', 'mse', 'mae', 'rmse', 'map', 'iou', 'dice', 'f1', 'pr',
    'ce', 'bce', 'cce', 'kl', 'js', 'wasserstein', 'hinge', 'huber',
    # Data Science
    'eda', 'etl', 'cv', 'loocv', 'kfold', 'oos', 'oob', 'smote', 'adasyn',
    # Math/Stats
    'pdf', 'cdf', 'pmf', 'std', 'var', 'cov', 'corr', 'sse', 'ssr', 'sst',
    'aic', 'bic', 'rss', 'tss', 'r2', 'mle', 'map', 'em', 'mcmc', 'gibbs',
    # Computer Vision
    'yolo', 'rcnn', 'ssd', 'fpn', 'roi', 'rpn', 'nms', 'iou', 'map', 'coco',
    'voc', 'imagenet', 'resnet', 'vgg', 'alexnet', 'googlenet', 'inception',
    # Architectures
    'vit', 'swin', 'bert', 'gpt', 'clip', 'dalle', 'stable', 'diffusion',
    # Hardware & Infrastructure
    'gpu', 'cpu', 'tpu', 'fpga', 'asic', 'cuda', 'cudnn', 'tensorrt',
    # Data Formats & Tools
    'api', 'sql', 'csv', 'json', 'xml', 'yaml', 'yml', 'hdf5', 'h5',
    'parquet', 'avro', 'orc', 'tfrecord', 'numpy', 'pandas', 'pytorch',
    # Web & Protocols
    'html', 'css', 'js', 'http', 'https', 'rest', 'grpc', 'websocket',
    'url', 'uri', 'ip', 'tcp', 'udp', 'dns', 'ssl', 'tls', 'oauth',
    # Other Common Acronyms
    'ocr', 'asr', 'tts', 'nlg', 'nlu', 'nmt', 'qa', 'ir', 'kg', 'rag'
}


def render_app_header(lecture_count: int, total_queries: int) -> None:
    """Render modern glassmorphic header with stats."""
    st.markdown(f"""
<style>
@keyframes gradient-shift {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}
.stat-card {{
    transition: all 0.3s ease;
}}
.stat-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 198, 255, 0.3);
}}
</style>
<div style="background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            background-size: 200% 200%;
            animation: gradient-shift 8s ease infinite;
            padding: 2.5rem 2rem; border-radius: 20px; margin-bottom: 2rem;
            border: 1px solid rgba(0, 198, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 2rem;">
        <div style="flex: 1; min-width: 300px;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
                <div style="width: 4px; height: 48px; background: linear-gradient(180deg, #00C6FF 0%, #2e7cf6 100%); border-radius: 4px;"></div>
                <div>
                    <h1 style="color: #00C6FF; font-weight: 800; font-size: 2.25rem; margin: 0; letter-spacing: -0.5px;
                               text-shadow: 0 0 30px rgba(0, 198, 255, 0.5);">
                        🎓 SmartTA Extensions
                    </h1>
                    <p style="color: #9aa2b1; font-size: 0.95rem; margin: 0.5rem 0 0 0; font-weight: 500;">
                        Intelligent YouTube Lecture Search & Navigation
                    </p>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 1.25rem; flex-wrap: wrap;">
            <div class="stat-card" style="text-align: center; 
                                          background: linear-gradient(135deg, rgba(0, 198, 255, 0.15) 0%, rgba(46, 124, 246, 0.15) 100%);
                                          padding: 1rem 1.5rem; border-radius: 16px; 
                                          border: 1px solid rgba(0, 198, 255, 0.3);
                                          backdrop-filter: blur(10px);
                                          box-shadow: 0 4px 16px rgba(0, 198, 255, 0.2);">
                <div style="color: #7dd3fc; font-size: 0.7rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 1px; text-transform: uppercase;">📚 Lectures</div>
                <div style="color: #00C6FF; font-size: 2rem; font-weight: 900; text-shadow: 0 0 20px rgba(0, 198, 255, 0.4);">{lecture_count}</div>
            </div>
            <div class="stat-card" style="text-align: center;
                                          background: linear-gradient(135deg, rgba(46, 124, 246, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
                                          padding: 1rem 1.5rem; border-radius: 16px;
                                          border: 1px solid rgba(99, 102, 241, 0.3);
                                          backdrop-filter: blur(10px);
                                          box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);">
                <div style="color: #a5b4fc; font-size: 0.7rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 1px; text-transform: uppercase;">🔍 Searches</div>
                <div style="color: #6366f1; font-size: 2rem; font-weight: 900; text-shadow: 0 0 20px rgba(99, 102, 241, 0.4);">{total_queries}</div>
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

    # Check if top results are weak (low confidence matches)
    max_score = max([r.get('score', r.get('kw_overlap', 0)) for r in results], default=0)
    is_weak_match = max_score < 0.65  # Less than 65% confidence (raised threshold for better detection)
    
    # Enhanced results header (dark theme)
    header_color = "#f59e0b" if is_weak_match else "#00C6FF"
    header_icon = "⚠️" if is_weak_match else "🎯"
    header_title = "Found Related Content (Low Confidence)" if is_weak_match else f"Found {len(results)} Relevant Segment{'s' if len(results) != 1 else ''}"
    
    st.markdown(f"""
    <style>
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateY(-10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    <div style="background: linear-gradient(135deg, rgba({'245, 158, 11' if is_weak_match else '0, 198, 255'}, 0.1) 0%, rgba({'217, 119, 6' if is_weak_match else '46, 124, 246'}, 0.1) 100%);
                padding: 2rem; border-radius: 16px; margin: 1.5rem 0;
                border-left: 5px solid {header_color};
                box-shadow: 0 4px 20px rgba({'245, 158, 11' if is_weak_match else '0, 198, 255'}, 0.2);
                animation: slideIn 0.5s ease-out;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.75rem;">
            <div style="font-size: 2.5rem;">{header_icon}</div>
            <h3 style="color: {header_color}; margin: 0; font-size: 1.4rem; font-weight: 700; letter-spacing: -0.3px;">
                {header_title}
            </h3>
        </div>
        <p style="color: #9aa2b1; margin: 0 0 0 4rem; font-size: 1rem; line-height: 1.5;">
            <span style="color: #{'fbbf24' if is_weak_match else '7dd3fc'}; font-weight: 600;">Query:</span> <em style="color: #e2e8f0;">"{query_text}"</em>
        </p>
    """, unsafe_allow_html=True)
    
    # Show compact warning for weak matches
    if is_weak_match:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.1); padding: 0.5rem 1rem; border-radius: 8px; 
                    margin: 0.75rem 0; border-left: 3px solid #f59e0b; font-size: 0.85rem;">
            <span style="color: #fbbf24;">ℹ️ <strong>Low confidence:</strong> Topic may not be directly covered in lectures.</span>
        </div>
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

    def _cat_of(lecture_id_or_title: str):
        # First try direct ID lookup (most common case)
        if lecture_id_or_title in _lec_info:
            _info = _lec_info[lecture_id_or_title]
            return _info.get("category", "Uncategorized"), _info
        
        # Fallback: try matching by title (exact match)
        for _id, _info in _lec_info.items():
            if _info.get("title") == lecture_id_or_title:
                return _info.get("category", "Uncategorized"), _info
        
        # Fallback 2: try fuzzy match with better normalization
        # Normalize: lowercase, convert underscores/dots to spaces, collapse whitespace
        def normalize(s):
            # Handle numeric prefixes: "28_polynomial" -> "28 polynomial", "28. Poly" -> "28 poly"
            s = s.lower().replace("_", " ").replace(".", " ")
            # Collapse multiple spaces
            s = re.sub(r'\s+', ' ', s).strip()
            return s
        
        normalized_input = normalize(lecture_id_or_title)
        for _id, _info in _lec_info.items():
            title = _info.get("title", "")
            normalized_title = normalize(title)
            # Also check normalized ID in case title isn't set
            normalized_id = normalize(_id)
            
            if normalized_input == normalized_title or normalized_input == normalized_id:
                return _info.get("category", "Uncategorized"), _info
            # Partial match as last resort (input contains normalized title/id)
            if normalized_input in normalized_title or normalized_title in normalized_input:
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
                st.markdown(f"<div style='margin-bottom:1.25rem'><span style='background:{cat_color};color:white;padding:0.4rem 1rem;border-radius:20px;font-size:0.8rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;box-shadow:0 2px 8px rgba(0,0,0,0.3);'>📂 {_cat}</span></div>", unsafe_allow_html=True)
                if _info and _info.get('professor_note'):
                    st.markdown(f"<div style='background:linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.1) 100%);border-left:4px solid {cat_color};padding:0.75rem 1rem;border-radius:12px;margin:0.75rem 0 1.25rem 0;box-shadow:0 2px 12px rgba(99,102,241,0.15);'><span style='font-size:1.2rem;margin-right:0.5rem;'>🧑‍🏫</span><em style='color:#e2e8f0;'>{_info['professor_note']}</em></div>", unsafe_allow_html=True)
                _render_lecture_results(items, query_text)
        else:
            _cat, _info = cat_cache[lecture]
            cat_color = _cat_color(_cat)
            st.markdown(f"<div style='margin-bottom:1.25rem'><span style='background:{cat_color};color:white;padding:0.4rem 1rem;border-radius:20px;font-size:0.8rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;box-shadow:0 2px 8px rgba(0,0,0,0.3);'>📂 {_cat}</span></div>", unsafe_allow_html=True)
            if _info and _info.get('professor_note'):
                st.markdown(f"<div style='background:linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(139,92,246,0.1) 100%);border-left:4px solid {cat_color};padding:0.75rem 1rem;border-radius:12px;margin:0.75rem 0 1.25rem 0;box-shadow:0 2px 12px rgba(99,102,241,0.15);'><span style='font-size:1.2rem;margin-right:0.5rem;'>🧑‍🏫</span><em style='color:#e2e8f0;'>{_info['professor_note']}</em></div>", unsafe_allow_html=True)
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
                badge_html = '<div style="display: flex; gap: 0.75rem; margin-bottom: 1.25rem; flex-wrap: wrap;">'
                if 'score' in r:
                    score = r['score']
                    if score >= 0.7:
                        bg = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                        icon = "🟢"
                    elif score >= 0.5:
                        bg = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
                        icon = "🟡"
                    else:
                        bg = "linear-gradient(135deg, #6b7280 0%, #475569 100%)"
                        icon = "⚪"
                    badge_html += f'''
                    <span style="background: {bg}; color: white; padding: 0.4rem 1rem; 
                                 border-radius: 16px; font-size: 0.8rem; font-weight: 700;
                                 box-shadow: 0 2px 8px rgba(0,0,0,0.3); letter-spacing: 0.5px;">
                        {icon} Score: {score:.1%}
                    </span>
                    '''
                if 'kw_overlap' in r:
                    kw = r['kw_overlap']
                    if kw >= 0.3:
                        bg_kw = "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)"
                        icon_kw = "🔑"
                    else:
                        bg_kw = "linear-gradient(135deg, #64748b 0%, #475569 100%)"
                        icon_kw = "🔍"
                    badge_html += f'''
                    <span style="background: {bg_kw}; color: white; padding: 0.4rem 1rem; 
                                 border-radius: 16px; font-size: 0.8rem; font-weight: 700;
                                 box-shadow: 0 2px 8px rgba(0,0,0,0.3); letter-spacing: 0.5px;">
                        {icon_kw} Keywords: {kw:.1%}
                    </span>
                    '''
                badge_html += '</div>'
                st.markdown(badge_html, unsafe_allow_html=True)

            # Highlight query terms
            raw_text = r.get('text', '')[:600]
            q_tokens_raw = re.findall(r"\w+", query_text.lower())
            # Smart filtering: highlight if (4+ chars OR important term) AND not stopword
            q_tokens = {
                t for t in q_tokens_raw 
                if t not in _STOPWORDS and (len(t) > 3 or t in _IMPORTANT_TERMS)
            }
            # Don't highlight anything if only stopwords remain (prevents highlighting "is", "what", etc.)
            if not q_tokens:
                q_tokens = set()  # Empty set = no highlighting
            # Sort by length to avoid nested replacement issues (longest first)
            q_tokens_sorted = sorted(q_tokens, key=len, reverse=True)
            # Escape HTML to prevent injection and use placeholder for highlighting
            import html
            highlighted = html.escape(raw_text)
            # Use unique placeholder to avoid re-matching
            for qt in q_tokens_sorted:
                pattern = re.compile(rf"\b({re.escape(qt)})\b", re.IGNORECASE)
                highlighted = pattern.sub(r'<mark style="background: rgba(251, 191, 36, 0.3); padding: 2px 4px; border-radius: 3px; font-weight: 600; color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.5);">\1</mark>', highlighted)

            st.markdown(f'<div style="line-height: 1.6; color: #F3F3F3;">{highlighted}</div>', unsafe_allow_html=True)
            st.markdown("")

            # Action button
            if st.button("▶️ Jump to Timestamp", key=f"jump_{global_idx}", width='stretch', type="primary"):
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
