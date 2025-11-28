"""Control panel and system maintenance UI for Professor Dashboard.
"""
import os
import json
import time
import logging
import streamlit as st
import faiss

from SmartTA_Extensions.backend.indexing import load_sentence_transformer

logger = logging.getLogger(__name__)


def render_control_panel(LOG_PATH: str, META_PATH: str, INDEX_PATH: str, TRANSCRIPT_DIR: str) -> None:
    """Render the Control Panel with quick actions."""
    st.markdown("### ⚙️ Control Panel")
    st.caption("Quick access to system maintenance tasks")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", width='stretch', type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Clear Analytics", width='stretch'):
            _handle_clear_analytics(LOG_PATH)
    
    with col3:
        if st.button("🧹 Clean Index", width='stretch'):
            _handle_clean_index(META_PATH, INDEX_PATH, TRANSCRIPT_DIR)
    
    with col4:
        if st.button("⚠️ Clear All Data", width='stretch'):
            st.session_state['confirm_clear_all'] = True


def _handle_clear_analytics(LOG_PATH: str) -> None:
    """Clear analytics log file."""
    if os.path.exists(LOG_PATH):
        try:
            os.remove(LOG_PATH)
            st.success("✅ Analytics cleared")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear analytics: {e}")
    else:
        st.info("No analytics data to clear")


def _handle_clean_index(META_PATH: str, INDEX_PATH: str, TRANSCRIPT_DIR: str) -> None:
    """Remove index entries for deleted lectures."""
    try:
        if not os.path.exists(META_PATH):
            st.info("No index to clean")
            return
        
        with open(META_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Get list of existing transcript files
        existing_transcripts = set()
        if os.path.exists(TRANSCRIPT_DIR):
            existing_transcripts = {os.path.splitext(f)[0] for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.txt')}
        
        # Filter metadata to only include existing lectures
        original_count = len(metadata)
        metadata = [m for m in metadata if m.get('lecture', '').replace('.txt', '') in existing_transcripts]
        removed_count = original_count - len(metadata)
        
        if removed_count > 0:
            # Save cleaned metadata
            with open(META_PATH, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Rebuild index
            embedder = load_sentence_transformer()
            if embedder and metadata:
                texts = [m['text'] for m in metadata]
                embeddings = embedder.encode(texts, show_progress_bar=True)
                
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                faiss.write_index(index, INDEX_PATH)
            
            st.success(f"✅ Cleaned index: removed {removed_count} deleted lecture(s)")
            time.sleep(0.5)
            st.rerun()
        else:
            st.info("Index is clean - no deleted lectures found")
    except Exception as e:
        st.error(f"Failed to clean index: {e}")
        logger.error(f"Clean index error: {e}")
