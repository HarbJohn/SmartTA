"""Streamlit entrypoint for the SmartTA unified experience."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable even when Streamlit runs the script directly
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from smartta_unified import config
from smartta_unified.chat import ensure_session_defaults, render_student_chat
from smartta_unified.dashboard import render_instructor_dashboard
from smartta_unified.style import BASE_STYLES
from smartta_unified.youtube import render_professor_dashboard, render_youtube_library


def _load_rag_resources() -> None:
    """Load FAISS indexes once per session."""
    if not config.ENABLE_RAG or not config.rag_engine1:
        return
    if st.session_state.get("engine_loaded"):
        return
    with st.spinner("Loading multimodal RAG indexes…"):
        config.rag_engine1.load_indexes()
        st.session_state.engine_loaded = True
    if config.psutil:
        try:
            mem_mb = config.psutil.Process().memory_info().rss / 1024 / 1024
            st.sidebar.caption(f"Memory usage: {mem_mb:.0f} MB")
        except Exception:
            pass


def _render_mode_selector() -> str:
    """Return the currently selected mode."""
    labels, modes = [], []
    if config.ENABLE_RAG or config.ENABLE_YOUTUBE:
        labels.append("🎓 Student Assistant")
        modes.append("Student")
    if config.ENABLE_RAG:
        labels.append("📊 Instructor Dashboard")
        modes.append("Instructor")
    if config.ENABLE_YOUTUBE:
        labels.append("🎯 Professor Dashboard")
        modes.append("Professor")
    if not modes:
        labels, modes = ["🎓 Student Assistant"], ["Student"]

    st.markdown(
        "<div style='border:1px solid rgba(0,198,255,.2); border-radius:20px; padding:20px; "
        "max-width:900px; margin:16px auto;'>",
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1, 1])
    with center:
        picked = st.radio("", labels, horizontal=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    return next((mode for mode, label in zip(modes, labels) if label == picked), modes[0])


def _require_login() -> None:
    """Simple username/password for instructor/professor tabs."""
    if st.session_state.get("authenticated"):
        return
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Instructor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col_a, col_b = st.columns([1, 1])
    if col_a.button("Login"):
        if username == "eece" and password == "690":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")
    if col_b.button("Cancel"):
        st.info("Login canceled.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


def run() -> None:
    ensure_session_defaults()
    st.session_state.setdefault("engine_loaded", False)

    st.set_page_config(page_title="🧠 SmartTA Unified Assistant", layout="wide")
    st.markdown(BASE_STYLES, unsafe_allow_html=True)
    st.markdown(
        "<div class='smartta-title'>🧠 SmartTA — Multimodal RAG Assistant</div>",
        unsafe_allow_html=True,
    )

    if config.RAG_IMPORT_ERROR:
        st.error(f"Failed to import rag_engine1: {config.RAG_IMPORT_ERROR}")
    if config.EXTENSION_IMPORT_ERROR:
        st.sidebar.warning(
            f"YouTube extensions not fully available: {config.EXTENSION_IMPORT_ERROR}"
        )

    if config.ENABLE_RAG and config.rag_engine1 and not st.session_state.engine_loaded:
        _load_rag_resources()
    if config.ENABLE_RAG and not config.rag_engine1:
        st.sidebar.warning("Student assistant disabled (missing rag_engine1).")
    if config.ENABLE_RAG and not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Missing OPENAI_API_KEY. Put it in .env or your environment.")

    mode = _render_mode_selector()

    if mode == "Student":
        if config.ENABLE_RAG and config.rag_engine1:
            render_student_chat(k=5, min_conf=0.25, rag_query_limit=config.RAG_QUERY_LIMIT)
        elif config.ENABLE_RAG:
            st.info("Student chat disabled (RAG engine unavailable).")
        if config.ENABLE_YOUTUBE:
            render_youtube_library()
    elif mode == "Instructor":
        if not config.ENABLE_RAG or not config.rag_engine1:
            st.warning("Instructor dashboard requires the RAG engine.")
            return
        _require_login()
        st.markdown("## 🧑‍🏫 Instructor Dashboard (RAG)")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        render_instructor_dashboard()
    elif mode == "Professor":
        if not config.ENABLE_YOUTUBE:
            st.warning("Professor dashboard requires the YouTube extensions.")
            return
        _require_login()
        st.markdown("## 🎯 Professor Dashboard (YouTube)")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        render_professor_dashboard()
    else:
        st.error("Unknown mode selected.")


if __name__ == "__main__":  
    run()
