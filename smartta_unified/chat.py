from __future__ import annotations

import html
import re
import tempfile
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from . import config
from .helpers import (
    append_chat_log,
    aub_logo_path,
    sanitize_question,
    update_feedback_entry,
)


SESSION_DEFAULTS = {
    "chat_history": [],
    "uploader_key": 0,
    "submitted": False,
    "authenticated": False,
    "rag_query_count": 0,
    "ui_scale": 100,
    "pending_tmp_img": None,
    "dl_txt": None,
    "dl_pdf": None,
    "dl_txt_key": 0,
    "dl_pdf_key": 0,
    "chat_input": "",
}


def ensure_session_defaults() -> None:
    """Initialize Streamlit session_state keys used by the chat UI."""
    for key, value in SESSION_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _extract_pdf_page(raw_key: Any) -> tuple[str, str]:
    """Normalize pdf/page identifiers from debug dictionaries."""
    if isinstance(raw_key, (tuple, list)):
        pdf = str(raw_key[0])
        page = str(raw_key[1]) if len(raw_key) > 1 else "?"
        return pdf, page
    parts = re.split(r"[:,()'\\s]+", str(raw_key))
    pdf = next((p for p in parts if p.endswith(".pdf")), "?")
    page = next((p for p in parts if p.isdigit()), "?")
    return pdf, page


def _render_score_row(pdf: str, page: Any, score: float, context: bool = False) -> str:
    safe_pdf = html.escape(str(pdf))
    safe_page = html.escape(str(page))
    score_str = f"{float(score):.3f}"
    if context:
        return (
            "<div class='retrieval-row'>"
            "<span class='bullet-dot'>•</span>"
            f"<span class='pdf-pill'>{safe_pdf}</span>"
            f"<span class='page-pill'>p.{safe_page}</span>"
            "<span class='arrow-icon'>→</span>"
            f"<span class='score-pill'>{score_str}</span>"
            "</div>"
        )
    return (
        "<div class='retrieval-row'>"
        "<span class='bullet-dot'>•</span>"
        f"<span class='pdf-pill'>{safe_pdf}</span>"
        f"<span class='page-pill'>p.{safe_page}</span>"
        "<span class='arrow-icon'>→</span>"
        f"<span class='score-pill'>{score_str}</span>"
        "</div>"
    )


def _confidence_from_chat(chat: Dict[str, Any]) -> float:
    if chat.get("confidence") is not None:
        try:
            return float(chat["confidence"])
        except Exception:
            pass
    contexts = chat.get("contexts") or []
    best = 0.0
    for ctx in contexts:
        try:
            best = max(best, float(ctx.get("score", 0.0)))
        except Exception:
            continue
    return best


def _render_chat_history() -> None:
    """Render chat history bubbles and diagnostics."""
    rag_engine = config.rag_engine1
    st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
    for global_idx, chat in enumerate(st.session_state.chat_history):
        st.markdown(
            f"<div class='chat-row user'><div class='avatar user'>You</div>"
            f"<div><div class='bubble user'>{chat['question']}</div></div></div>",
            unsafe_allow_html=True,
        )
        if isinstance(chat.get("img_path"), str) and Path(chat["img_path"]).exists():
            st.image(chat["img_path"], caption="Attached image", use_container_width=True)

        st.markdown(
            f"<div class='chat-row'><div class='avatar assistant'>TA</div>"
            f"<div><div class='bubble assistant'>{chat['answer']}</div></div></div>",
            unsafe_allow_html=True,
        )

        feedback_cols = st.columns([0.18, 0.18])
        with feedback_cols[0]:
            st.markdown("<div class='feedback-btn helpful'>", unsafe_allow_html=True)
            if st.button("👍 Helpful", key=f"helpful_{global_idx}"):
                chat["feedback"] = "helpful"
                update_feedback_entry(chat.get("qid"), "helpful")
                st.toast("Marked helpful ✅")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with feedback_cols[1]:
            st.markdown("<div class='feedback-btn unclear'>", unsafe_allow_html=True)
            if st.button("👎 Unclear", key=f"unclear_{global_idx}"):
                chat["feedback"] = "unclear"
                update_feedback_entry(chat.get("qid"), "unclear")
                st.toast("Flagged as unclear ⚠️")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        if chat.get("feedback"):
            tag = chat["feedback"]
            st.markdown(
                f"<div class='feedback-status {tag}-tag'>Marked {tag.title()}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Retrieval Diagnostics"):
            tabs = st.tabs(["openai→text", "img→img", "text↔img", "Contexts"])
            debug = chat.get("debug", {}) or {}

            with tabs[0]:
                st.markdown("<span class='model-badge model-text'>text-embedding-3</span>", unsafe_allow_html=True)
                if debug.get("text_openai") and rag_engine:
                    for key2, score in list(debug["text_openai"].items())[:5]:
                        try:
                            idx = int(key2)
                            if idx < len(rag_engine.chunk_meta):
                                meta = rag_engine.chunk_meta[idx]
                                st.markdown(
                                    _render_score_row(meta["pdf"], meta["page"], score),
                                    unsafe_allow_html=True,
                                )
                        except Exception:
                            continue
                else:
                    st.caption("No text matches.")

            with tabs[1]:
                st.markdown("<span class='model-badge model-img'>CLIP (img→img)</span>", unsafe_allow_html=True)
                records = debug.get("img2img") or {}
                if not records:
                    st.caption("No image→image matches.")
                else:
                    for key2, score in sorted(records.items(), key=lambda x: -x[1])[:5]:
                        pdf, page = _extract_pdf_page(key2)
                        st.markdown(
                            _render_score_row(pdf, page, score),
                            unsafe_allow_html=True,
                        )

            with tabs[2]:
                st.markdown("<span class='model-badge model-clip'>CLIP (text↔img)</span>", unsafe_allow_html=True)
                records = debug.get("text2img") or {}
                if not records:
                    st.caption("No text↔image matches.")
                else:
                    for key2, score in sorted(records.items(), key=lambda x: -x[1])[:5]:
                        pdf, page = _extract_pdf_page(key2)
                        st.markdown(
                            _render_score_row(pdf, page, score),
                            unsafe_allow_html=True,
                        )

            with tabs[3]:
                st.markdown("<span class='model-badge model-ctx'>Context Chunks</span>", unsafe_allow_html=True)
                contexts = chat.get("contexts") or []
                if contexts:
                    for ctx in contexts[:5]:
                        st.markdown(
                            _render_score_row(
                                ctx.get("pdf"),
                                ctx.get("page"),
                                ctx.get("score", 0.0),
                                context=True,
                            ),
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No chunks included.")

    if st.session_state.submitted and st.session_state.get("chat_input", "").strip():
        st.markdown(
            "<div class='chat-row'><div class='avatar assistant'>TA</div>"
            "<div class='typing'>SmartTA is thinking… "
            "<span class='dot'></span><span class='dot'></span><span class='dot'></span></div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _handle_attachment() -> None:
    """Manage image attachment lifecycle and preview."""
    uploaded = st.file_uploader(
        "Optional: Upload slide image",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded:
        if st.session_state.pending_tmp_img and Path(st.session_state.pending_tmp_img).exists():
            try:
                Path(st.session_state.pending_tmp_img).unlink(missing_ok=True)
            except Exception:
                pass
        suffix = Path(uploaded.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        st.session_state.pending_tmp_img = tmp.name

    if st.session_state.pending_tmp_img and Path(st.session_state.pending_tmp_img).exists():
        st.image(st.session_state.pending_tmp_img, caption="Attached image", use_container_width=True)
        if st.button("🗑️ Remove attachment"):
            try:
                Path(st.session_state.pending_tmp_img).unlink(missing_ok=True)
            except Exception:
                pass
            st.session_state.pending_tmp_img = None
            st.session_state.uploader_key += 1
            st.rerun()


def _export_text() -> None:
    lines = []
    for chat in st.session_state.chat_history:
        lines.append(f"You: {chat['question']}")
        if chat.get("img_path"):
            lines.append("[image attached]")
        lines.append(f"SmartTA: {chat['answer']}\n")
    st.session_state.dl_txt = "\n".join(lines).encode("utf-8")


def _export_pdf() -> None:
    deps = config.REPORTLAB_DEPS
    if not deps:
        return
    buf = BytesIO()
    doc = deps.SimpleDocTemplate(
        buf,
        pagesize=deps.A4,
        leftMargin=0.7 * deps.inch,
        rightMargin=0.7 * deps.inch,
        topMargin=0.6 * deps.inch,
        bottomMargin=0.6 * deps.inch,
    )
    styles = deps.getSampleStyleSheet()
    title_style = deps.ParagraphStyle(
        "TitleCenter",
        parent=styles["Title"],
        alignment=deps.TA_CENTER,
        fontSize=26,
        leading=30,
        spaceAfter=12,
    )
    subtitle_style = deps.ParagraphStyle(
        "SubCenter",
        parent=styles["BodyText"],
        alignment=deps.TA_CENTER,
        fontSize=14,
        leading=18,
        spaceAfter=8,
    )
    meta_style = deps.ParagraphStyle(
        "MetaCenter",
        parent=styles["Italic"],
        alignment=deps.TA_CENTER,
        fontSize=10,
        leading=12,
        spaceAfter=12,
    )
    qa_q_style = deps.ParagraphStyle(
        "QAQ", parent=styles["BodyText"], fontSize=11, leading=14, spaceAfter=4
    )
    qa_a_style = deps.ParagraphStyle(
        "QAA", parent=styles["BodyText"], fontSize=11, leading=16, spaceAfter=10
    )
    story: list[Any] = []
    logo = aub_logo_path()
    if logo and Path(logo).exists():
        story.append(deps.RLImage(logo, width=2.6 * deps.inch, height=2.6 * deps.inch))
        story.append(deps.Spacer(1, 0.12 * deps.inch))
    story.append(
        deps.Paragraph("<b>American University of Beirut – EECE Department</b>", title_style)
    )
    story.append(
        deps.Paragraph("EECE 490/690 — Intro to Machine Learning", subtitle_style)
    )
    story.append(
        deps.Paragraph(
            f"<i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            meta_style,
        )
    )
    story.append(deps.Spacer(1, 0.25 * deps.inch))
    for chat in st.session_state.chat_history:
        story.append(deps.Paragraph(f"<b>Question:</b> {chat.get('question','')}", qa_q_style))
        if isinstance(chat.get("img_path"), str) and Path(chat["img_path"]).exists():
            story.append(
                deps.RLImage(chat["img_path"], width=5.8 * deps.inch, height=3.6 * deps.inch)
            )
            story.append(deps.Spacer(1, 0.08 * deps.inch))
        story.append(deps.Paragraph(f"<b>Answer:</b> {chat.get('answer','')}", qa_a_style))
        story.append(deps.Spacer(1, 0.12 * deps.inch))
        story.append(
            deps.HRFlowable(width="100%", thickness=0.5, color=deps.colors.HexColor("#e6e6e6"))
        )
        story.append(deps.Spacer(1, 0.12 * deps.inch))
    doc.build(story)
    buf.seek(0)
    st.session_state.dl_pdf = buf.getvalue()


def render_student_chat(k: int, min_conf: float, rag_query_limit: int) -> None:
    """Render the combined chat + controls for the student assistant."""
    ensure_session_defaults()
    if not config.ENABLE_RAG:
        return
    if not config.rag_engine1:
        st.error("RAG engine failed to load. Check installation.")
        return

    st.markdown("### Display")
    st.session_state.ui_scale = st.slider(
        "🔎 UI scale", 90, 130, st.session_state.ui_scale, 1
    )
    st.markdown(
        f"<style>.stApp {{ zoom: {st.session_state.ui_scale/100}; }}</style>",
        unsafe_allow_html=True,
    )

    def _mark_submit() -> None:
        st.session_state.submitted = True

    st.divider()
    st.subheader("Conversation")
    _render_chat_history()

    st.markdown("---")
    st.markdown("### Ask a question:")
    question = st.text_input(
        "",
        placeholder="What's your question?",
        key="chat_input",
        on_change=_mark_submit,
    )

    _handle_attachment()

    controls = st.columns([1, 0.05, 1, 0.05, 1, 0.05, 1])
    with controls[0]:
        ask_clicked = st.button("Ask SmartTA 🚀", use_container_width=True)
    with controls[2]:
        clear_chat = st.button("Clear Chat 🧹", use_container_width=True)
    with controls[4]:
        export_txt = st.button("Export Chat (.txt)", use_container_width=True)
    with controls[6]:
        export_pdf = st.button(
            "Export PDF 📑", disabled=not bool(config.REPORTLAB_DEPS), use_container_width=True
        )

    if export_txt and st.session_state.chat_history:
        _export_text()

    if export_pdf and st.session_state.chat_history and config.REPORTLAB_DEPS:
        _export_pdf()

    downloads = st.columns(2)
    if st.session_state.dl_txt:
        if downloads[0].download_button(
            "⬇️ Download Chat (.txt)",
            data=st.session_state.dl_txt,
            file_name="SmartTA_chat.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"dl_txt_inline_{st.session_state.dl_txt_key}",
        ):
            st.session_state.dl_txt = None
            st.session_state.dl_txt_key += 1
            st.rerun()
    if st.session_state.dl_pdf:
        if downloads[1].download_button(
            "⬇️ Download PDF",
            data=st.session_state.dl_pdf,
            file_name="SmartTA_Session_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"dl_pdf_inline_{st.session_state.dl_pdf_key}",
        ):
            st.session_state.dl_pdf = None
            st.session_state.dl_pdf_key += 1
            st.rerun()

    if clear_chat:
        if st.session_state.pending_tmp_img and Path(st.session_state.pending_tmp_img).exists():
            try:
                Path(st.session_state.pending_tmp_img).unlink(missing_ok=True)
            except Exception:
                pass
        st.session_state.pending_tmp_img = None
        st.session_state.chat_history = []
        st.session_state.uploader_key += 1
        st.session_state.submitted = False
        st.session_state.dl_txt = None
        st.session_state.dl_pdf = None
        st.rerun()

    question = sanitize_question(question)
    if (ask_clicked or st.session_state.submitted) and question.strip():
        if st.session_state.rag_query_count >= rag_query_limit:
            st.error("Query limit reached for this session. Reload to continue.")
            st.session_state.submitted = False
            st.stop()

        tmp_img = (
            st.session_state.pending_tmp_img
            if st.session_state.pending_tmp_img
            and Path(st.session_state.pending_tmp_img).exists()
            else None
        )

        with st.spinner(""):
            try:
                result: Dict[str, Any] = config.rag_engine1.answer_with_citations(
                    question=question,
                    screenshot_path=tmp_img,
                    k=k,
                    min_conf=min_conf,
                    model="gpt-4o-mini",
                    debug=True,
                )
            except Exception:
                st.error("OpenAI call failed. Check OPENAI_API_KEY.")
                st.session_state.submitted = False
                st.stop()
                return

        contexts = result.get("contexts", [])
        confidence_val = 0.0
        for ctx in contexts:
            try:
                confidence_val = max(confidence_val, float(ctx.get("score", 0.0)))
            except Exception:
                continue

        record = {
            "qid": str(uuid.uuid4()),
            "question": question,
            "answer": result.get("answer", ""),
            "debug": result.get("page_scores", {}),
            "contexts": contexts,
            "confidence": confidence_val,
            "img_path": tmp_img,
            "feedback": None,
        }
        st.session_state.chat_history.append(record)
        append_chat_log({**record, "img_path": bool(tmp_img)})

        st.session_state.pending_tmp_img = None
        st.session_state.uploader_key += 1
        st.session_state.submitted = False
        st.session_state.rag_query_count += 1
        st.rerun()
