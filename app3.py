# app3.py — SmartTA (Unified): RAG chat + YouTube + Dashboards
# --------------------------------------------------------------
# Changes you asked for:
# 1) Remove confidence line in chat and use a real chat-bubble UI
# 2) Show “SmartTA is thinking…” typing bubble while answering
# 3) Add “🗑️ Remove attachment” to delete the currently attached screenshot
# 4) PDF export now includes AUB logo from SmartTA_RAG/UI/aub_logo.png
# 5) Instructor dashboard reads logs from ./data and SmartTA_RAG/UI/data

import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys, json, uuid, tempfile, shutil
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

import streamlit as st
import pandas as pd
import altair as alt

# ========== Paths & Imports ===================================================
CURRENT_DIR      = Path(__file__).resolve().parent
SMARTTA_RAG_DIR  = CURRENT_DIR / "SmartTA_RAG"
SMARTTA_EXT_DIR  = CURRENT_DIR / "SmartTA_Extensions"

for p in (SMARTTA_RAG_DIR, SMARTTA_EXT_DIR):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Lightweight .env loader (root + SmartTA_RAG)
def _load_env_file(path: Path):
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=str(path), override=False)
        return
    except Exception:
        pass
    try:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass

_load_env_file(CURRENT_DIR / ".env")
_load_env_file(SMARTTA_RAG_DIR / ".env")

# RAG engine
try:
    import rag_engine1  # from SmartTA_RAG
except Exception as e:
    st.error(f"Failed to import rag_engine1: {e}")
    st.stop()

# YouTube extensions
try:
    from SmartTA_Extensions.frontend.tabs.student    import render_student_tab
    from SmartTA_Extensions.frontend.tabs.professor  import render_professor_tab
    from SmartTA_Extensions.utils.helpers            import load_youtube_lectures, get_youtube_lectures_categorized
    from SmartTA_Extensions.backend.indexing         import reindex_if_needed
except Exception as e:
    render_student_tab = None
    render_professor_tab = None
    load_youtube_lectures = None
    get_youtube_lectures_categorized = None
    reindex_if_needed = None
    st.sidebar.warning(f"YouTube extensions not fully available: {e}")

try:
    import psutil
except Exception:
    psutil = None

# ========== App Config ========================================================
st.set_page_config(page_title="🧠 SmartTA Unified Assistant", layout="wide")

# Chat logs: keep local ./data, but also read SmartTA_RAG/UI/data if present
APP_DATA_DIR   = (CURRENT_DIR / "data")
APP_DATA_DIR.mkdir(exist_ok=True)
LOCAL_CHAT_LOG = APP_DATA_DIR / "chat_logs.ndjson"

RAG_UI_DIR     = SMARTTA_RAG_DIR / "UI" / "data"
RAG_UI_DIR.mkdir(parents=True, exist_ok=True)
RAG_UI_CHAT_LOG = RAG_UI_DIR / "chat_logs.ndjson"  # legacy location

# YouTube data (canonical location under extensions)
EXT_DATA_DIR   = SMARTTA_EXT_DIR / "data"
EXT_DATA_DIR.mkdir(exist_ok=True)
EXT_META_SEG   = EXT_DATA_DIR / "segments_metadata.json"
EXT_META_PATH  = EXT_DATA_DIR / "metadata.json"
EXT_INDEX_PATH = EXT_DATA_DIR / "course.index"
EXT_LOG_PATH   = EXT_DATA_DIR / "queries_log.json"
EXT_UPLOADS    = EXT_DATA_DIR / "uploaded_files"
EXT_TRANS      = EXT_DATA_DIR / "transcripts"
EXT_MODELS     = EXT_DATA_DIR / "models"
EXT_LECTURES   = EXT_DATA_DIR / "lectures"
for d in (EXT_UPLOADS, EXT_TRANS, EXT_MODELS, EXT_LECTURES):
    d.mkdir(exist_ok=True)

# ========== Styles (Chat look + badges + typing bubble) ======================
st.markdown("""
<style>
:root {
  --bg:#0D0E12; --text:#F3F3F3; --muted:#9aa2b1; --card:#171923;
  --blue:#2e7cf6; --cyan:#00C6FF; --green:#22c55e; --amber:#f59e0b; --red:#ef4444; --purple:#6d28d9;
}
html, body, .stApp { background:var(--bg); color:var(--text); font-family:Inter, system-ui; }
.smartta-title{
  text-align:center; font-size:2.4rem; font-weight:900; margin:20px 0;
  background:linear-gradient(90deg,var(--cyan),var(--blue));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 0 20px rgba(0,198,255,.3);
}
/* Chat card + rows */
.chat-card{ background:linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,.01));
  border-radius:16px; padding:18px; box-shadow:0 8px 30px rgba(2,6,23,.6);}
.chat-row{ display:flex; gap:12px; align-items:flex-start; margin:10px 0; }
.chat-row.user{ flex-direction: row-reverse; }
.avatar{ width:36px; height:36px; border-radius:999px; display:flex; align-items:center; justify-content:center; font-weight:700;}
.avatar.user{ background:linear-gradient(135deg,#0066FF,#00BFFF); color:#fff; }
.avatar.assistant{ background:linear-gradient(135deg,#1f2937,#111827); color:#9fb0c8; border:1px solid rgba(255,255,255,.03); }
.bubble{ padding:12px 14px; border-radius:12px; max-width:78%; font-size:.98rem; line-height:1.45; border:1px solid rgba(255,255,255,.04);}
.bubble.user{ background:linear-gradient(135deg,#0066FF,#00BFFF); color:#fff; border-bottom-left-radius:4px;}
.bubble.assistant{ background:rgba(255,255,255,.03); color:var(--text); }
.meta{ font-size:.78rem; color:var(--muted); margin-top:6px; }
.model-badge{ display:inline-block; font-weight:700; font-size:.74rem; padding:2px 8px; border-radius:999px; margin-right:6px;}
.model-text{ background:#1d4ed8; color:#fff;}
.model-img{ background:#15803d; color:#fff;}
.model-clip{ background:#b45309; color:#fff;}
.model-ctx{ background:#6d28d9; color:#fff;}
/* Typing bubble */
.typing { display:inline-block; padding:10px 14px; border-radius:12px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.06);}
.dot{ height:6px; width:6px; background:#9aa2b1; border-radius:999px; display:inline-block; margin:0 2px;
  animation: blink 1.3s infinite;}
.dot:nth-child(2){ animation-delay:.2s;} .dot:nth-child(3){ animation-delay:.4s;}
@keyframes blink { 0%{opacity:.25} 50%{opacity:1} 100%{opacity:.25} }
.metric-card{ background:linear-gradient(135deg,#1e293b,#0f172a); border-radius:14px; padding:18px; text-align:center; box-shadow:0 0 14px rgba(0,0,0,.3); }
.metric-card h2{ margin:0; font-size:1.8rem; color:#00C6FF;}
.metric-card p{ margin:4px 0 0; color:var(--muted);}
.login-box{ background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.15); padding:25px; border-radius:12px; width:350px; margin:auto; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ========== Optional libs =====================================================
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception:
    _HAS_SKLEARN = False

# ========== Helpers ===========================================================
def _aub_logo() -> str | None:
    # Prefer SmartTA_RAG/UI/aub_logo.png (your path), fallback to SmartTA_RAG/aub_logo.png
    p1 = SMARTTA_RAG_DIR / "UI" / "aub_logo.png"
    p2 = SMARTTA_RAG_DIR / "aub_logo.png"
    if p1.exists(): return str(p1)
    if p2.exists(): return str(p2)
    return None

def _tuple_key_dict_to_str_key(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        out[f"{k[0]}:{k[1]}" if isinstance(k, (tuple, list)) else str(k)] = v
    return out

def _sanitize_debug(debug: dict) -> dict:
    out = {}
    if not debug: return out
    if "text_openai" in debug: out["text_openai"] = {str(k): float(v) for k, v in debug["text_openai"].items()}
    if "img2img"    in debug: out["img2img"]    = _tuple_key_dict_to_str_key(debug["img2img"])
    if "text2img"   in debug: out["text2img"]   = _tuple_key_dict_to_str_key(debug["text2img"])
    return out

def _write_log_line(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

def _append_chat_log(entry: dict):
    e = dict(entry)
    e.setdefault("qid", str(uuid.uuid4()))
    e["timestamp"] = datetime.now(timezone.utc).isoformat()
    e["debug"] = _sanitize_debug(e.get("debug", {}))
    e["contexts"] = [{"pdf": c.get("pdf"), "page": c.get("page"), "score": float(c.get("score", 0.0))}
                     for c in e.get("contexts", []) or []]
    if e.get("img_path"): e["img_path"] = True
    e["feedback"] = e.get("feedback")
    # Write to the local canonical file and also mirror to legacy RAG/UI one if you want
    _write_log_line(LOCAL_CHAT_LOG, e)
    try:
        _write_log_line(RAG_UI_CHAT_LOG, e)
    except Exception:
        pass

def _read_logs_df() -> pd.DataFrame:
    rows = []
    for path in (LOCAL_CHAT_LOG, RAG_UI_CHAT_LOG):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except Exception: pass
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "qid" not in df.columns:
        df["qid"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["date"] = df["ts"].dt.date
    def _top(L):
        try: return max([float(c.get("score",0.0)) for c in (L or [])] + [0.0])
        except Exception: return 0.0
    df["top_score"] = df.get("contexts", []).apply(_top)
    if "feedback" not in df.columns: df["feedback"] = None
    df["feedback"] = df["feedback"].fillna("none")
    # Deduplicate on qid (keep latest)
    df = df.sort_values("ts").drop_duplicates(subset=["qid"], keep="last")
    return df

def _sanitize_q(q: str) -> str:
    if not isinstance(q, str): return ""
    import re as _re
    q = q.replace("\r", " ")
    q = "".join(ch for ch in q if ch.isprintable() or ch in "\n\t ")
    return _re.sub(r"\s+", " ", q).strip()[:1000]

@contextmanager
def ext_cwd():
    old = os.getcwd()
    os.chdir(str(SMARTTA_EXT_DIR))
    try:
        yield
    finally:
        os.chdir(old)

# ========== Init flags + session =============================================
enable_rag      = os.getenv("SMARTTA_ENABLE_RAG", "1") not in ("0","false","False")
enable_youtube  = os.getenv("SMARTTA_ENABLE_YOUTUBE", "1") not in ("0","false","False")

if "engine_loaded" not in st.session_state and enable_rag:
    with st.spinner("Loading multimodal RAG indexes…"):
        rag_engine1.load_indexes()
        st.session_state.engine_loaded = True
    if psutil:
        try:
            mem_mb = psutil.Process().memory_info().rss/1024/1024
            st.sidebar.caption(f"Memory: {mem_mb:.0f} MB")
        except Exception:
            pass

# session keys
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploader_key", 0)         # used to reset uploader
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("rag_query_count", 0)
st.session_state.setdefault("ui_scale", 100)
st.session_state.setdefault("pending_tmp_img", None)   # path for the *current* attachment
st.session_state.setdefault("dl_txt", None)
st.session_state.setdefault("dl_pdf", None)
st.session_state.setdefault("dl_txt_key", 0)
st.session_state.setdefault("dl_pdf_key", 0)

# ========== Mode selector =====================================================
st.markdown("<div class='smartta-title'>🧠 SmartTA — Multimodal RAG Assistant</div>", unsafe_allow_html=True)

labels, modes = [], []
if enable_rag or enable_youtube: labels.append("🎓 Student Assistant"); modes.append("Student Assistant")
if enable_rag:                  labels.append("📊 Instructor Dashboard"); modes.append("Instructor Dashboard")
if enable_youtube:              labels.append("🎯 Professor Dashboard"); modes.append("Professor Dashboard")
if not modes:                   labels, modes = ["🎓 Student Assistant"], ["Student Assistant"]

st.markdown("<div style='border:1px solid rgba(0,198,255,.2); border-radius:20px; padding:20px; max-width:900px; margin:16px auto;'>", unsafe_allow_html=True)
_, c, _ = st.columns([1,1,1])
with c: picked = st.radio("", labels, horizontal=True, label_visibility="collapsed")
mode = next((m for m, l in zip(modes, labels) if l == picked), modes[0])
st.markdown("</div>", unsafe_allow_html=True)

k = 5
min_conf = 0.25
RAG_QUERY_LIMIT = int(os.getenv("SMARTTA_RAG_QUERY_LIMIT","50"))

if enable_rag and not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Missing OPENAI_API_KEY. Put it in .env or your environment.")

# ========== Auth (for dashboards) ============================================
if mode in ("Instructor Dashboard","Professor Dashboard") and not st.session_state.authenticated:
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Instructor Login")
    u = st.text_input("Username"); p = st.text_input("Password", type="password")
    colA, colB = st.columns([1,1])
    if colA.button("Login"):
        if u == "eece" and p == "690":
            st.session_state.authenticated = True; st.success("✅ Login successful!"); st.rerun()
        else:
            st.error("❌ Invalid credentials.")
    if colB.button("Cancel"): st.info("Login canceled.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ========== Student Assistant =================================================
if mode == "Student Assistant":

    # ----- RAG Chat -----
    if enable_rag:
        st.markdown("### Display")
        st.session_state.ui_scale = st.slider("🔎 UI scale", 90, 130, st.session_state.ui_scale, 1)
        st.markdown(f"<style>.stApp {{ zoom: {st.session_state.ui_scale/100}; }}</style>", unsafe_allow_html=True)

        def _mark_submit(): st.session_state.submitted = True

        st.divider(); st.subheader("Conversation")
        st.markdown("<div class='chat-card'>", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            # user bubble
            st.markdown(
                f"<div class='chat-row user'><div class='avatar user'>You</div>"
                f"<div><div class='bubble user'>{chat['question']}</div></div></div>",
                unsafe_allow_html=True
            )
            if isinstance(chat.get("img_path"), str) and Path(chat["img_path"]).exists():
                st.image(chat["img_path"], caption="Attached image", use_container_width=True)

            # assistant bubble (no confidence shown)
            st.markdown(
                f"<div class='chat-row'><div class='avatar assistant'>TA</div>"
                f"<div><div class='bubble assistant'>{chat['answer']}</div></div></div>",
                unsafe_allow_html=True
            )

            # Retrieval diagnostics (with colored model badges)
            with st.expander("Retrieval Diagnostics"):
                tabs = st.tabs(["openai→text", "img→img", "text↔️img", "Contexts"])
                dbg = chat.get("debug", {}) or {}

                with tabs[0]:
                    st.markdown("<span class='model-badge model-text'>text-embedding-3</span>", unsafe_allow_html=True)
                    if dbg.get("text_openai"):
                        for k2, sc in list(dbg["text_openai"].items())[:5]:
                            try:
                                idx = int(k2)
                                if idx < len(rag_engine1.chunk_meta):
                                    meta = rag_engine1.chunk_meta[idx]
                                    st.markdown(f"• *{meta['pdf']}* p.{meta['page']} → `{float(sc):.3f}`")
                            except Exception:
                                pass
                    else: st.caption("No text matches.")

                with tabs[1]:
                    st.markdown("<span class='model-badge model-img'>CLIP (img→img)</span>", unsafe_allow_html=True)
                    if dbg.get("img2img"):
                        import re as _re
                        for key2, sc in sorted(dbg["img2img"].items(), key=lambda x: -x[1])[:5]:
                            if isinstance(key2, (tuple, list)):
                                pdf, page = key2[0], key2[1] if len(key2) > 1 else "?"
                            else:
                                parts = _re.split(r"[:,()'\\s]+", str(key2))
                                pdf = next((p for p in parts if p.endswith(".pdf")), "?")
                                page = next((p for p in parts if p.isdigit()), "?")
                            st.markdown(f"• *{pdf}* p.{page} → `{float(sc):.3f}`")
                    else: st.caption("No image→image matches.")

                with tabs[2]:
                    st.markdown("<span class='model-badge model-clip'>CLIP (text↔️img)</span>", unsafe_allow_html=True)
                    if dbg.get("text2img"):
                        import re as _re
                        for key2, sc in sorted(dbg["text2img"].items(), key=lambda x: -x[1])[:5]:
                            if isinstance(key2, (tuple, list)):
                                pdf, page = key2[0], key2[1] if len(key2) > 1 else "?"
                            else:
                                parts = _re.split(r"[:,()'\\s]+", str(key2))
                                pdf = next((p for p in parts if p.endswith(".pdf")), "?")
                                page = next((p for p in parts if p.isdigit()), "?")
                            st.markdown(f"• *{pdf}* p.{page} → `{float(sc):.3f}`")
                    else: st.caption("No text↔image matches.")

                with tabs[3]:
                    st.markdown("<span class='model-badge model-ctx'>Context Chunks</span>", unsafe_allow_html=True)
                    if chat.get("contexts"):
                        for ctx in chat["contexts"][:5]:
                            st.markdown(f"• *{ctx['pdf']}* p.{ctx['page']} (score=`{float(ctx['score']):.3f}`)")
                    else: st.caption("No chunks included.")
        # typing bubble while submitted
        if st.session_state.submitted and (st.session_state.get("chat_input","").strip()):
            st.markdown(
                "<div class='chat-row'><div class='avatar assistant'>TA</div>"
                "<div class='typing'>SmartTA is thinking… <span class='dot'></span><span class='dot'></span><span class='dot'></span></div></div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Input footer
        st.markdown("---")
        st.markdown("### Ask a question:")
        q = st.text_input("", placeholder="What's your question?", key="chat_input", on_change=_mark_submit)

        # Attachment
        img = st.file_uploader("Optional: Upload slide image", type=["png","jpg","jpeg"], key=f"uploader_{st.session_state.uploader_key}")
        # If user picked a file, stash its bytes to a temp file so we can preview and later delete if needed
        if img:
            # If we already had a pending temp, clean it
            if st.session_state.pending_tmp_img and Path(st.session_state.pending_tmp_img).exists():
                try: Path(st.session_state.pending_tmp_img).unlink(missing_ok=True)
                except Exception: pass
            suffix = Path(img.name).suffix or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(img.read()); tmp.flush()
            st.session_state.pending_tmp_img = tmp.name

        # Preview + Remove button
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

        c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1])
        with c1: ask_clicked = st.button("Ask SmartTA 🚀")
        with c2: clear = st.button("Clear Chat 🧹")
        with c3: export_txt = st.button("Export Chat (.txt)")
        with c4: export_pdf = st.button("Export PDF 📑", disabled=not REPORTLAB_OK)
        with c5: pass

        # Export TXT
        if export_txt and st.session_state.chat_history:
            lines = []
            for c in st.session_state.chat_history:
                lines.append(f"You: {c['question']}")
                if c.get("img_path"): lines.append("[image attached]")
                lines.append(f"SmartTA: {c['answer']}\n")
            st.session_state.dl_txt = ("\n".join(lines)).encode("utf-8")

        # Export PDF (with AUB logo path fix)
        if export_pdf and st.session_state.chat_history and REPORTLAB_OK:
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=0.7*inch, rightMargin=0.7*inch,
                                    topMargin=0.6*inch, bottomMargin=0.6*inch)
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=26, leading=30, spaceAfter=12)
            subtitle_style = ParagraphStyle('SubCenter', parent=styles['BodyText'], alignment=TA_CENTER, fontSize=14, leading=18, spaceAfter=8)
            meta_style = ParagraphStyle('MetaCenter', parent=styles['Italic'], alignment=TA_CENTER, fontSize=10, leading=12, spaceAfter=12)
            qa_q_style = ParagraphStyle('QAQ', parent=styles['BodyText'], fontSize=11, leading=14, spaceAfter=4)
            qa_a_style = ParagraphStyle('QAA', parent=styles['BodyText'], fontSize=11, leading=16, spaceAfter=10)

            story = []
            lg = _aub_logo()
            if lg and Path(lg).exists():
                story.append(RLImage(lg, width=2.6*inch, height=2.6*inch)); story.append(Spacer(1, 0.12*inch))
            story.append(Paragraph("<b>American University of Beirut – EECE Department</b>", title_style))
            story.append(Paragraph("EECE 490/690 — Intro to Machine Learning", subtitle_style))
            story.append(Paragraph(f"<i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", meta_style))
            story.append(Spacer(1, 0.25*inch))
            for c in st.session_state.chat_history:
                story.append(Paragraph(f"<b>Question:</b> {c.get('question','')}", qa_q_style))
                if isinstance(c.get("img_path"), str) and Path(c["img_path"]).exists():
                    story.append(RLImage(c["img_path"], width=5.8*inch, height=3.6*inch)); story.append(Spacer(1,0.08*inch))
                story.append(Paragraph(f"<b>Answer:</b> {c.get('answer','')}", qa_a_style))
                story.append(Spacer(1,0.12*inch))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e6e6e6"))); story.append(Spacer(1,0.12*inch))
            doc.build(story); buf.seek(0); st.session_state.dl_pdf = buf.getvalue()

        # Download buttons inline
        dl_cols = st.columns(2)
        if st.session_state.dl_txt:
            if dl_cols[0].download_button("⬇️ Download Chat (.txt)", data=st.session_state.dl_txt,
                                          file_name="SmartTA_chat.txt", mime="text/plain",
                                          use_container_width=True, key=f"dl_txt_inline_{st.session_state.dl_txt_key}"):
                st.session_state.dl_txt = None; st.session_state.dl_txt_key += 1; st.rerun()
        if st.session_state.dl_pdf:
            if dl_cols[1].download_button("⬇️ Download PDF", data=st.session_state.dl_pdf,
                                          file_name="SmartTA_Session_Report.pdf", mime="application/pdf",
                                          use_container_width=True, key=f"dl_pdf_inline_{st.session_state.dl_pdf_key}"):
                st.session_state.dl_pdf = None; st.session_state.dl_pdf_key += 1; st.rerun()

        # Clear chat
        if clear:
            # also remove pending temp attachment if any
            if st.session_state.pending_tmp_img and Path(st.session_state.pending_tmp_img).exists():
                try: Path(st.session_state.pending_tmp_img).unlink(missing_ok=True)
                except Exception: pass
            st.session_state.pending_tmp_img = None
            st.session_state.chat_history = []
            st.session_state.uploader_key += 1
            st.session_state.submitted = False
            st.session_state.dl_txt = None
            st.session_state.dl_pdf = None
            st.rerun()

        # Submit question
        q = _sanitize_q(q)
        if (ask_clicked or st.session_state.submitted) and q.strip():
            if st.session_state.rag_query_count >= RAG_QUERY_LIMIT:
                st.error("Query limit reached for this session. Reload to continue.")
                st.session_state.submitted = False
                st.stop()

            # if a pending temp image exists, use it. Then clear the pending pointer.
            tmp_img = st.session_state.pending_tmp_img if st.session_state.pending_tmp_img and Path(st.session_state.pending_tmp_img).exists() else None

            with st.spinner("SmartTA is thinking…"):
                try:
                    out = rag_engine1.answer_with_citations(
                        question=q,
                        screenshot_path=tmp_img,
                        k=k, min_conf=min_conf,
                        model="gpt-4o-mini",
                        debug=True
                    )
                except Exception:
                    st.error("OpenAI call failed. Check OPENAI_API_KEY.")
                    st.session_state.submitted = False
                    st.stop()

            rec = {
                "qid": str(uuid.uuid4()),
                "question": q,
                "answer": out.get("answer",""),
                "debug": out.get("page_scores", {}),
                "contexts": out.get("contexts", []),
                "img_path": tmp_img,
                "feedback": None,
            }
            st.session_state.chat_history.append(rec)
            _append_chat_log({**rec, "img_path": bool(tmp_img)})

            # Clear pending state & reset uploader
            st.session_state.pending_tmp_img = None
            st.session_state.uploader_key += 1
            st.session_state.submitted = False
            st.session_state.rag_query_count += 1
            st.rerun()

    # ----- YouTube Student Library -----
    if enable_youtube and all([render_student_tab, load_youtube_lectures, get_youtube_lectures_categorized, reindex_if_needed]):
        st.markdown("---")
        st.markdown("## 📚 Lecture Library & Search (YouTube)")
        st.caption("Search transcribed lectures with semantic search")

        with ext_cwd():
            # be lenient if segments file has a wrong shape
            if EXT_META_SEG.exists():
                try:
                    seg = json.loads(EXT_META_SEG.read_text(encoding="utf-8"))
                    all_segments = seg if isinstance(seg, dict) else {}
                except Exception:
                    all_segments = {}
            else:
                all_segments = {}

            with st.spinner("Initializing YouTube lecture search…"):
                metadata, index, new_indexes = reindex_if_needed(all_segments, load_youtube_lectures)
                if new_indexes > 0:
                    st.success(f"✅ Indexed {new_indexes} new lecture(s)")

            lecture_categories = get_youtube_lectures_categorized()
            yt = load_youtube_lectures()
            all_lectures = [info.get("title") for info in yt.get("lectures", {}).values()]

            render_student_tab(
                DATA_DIR=str(EXT_DATA_DIR),
                META_SEGMENTS=str(EXT_META_SEG),
                META_PATH=str(EXT_META_PATH),
                INDEX_PATH=str(EXT_INDEX_PATH),
                LOG_PATH=str(EXT_LOG_PATH),
                UPLOADS_DIR=str(EXT_UPLOADS),
                all_lectures=all_lectures,
                lecture_categories=lecture_categories,
                metadata=metadata,
                index=index,
            )
    elif enable_youtube:
        st.info("YouTube library unavailable (extension import failed).")

# ========== Instructor Dashboard =============================================
elif mode == "Instructor Dashboard":
    st.markdown("## 🧑‍🏫 Instructor Dashboard (RAG)")
    if st.button("🚪 Logout"): st.session_state.authenticated=False; st.rerun()

    df = _read_logs_df()
    if df.empty:
        st.info("No logs yet — ask some questions first.")
        st.stop()

    st.markdown("### 📅 Filters")
    df = df[df["date"].notna()]
    if df.empty:
        st.info("No valid logs with dates yet.")
        st.stop()

    min_d, max_d = df["date"].min(), df["date"].max()
    d_from, d_to = st.date_input("Select Date Range", value=(min_d, max_d))
    conf_filter = st.slider("Min confidence filter", 0.0, 1.0, 0.0, 0.01)
    fb_filter = st.selectbox("Feedback filter", ["all","helpful only","unclear only"])

    df = df[(df["date"]>=d_from) & (df["date"]<=d_to) & (df["top_score"]>=conf_filter)]
    if fb_filter=="helpful only": df = df[df["feedback"]=="helpful"]
    elif fb_filter=="unclear only": df = df[df["feedback"]=="unclear"]
    if df.empty:
        st.warning("No data after filters.")
        st.stop()

    total_q = int(len(df)); avg_conf = float(df["top_score"].mean())
    active_days = int(df["date"].nunique())
    helpful_cnt = int((df["feedback"]=="helpful").sum())
    unclear_cnt = int((df["feedback"]=="unclear").sum())
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f"<div class='metric-card'><h2>{total_q}</h2><p>Total Q&A Rows</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>{avg_conf:.3f}</h2><p>Average Confidence</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h2>{active_days}</h2><p>Active Days</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h2>{helpful_cnt}</h2><p>👍 Helpful</p></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='metric-card'><h2>{unclear_cnt}</h2><p>👎 Unclear</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔝 Top 10 Topics")
    qs = df["question"].dropna().astype(str)
    if not qs.empty and _HAS_SKLEARN:
        try:
            vec = CountVectorizer(ngram_range=(2,3), stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
            X = vec.fit_transform(qs)
            freqs = X.toarray().sum(axis=0)
            vocab = vec.get_feature_names_out()
            topic_df = pd.DataFrame({"Phrase": vocab, "Count": freqs}).sort_values("Count", ascending=False).head(10)
            st.altair_chart(alt.Chart(topic_df).mark_bar(color="#2e7cf6")
                            .encode(x="Count:Q", y=alt.Y("Phrase:N", sort='-x'), tooltip=["Phrase","Count"])
                            .properties(height=320), use_container_width=True)
        except Exception:
            st.info("Not enough signal to compute topics yet.")
    else:
        st.caption("No questions to analyze or scikit-learn not installed.")

    st.markdown("---"); st.markdown("### 🕒 Engagement Over Time")
    daily = df.groupby("date").size().reset_index(name="Questions")
    st.altair_chart(alt.Chart(daily).mark_area(color="#20c997")
                    .encode(x="date:T", y="Questions:Q", tooltip=["date:T","Questions:Q"])
                    .properties(height=300), use_container_width=True)

# ========== Professor Dashboard ==============================================
elif mode == "Professor Dashboard":
    if render_professor_tab and get_youtube_lectures_categorized:
        with ext_cwd():
            lecture_categories = get_youtube_lectures_categorized()
            render_professor_tab(
                DATA_DIR=str(EXT_DATA_DIR),
                LECTURES_DIR=str(EXT_LECTURES),
                TRANSCRIPT_DIR=str(EXT_TRANS),
                META_SEGMENTS=str(EXT_META_SEG),
                META_PATH=str(EXT_META_PATH),
                INDEX_PATH=str(EXT_INDEX_PATH),
                LOG_PATH=str(EXT_LOG_PATH),
                UPLOADS_DIR=str(EXT_UPLOADS),
                lecture_categories=lecture_categories,
            )
    else:
        st.info("Professor dashboard not available (extension import failed).")

else:
    st.error("Unknown mode selected.")