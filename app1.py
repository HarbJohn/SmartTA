# app.py — SmartTA Final (Crash fix for date filter + robust logs + same UI/features)

import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import re
import json
import uuid
import tempfile
from io import BytesIO
from datetime import datetime, date as date_cls
from pathlib import Path
from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt

# ---------- PDF Export ----------
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

# ---------- n-grams ----------
_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception:
    _HAS_SKLEARN = False

# ---------- Paths & RAG ----------
CURRENT_DIR = Path(__file__).resolve().parent
SMARTTA_RAG_ROOT = CURRENT_DIR / "SmartTA_RAG"
sys.path.insert(0, str(SMARTTA_RAG_ROOT))
import rag_engine1  # noqa

# ---------- SmartTA_Extensions (YouTube) ----------
SMARTTA_EXTENSIONS = CURRENT_DIR / "SmartTA_Extensions"
if str(SMARTTA_EXTENSIONS) not in sys.path:
    sys.path.insert(0, str(SMARTTA_EXTENSIONS))

@contextmanager
def ext_cwd():
    old = os.getcwd()
    os.chdir(str(SMARTTA_EXTENSIONS))
    try:
        yield
    finally:
        os.chdir(old)

# ---------- Logs (RAG) ----------
DATA_DIR = SMARTTA_RAG_ROOT / "UI" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = DATA_DIR / "chat_logs.ndjson"

# ---------- Page ----------
st.set_page_config(page_title="🧠 SmartTA — Multimodal RAG Assistant", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
:root { --bg:#0D0E12; --text:#F3F3F3; --blue:#2e7cf6; --cyan:#00C6FF; --green:#22c55e; --amber:#f59e0b; --red:#ef4444; --purple:#6d28d9; --card:#171923; --muted:#9aa2b1; }
html, body, .stApp { background:var(--bg); color:var(--text); font-family:'Inter', system-ui; }
.smartta-title { text-align:center; font-size:2.4rem; font-weight:900; margin:20px 0; background:linear-gradient(90deg,var(--cyan),var(--blue)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-shadow:0 0 20px rgba(0,198,255,0.3); }
.user-bubble { background:linear-gradient(135deg,#0066FF,#00BFFF); color:#fff; padding:12px 14px; border-radius:14px; margin:8px 0; max-width:75%; }
.assistant-bubble { background:rgba(255,255,255,.07); color:var(--text); padding:12px 14px; border-radius:14px; margin:8px 0; max-width:85%; border-left:5px solid transparent; }
.bubble-high{border-left-color:var(--green)} .bubble-mid{border-left-color:var(--amber)} .bubble-low{border-left-color:var(--red)}
.metric-card{background:linear-gradient(135deg,#1e293b,#0f172a); border-radius:14px; padding:18px; text-align:center; box-shadow:0 0 14px rgba(0,0,0,.3);}
.metric-card h2{margin:0;font-size:1.8rem;color:#00C6FF} .metric-card p{margin:4px 0 0 0;color:var(--muted)}
.chat-row{display:flex;gap:12px;align-items:flex-start;margin:8px 0;}
.avatar{width:36px;height:36px;border-radius:999px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;}
.avatar.user{background:linear-gradient(135deg,#0066FF,#00BFFF);color:#fff;}
.avatar.assistant{background:linear-gradient(135deg,#1f2937,#111827);color:#9fb0c8;border:1px solid rgba(255,255,255,0.03);}
.bubble{padding:12px 14px;border-radius:12px;max-width:78%;font-size:.98rem;line-height:1.4;}
.bubble.user{background:linear-gradient(135deg,#0066FF,#00BFFF);color:#fff;border-bottom-right-radius:4px;}
.bubble.assistant{background:rgba(255,255,255,.03);color:var(--text);border:1px solid rgba(255,255,255,.04);}
.meta{font-size:.78rem;color:var(--muted);margin-top:6px;}
.chat-card{background:linear-gradient(180deg,rgba(255,255,255,.02),rgba(255,255,255,.01));border-radius:16px;padding:18px;box-shadow:0 8px 30px rgba(2,6,23,.6);}
.chat-row.user{flex-direction:row-reverse;} .chat-row.user .bubble.user{border-bottom-left-radius:4px;border-bottom-right-radius:12px;}
div[role="radiogroup"]{display:flex!important;gap:12px!important;justify-content:center!important;}
.chat-input-footer{position:relative;margin-top:24px;padding:16px 0;background:transparent;border:none;box-shadow:none;z-index:10;}
.model-badge{display:inline-block;font-weight:700;font-size:.8rem;padding:3px 8px;border-radius:999px;margin-bottom:4px;}
.model-text{background:#1d4ed8;color:#fff;} .model-img{background:#15803d;color:#fff;} .model-clip{background:#b45309;color:#fff;} .model-ctx{background:#6d28d9;color:#fff;}
.login-box{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.15);padding:25px;border-radius:12px;width:350px;margin:auto;text-align:center;box-shadow:0 0 20px rgba(0,0,0,.4);}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def find_aub_logo() -> str | None:
    candidates = [
        CURRENT_DIR / "smartta_unified" / "aub_logo.png",
        SMARTTA_RAG_ROOT / "UI" / "aub_logo.png",
        SMARTTA_RAG_ROOT / "aub_logo.png",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None

def _tuple_key_dict_to_str_key(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(k, (tuple, list)): out[f"{k[0]}:{k[1]}"] = v
        else: out[str(k)] = v
    return out

def sanitize_debug(debug: dict) -> dict:
    out = {}
    if not debug: return out
    if "text_openai" in debug: out["text_openai"] = {str(k): float(v) for k, v in debug["text_openai"].items()}
    if "img2img" in debug: out["img2img"] = _tuple_key_dict_to_str_key(debug["img2img"])
    if "text2img" in debug: out["text2img"] = _tuple_key_dict_to_str_key(debug["text2img"])
    return out

def append_log(entry: dict):
    e = dict(entry)
    e.setdefault("qid", str(uuid.uuid4()))
    e["timestamp"] = datetime.utcnow().isoformat()
    e["debug"] = sanitize_debug(e.get("debug", {}))
    e["contexts"] = [
        {"pdf": c.get("pdf"), "page": c.get("page"), "score": float(c.get("score", 0.0))}
        for c in (e.get("contexts") or [])
    ]
    if e.get("img_path"): e["img_path"] = True
    e["feedback"] = e.get("feedback", None)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

def update_feedback(qid: str, feedback_value: str):
    if not qid: return
    rows = []
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s: continue
            try: rows.append(json.loads(s))
            except Exception: pass
    for r in rows:
        if r.get("qid") == qid:
            r["feedback"] = feedback_value
            break
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_logs_df() -> pd.DataFrame:
    """
    Robust loader for chat_logs.ndjson
    FIX: drop rows with invalid timestamps BEFORE computing 'date',
    so 'date' never mixes datetime.date with NaN(float).
    Also guard when 'contexts' column is missing.
    """
    if not LOG_PATH.exists():
        return pd.DataFrame()

    rows = []
    for l in LOG_PATH.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l: continue
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # stable qids
    if "qid" not in df.columns:
        df["qid"] = [str(uuid.uuid4()) for _ in range(len(df))]
    else:
        df["qid"] = df["qid"].astype(str).replace({"": None})
        mask = df["qid"].isna()
        if mask.any():
            df.loc[mask, "qid"] = [str(uuid.uuid4()) for _ in range(int(mask.sum()))]

    # timestamps -> drop invalid
    df["ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df = df[df["ts"].notna()].copy()                # <<< important
    if df.empty:
        return df

    # pure date objects
    df["date"] = df["ts"].dt.date

    # top score safe
    def _top(L):
        try: return max([float(c.get("score", 0.0)) for c in (L or [])] + [0.0])
        except Exception: return 0.0

    if "contexts" in df.columns:
        df["top_score"] = df["contexts"].apply(_top)
    else:
        df["top_score"] = 0.0

    if "feedback" not in df.columns:
        df["feedback"] = None
    df["feedback"] = df["feedback"].fillna("none")

    # latest per qid
    df = df.sort_values("ts").drop_duplicates(subset=["qid"], keep="last").reset_index(drop=True)
    return df

def _confidence_to_badge(conf: float) -> str:
    if conf < 0.4: return "bubble-low"
    if conf < 0.75: return "bubble-mid"
    return "bubble-high"

def _canon_q(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ALIAS_MAP = { }
def _alias_q(s: str) -> str:
    k = _canon_q(s)
    return ALIAS_MAP.get(k, k)

# ---------- INIT ----------
if "engine_loaded" not in st.session_state:
    with st.spinner("Loading RAG indexes..."):
        rag_engine1.load_indexes()
        st.session_state.engine_loaded = True

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("dl_txt", None)
st.session_state.setdefault("dl_pdf", None)
st.session_state.setdefault("dl_txt_key", 0)
st.session_state.setdefault("dl_pdf_key", 0)
st.session_state.setdefault("ui_scale", 100)

# ---------- Mode Selector ----------
st.markdown("<div class='smartta-title'>🧠 SmartTA — Multimodal RAG Assistant</div>", unsafe_allow_html=True)
st.markdown("<div style='border:1px solid rgba(0,198,255,0.2); border-radius:20px; padding:20px; max-width:900px; margin:16px auto;'>", unsafe_allow_html=True)
_, c, _ = st.columns([1,1,1])
with c:
    mode_choice = st.radio("", ["🎓 Student Chat", "📊 Instructor Dashboard"], horizontal=True, label_visibility="collapsed")
mode = "Student Chat" if "Student" in mode_choice else "Instructor Dashboard"
st.markdown("</div>", unsafe_allow_html=True)

# retrieval params
k = 5
min_conf = 0.25

# ---------- Login Gate ----------
if mode == "Instructor Dashboard" and not st.session_state.authenticated:
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Instructor Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    colA, colB = st.columns([1,1])
    if colA.button("Login"):
        if u == "eece" and p == "690":
            st.session_state.authenticated = True
            st.success("✅ Login successful!"); st.rerun()
        else:
            st.error("❌ Invalid credentials.")
    if colB.button("Cancel"): st.info("Login canceled.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------- STUDENT CHAT ----------
if mode == "Student Chat":
    # optional UI zoom
    st.markdown("### Display")
    st.session_state.ui_scale = st.slider("🔎 UI scale", 90, 130, st.session_state.ui_scale, 1, help="Zoom the student chat interface (%)")
    st.markdown(f"<style>.stApp {{ zoom: {st.session_state.ui_scale/100}; }}</style>", unsafe_allow_html=True)

    def _mark_submit(): st.session_state.submitted = True

    st.divider(); st.subheader("Conversation")
    st.markdown("<div class='chat-widget'><div class='chat-card'>", unsafe_allow_html=True)

    for idx, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"<div class='chat-row user'><div class='avatar user'>You</div><div><div class='bubble user'>{chat['question']}</div></div></div>", unsafe_allow_html=True)
        if isinstance(chat.get("img_path"), str) and os.path.exists(chat["img_path"]):
            st.image(chat["img_path"], caption="Attached image", use_container_width=True)
        badge = _confidence_to_badge(chat.get("confidence", 0.0))
        st.markdown(f"<div class='chat-row'><div class='avatar assistant'>TA</div><div><div class='bubble assistant {badge}'>{chat['answer']}</div><div class='meta'>Confidence: {chat.get('confidence',0.0):.2f}</div></div></div>", unsafe_allow_html=True)

        fb1, fb2 = st.columns([0.12, 0.12])
        with fb1:
            if st.button("👍 Helpful", key=f"helpful_{idx}"):
                chat["feedback"] = "helpful"; update_feedback(chat.get("qid"), "helpful"); st.success("Thanks!")
        with fb2:
            if st.button("👎 Unclear", key=f"unclear_{idx}"):
                chat["feedback"] = "unclear"; update_feedback(chat.get("qid"), "unclear"); st.warning("Noted.")

        with st.expander("Retrieval Diagnostics"):
            t1, t2, t3, t4 = st.tabs(["openai→text", "img→img", "text↔img", "Contexts"])
            dbg = chat.get("debug", {}) or {}
            with t1:
                st.markdown("<span class='model-badge model-text'>text-embedding-3</span>", unsafe_allow_html=True)
                if dbg.get("text_openai"):
                    for k2, sc in list(dbg["text_openai"].items())[:5]:
                        meta = rag_engine1.chunk_meta[int(k2)]
                        st.markdown(f"• **{meta['pdf']}** p.{meta['page']} → <span style='color:#22c55e'>`{float(sc):.3f}`</span>", unsafe_allow_html=True)
                else: st.caption("No text matches.")
            with t2:
                st.markdown("<span class='model-badge model-img'>CLIP (img→img)</span>", unsafe_allow_html=True)
                if dbg.get("img2img"):
                    for key2, sc in sorted(dbg["img2img"].items(), key=lambda x: -x[1])[:5]:
                        if isinstance(key2, (tuple, list)):
                            pdf, page = key2[0], key2[1] if len(key2)>1 else "?"
                        else:
                            parts = re.split(r"[:,()'\\s]+", str(key2)); pdf = next((p for p in parts if p.endswith(".pdf")), "?"); page = next((p for p in parts if p.isdigit()), "?")
                        st.markdown(f"• **{pdf}** p.{page} → <span style='color:#22c55e'>`{float(sc):.3f}`</span>", unsafe_allow_html=True)
                else: st.caption("No image→image matches.")
            with t3:
                st.markdown("<span class='model-badge model-clip'>CLIP (text↔img)</span>", unsafe_allow_html=True)
                if dbg.get("text2img"):
                    for key2, sc in sorted(dbg["text2img"].items(), key=lambda x: -x[1])[:5]:
                        if isinstance(key2, (tuple, list)):
                            pdf, page = key2[0], key2[1] if len(key2)>1 else "?"
                        else:
                            parts = re.split(r"[:,()'\\s]+", str(key2)); pdf = next((p for p in parts if p.endswith(".pdf")), "?"); page = next((p for p in parts if p.isdigit()), "?")
                        st.markdown(f"• **{pdf}** p.{page} → <span style='color:#22c55e'>`{float(sc):.3f}`</span>", unsafe_allow_html=True)
                else: st.caption("No text↔image matches.")
            with t4:
                st.markdown("<span class='model-badge model-ctx'>Context Chunks</span>", unsafe_allow_html=True)
                if chat.get("contexts"):
                    for ctx in chat["contexts"][:5]:
                        st.markdown(f"• **{ctx['pdf']}** p.{ctx['page']} (score=<span style='color:#22c55e'>`{float(ctx['score']):.3f}`</span>)", unsafe_allow_html=True)
                else: st.caption("No chunks included.")

    if st.session_state.submitted and st.session_state.get("chat_input","").strip():
        st.markdown("<div style='text-align:center; padding:14px; color:var(--cyan);'><b>💭 SmartTA is thinking...</b></div>", unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='chat-input-footer'>", unsafe_allow_html=True)
    st.markdown("### Ask a question:")
    q = st.text_input("", placeholder="What's your question?", key="chat_input", on_change=lambda: st.session_state.update(submitted=True))
    img = st.file_uploader("Optional: Upload slide image", type=["png","jpg","jpeg"], key=f"uploader_{st.session_state.uploader_key}")
    if img: st.image(img, caption="Uploaded preview", use_container_width=True)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1: ask_clicked = st.button("Ask SmartTA 🚀", key="ask_btn")
    with c2: clear = st.button("Clear Chat 🧹", key="clear_btn")
    with c3: export_txt = st.button("Export Chat (.txt)", key="export_txt_btn")
    with c4: export_pdf = st.button("Export PDF 📑", disabled=not REPORTLAB_OK, key="export_pdf_btn")

    if export_txt and st.session_state.chat_history:
        lines = []
        for c in st.session_state.chat_history:
            lines.append(f"You: {c['question']}")
            if c.get("img_path"): lines.append("[image attached]")
            lines.append(f"SmartTA: {c['answer']}\n")
        st.session_state.dl_txt = ("\n".join(lines)).encode("utf-8")

    if export_pdf and st.session_state.chat_history and REPORTLAB_OK:
        aub_logo = find_aub_logo()
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.6*inch, bottomMargin=0.6*inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=26, leading=30, spaceAfter=12)
        subtitle_style = ParagraphStyle('SubCenter', parent=styles['BodyText'], alignment=TA_CENTER, fontSize=14, leading=18, spaceAfter=8)
        meta_style = ParagraphStyle('MetaCenter', parent=styles['Italic'], alignment=TA_CENTER, fontSize=10, leading=12, spaceAfter=12)
        qa_q_style = ParagraphStyle('QAQ', parent=styles['BodyText'], fontSize=11, leading=14, spaceAfter=4)
        qa_a_style = ParagraphStyle('QAA', parent=styles['BodyText'], fontSize=11, leading=16, spaceAfter=10)

        story = []
        if aub_logo and os.path.exists(aub_logo):
            logo = RLImage(aub_logo, width=2.6*inch, height=2.6*inch); logo.hAlign = 'CENTER'; story.append(logo)
        story.append(Spacer(1,0.12*inch))
        story.append(Paragraph("<b>American University of Beirut – EECE Department</b>", title_style))
        story.append(Paragraph("EECE 490/690 — Intro to Machine Learning", subtitle_style))
        story.append(Paragraph(f"<i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", meta_style))
        story.append(Spacer(1,0.25*inch))
        for c in st.session_state.chat_history:
            story.append(Paragraph(f"<b>Question:</b> {c.get('question','')}", qa_q_style))
            if isinstance(c.get("img_path"), str) and os.path.exists(c["img_path"]):
                img_obj = RLImage(c["img_path"], width=5.8*inch, height=3.6*inch); img_obj.hAlign = 'CENTER'
                story.append(img_obj); story.append(Spacer(1,0.08*inch))
            story.append(Paragraph(f"<b>Answer:</b> {c.get('answer','')}", qa_a_style))
            story.append(Spacer(1,0.12*inch))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e6e6e6')))
            story.append(Spacer(1,0.12*inch))
        doc.build(story); buf.seek(0); st.session_state.dl_pdf = buf.getvalue()

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
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        components.html(r"""
        <script>
        (function(){
          function pad(){try{
            var f=document.querySelector('.chat-input-footer'), a=document.querySelector('.stApp');
            if(!f||!a) return; a.style.paddingBottom=((f.offsetHeight||0)+28)+'px';
          }catch(e){}}
          setTimeout(pad,160); window.addEventListener('resize',pad);
          new MutationObserver(pad).observe(document.body,{childList:true,subtree:true,attributes:true});
        })();
        </script>
        """, height=0)
    except Exception: pass

    if clear:
        st.session_state.chat_history = []
        st.session_state.uploader_key += 1
        st.session_state.submitted = False
        st.session_state.dl_txt = None
        st.session_state.dl_pdf = None
        st.rerun()

# ---------- Student Library (SmartTA_Extensions) ----------
try:
    from frontend.tabs.student import render_student_tab
    from utils.helpers import load_youtube_lectures, get_youtube_lectures_categorized
    from backend.indexing import reindex_if_needed
except Exception:
    render_student_tab = None

if mode == "Student Chat":
    st.markdown("---")
    st.subheader("📚 Lecture Library & Search (YouTube) — Extended")

    if render_student_tab is None:
        st.warning("Extended student features unavailable (missing SmartTA_Extensions).")
    else:
        with ext_cwd():
            os.makedirs("data", exist_ok=True)
            os.makedirs("data/transcripts", exist_ok=True)
            os.makedirs("data/models", exist_ok=True)
            os.makedirs("data/uploaded_files", exist_ok=True)
            os.makedirs("data/lectures", exist_ok=True)

            META_SEGMENTS = "data/segments_metadata.json"
            META_PATH = "data/metadata.json"
            INDEX_PATH = "data/course.index"
            LOG_PATH_EXT = "data/queries_log.json"
            UPLOADS_DIR = "data/uploaded_files"

            if os.path.exists(META_SEGMENTS):
                with open(META_SEGMENTS, "r", encoding="utf-8") as f:
                    all_segments = json.load(f)
            else:
                all_segments = {}

            with st.spinner("Initializing Student Library (YouTube + FAISS)…"):
                metadata, index, new_indexes = reindex_if_needed(all_segments, load_youtube_lectures)
                if new_indexes == 0: st.caption("Student Library is up-to-date.")
                else: st.info(f"Indexed {new_indexes} new lecture(s).")

            yt = load_youtube_lectures()
            try:
                lecture_categories = get_youtube_lectures_categorized()
            except Exception:
                lecture_categories = {}

            all_lectures = []
            try:
                all_lectures = [
                    yt["lectures"][lec_id]["title"]
                    for cat_ids in lecture_categories.values()
                    for lec_id in cat_ids
                ]
                all_lectures = sorted(all_lectures)
            except Exception:
                pass

            try:
                render_student_tab(
                    "data",
                    META_SEGMENTS,
                    META_PATH,
                    INDEX_PATH,
                    LOG_PATH_EXT,
                    UPLOADS_DIR,
                    all_lectures,
                    lecture_categories,
                    metadata,
                    index,
                )
            except Exception as e:
                st.error(f"Failed to load extended student library: {e}")

    # final send
    if ('ask_clicked' in locals() and ask_clicked) or (st.session_state.submitted and (st.session_state.get("chat_input","").strip())):
        tmp_img = None
        if img:
            suffix = Path(img.name).suffix.lower() or ".png"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(img.read()); tmp.flush()
            tmp_img = tmp.name

        out = rag_engine1.answer_with_citations(
            question=st.session_state.get("chat_input",""), screenshot_path=tmp_img, k=k, min_conf=min_conf,
            model="gpt-4o-mini", debug=True
        )
        conf = max([c["score"] for c in out.get("contexts", [])], default=0.0)
        record = {
            "qid": str(uuid.uuid4()),
            "question": st.session_state.get("chat_input",""),
            "answer": out.get("answer", ""),
            "debug": out.get("page_scores", {}),
            "contexts": out.get("contexts", []),
            "img_path": tmp_img,
            "confidence": conf,
            "feedback": None,
        }
        st.session_state.chat_history.append(record)
        append_log({**record, "img_path": bool(tmp_img)})
        st.session_state.uploader_key += 1
        st.session_state.submitted = False
        st.rerun()

# ---------- INSTRUCTOR DASHBOARD ----------
elif mode == "Instructor Dashboard" and st.session_state.authenticated:
    st.markdown("## 🧑‍🏫 Instructor Dashboard")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

    df = read_logs_df()
    if df.empty:
        st.info("No logs yet — ask some questions first.")
        st.stop()

    # ===== Filters (CRASH FIXED) =====
    st.markdown("### 📅 Filters")
    # keep only rows with valid 'date' (read_logs_df already enforces this)
    df = df[df["date"].notna()].copy()
    if df.empty:
        st.info("No valid dated logs yet — ask some questions first.")
        st.stop()

    # compute bounds safely
    date_series = pd.Series(list(df["date"]))
    min_d = date_series.min()
    max_d = date_series.max()
    today = datetime.now().date()
    if not isinstance(min_d, date_cls): min_d = today
    if not isinstance(max_d, date_cls): max_d = today

    d_range = st.date_input("Select Date Range", value=(min_d, max_d))
    if isinstance(d_range, tuple) and len(d_range) == 2:
        d_from, d_to = d_range
    else:
        d_from, d_to = (min_d, max_d)

    conf_filter = st.slider("Min confidence filter", 0.0, 1.0, 0.0, 0.01)
    fb_filter = st.selectbox("Feedback filter", ["all", "helpful only", "unclear only"])

    # Apply filters safely (date is pure datetime.date)
    mask = (df["date"] >= d_from) & (df["date"] <= d_to) & (df["top_score"] >= conf_filter)
    df = df[mask].copy()

    if fb_filter == "helpful only":
        df = df[df["feedback"] == "helpful"]
    elif fb_filter == "unclear only":
        df = df[df["feedback"] == "unclear"]

    if df.empty:
        st.warning("No data after filters.")
        st.stop()

    # Prep for repeats
    df["q_norm"] = df["question"].fillna("").astype(str).map(_alias_q)
    base_for_repeats = df

    # KPIs
    total_q = int(len(df))
    avg_conf = float(df["top_score"].mean())
    active_days = int(df["date"].nunique())
    helpful_cnt = int((df["feedback"] == "helpful").sum())
    unclear_cnt = int((df["feedback"] == "unclear").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f"<div class='metric-card'><h2>{total_q}</h2><p>Total Q&A Rows</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>{avg_conf:.3f}</h2><p>Average Confidence</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h2>{active_days}</h2><p>Active Days</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h2>{helpful_cnt}</h2><p>👍 Helpful</p></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='metric-card'><h2>{unclear_cnt}</h2><p>👎 Unclear</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # 🔝 Top 10 Topics
    st.markdown("### 🔝 Top 10 Topics")
    st.caption("Frequent 2–5 word phrases from student questions (stop-words removed) — provides deeper context on student inquiries.")
    questions = df["question"].dropna().astype(str)
    if not questions.empty:
        try:
            if _HAS_SKLEARN:
                vectorizer = CountVectorizer(
                    ngram_range=(2, 5),
                    stop_words="english",
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                )
                X = vectorizer.fit_transform(questions)
                freqs = X.toarray().sum(axis=0)
                vocab = vectorizer.get_feature_names_out()
                topic_df = pd.DataFrame({"Phrase": vocab, "Count": freqs}).sort_values("Count", ascending=False).head(10)

                st.altair_chart(
                    alt.Chart(topic_df).mark_bar(color="#2e7cf6")
                    .encode(x="Count:Q", y=alt.Y("Phrase:N", sort='-x', axis=alt.Axis(labelFontSize=12, labelLimit=300)), tooltip=["Phrase","Count"])
                    .properties(height=450, width=900),
                    use_container_width=True
                )
            else:
                st.info("scikit-learn not available; skipping n-gram topics.")
        except Exception:
            st.info("Not enough signal to compute topics yet.")
    else:
        st.caption("No questions to analyze.")

    st.markdown("---")

    # 📄 Most Referenced Slides
    st.markdown("### 📄 Most Referenced Slides")
    mentioned = []
    for L in df["contexts"].dropna():
        if isinstance(L, list):
            for c in L:
                try:
                    mentioned.append(f"{c.get('pdf')} p.{c.get('page')}")
                except Exception:
                    pass
    if mentioned:
        conf_df = pd.Series(mentioned).value_counts().reset_index()
        conf_df.columns = ["Slide", "Mentions"]
        conf_df = conf_df.head(12)
        st.altair_chart(
            alt.Chart(conf_df).mark_bar(color="#ef4444")
            .encode(x="Mentions:Q", y=alt.Y("Slide:N", sort='-x'), tooltip=["Slide","Mentions"])
            .properties(height=320),
            use_container_width=True
        )
    else:
        st.caption("No slides referenced yet.")

    st.markdown("---")

    # 🧩 Slide References per Chapter (C1–C6)
    st.markdown("### 🧩 Slide References per Chapter (C1–C6)")
    st.caption("Counts unique slides referenced per chapter.")
    all_refs = []
    for L in df["contexts"].dropna():
        if isinstance(L, list):
            for c in L:
                pdf = str(c.get("pdf", ""))
                page = c.get("page", None)
                m = re.search(r"(C[1-6])", pdf)
                if m: all_refs.append((m.group(1), page))
    if all_refs:
        refs_df = pd.DataFrame(all_refs, columns=["Chapter", "Page"]).drop_duplicates()
        slide_counts = (refs_df["Chapter"].value_counts().reindex(["C1","C2","C3","C4","C5","C6"]).fillna(0).astype(int).reset_index())
        slide_counts.columns = ["Chapter", "UniqueSlides"]
        st.altair_chart(
            alt.Chart(slide_counts).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#3b82f6")
            .encode(x=alt.X("Chapter:N", sort=["C1","C2","C3","C4","C5","C6"], title="Chapter"),
                    y=alt.Y("UniqueSlides:Q", title="Unique Slides"),
                    tooltip=["Chapter","UniqueSlides"])
            .properties(height=350, title="Unique Slides Referenced per Chapter")
            .configure_axis(labelColor="#e5e7eb", titleColor="#e5e7eb"),
            use_container_width=True
        )
    else:
        st.caption("No slide references yet.")

    st.markdown("---")

    # ❗ Low-rated / Low-confidence
    st.markdown("### ❗ Low-Rated & Low-Confidence Responses")
    tabA, tabB = st.tabs(["👎 Unclear (by time)", "Lowest Confidence (top 10)"])
    with tabA:
        unclear = df[df["feedback"] == "unclear"][["ts","question","top_score","answer"]]
        if unclear.empty:
            st.caption("No 'unclear' feedback yet.")
        else:
            unclear = unclear.sort_values("ts", ascending=False).rename(columns={"ts":"Time","top_score":"Confidence"})
            st.dataframe(unclear, use_container_width=True)
    with tabB:
        low_conf = df.sort_values("top_score").head(10)[["question","top_score","answer","date"]]
        st.dataframe(low_conf.rename(columns={"top_score":"Confidence"}), use_container_width=True)

    st.markdown("---")

    # 👍/👎 Feedback distribution
    st.markdown("### 👍/👎 Feedback Distribution")
    fb_counts = (df["feedback"].value_counts().reindex(["helpful","unclear","none"]).fillna(0).astype(int).reset_index())
    fb_counts.columns = ["Feedback","Count"]
    st.altair_chart(
        alt.Chart(fb_counts).mark_bar(color="#34d399")
        .encode(x=alt.X("Feedback:N", title="Type"),
                y=alt.Y("Count:Q", title="Count"),
                tooltip=["Feedback","Count"])
        .properties(height=300),
        use_container_width=True
    )

    # Repeated questions
    st.markdown("---"); st.markdown("### ♻️ Repeated Questions")
    rep = (base_for_repeats.groupby("q_norm").size().sort_values(ascending=False).reset_index(name="Times").rename(columns={"q_norm":"Question"}))
    if not rep.empty:
        st.dataframe(rep.head(15), use_container_width=True, height=300)
    else:
        st.caption("No repeated questions yet.")

    # ⏱️ Engagement Heatmap (Hour × Day)
    st.markdown("### ⏱️ Engagement Heatmap (Hour × Day)")
    ts = df["ts"].dropna()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    grid = pd.MultiIndex.from_product([day_order, list(range(24))], names=["dow","hour"]).to_frame(index=False)
    if not ts.empty:
        counts = (pd.DataFrame({"dow": ts.dt.day_name(), "hour": ts.dt.hour}).value_counts(["dow","hour"]).rename("count").reset_index())
        heat = grid.merge(counts, on=["dow","hour"], how="left").fillna({"count":0})
    else:
        heat = grid.copy(); heat["count"] = 0
    def _fmt_hour(h):
        if h == 0: return "12 am"
        if h < 12: return f"{h} am"
        if h == 12: return "12 pm"
        return f"{h-12} pm"
    heat["hour_label"] = heat["hour"].map(_fmt_hour)
    chart = (alt.Chart(heat).mark_rect().encode(
        x=alt.X("hour:O", sort=list(range(24)), title="Hour",
                axis=alt.Axis(values=list(range(24)),
                              labelExpr=('datum.value==0 ? "12 am" : '
                                         'datum.value<12 ? datum.value + " am" : '
                                         'datum.value==12 ? "12 pm" : (datum.value-12) + " pm"'))),
        y=alt.Y("dow:N", sort=day_order, title="Day"),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
        tooltip=[alt.Tooltip("dow:N", title="Day"), alt.Tooltip("hour_label:N", title="Hour"), alt.Tooltip("count:Q", title="Count")],
    ).properties(height=280))
    st.altair_chart(chart, use_container_width=True)

    # 🕒 Engagement Over Time
    st.markdown("### 🕒 Engagement Over Time")
    daily = df.groupby("date").size().reset_index(name="Questions")
    st.altair_chart(
        alt.Chart(daily).mark_area(color="#20c997").encode(x="date:T", y="Questions:Q", tooltip=["date:T","Questions:Q"]).properties(height=300),
        use_container_width=True
    )

    st.markdown("---")
    st.markdown("### 📥 Student Questions Database")
    csv_bytes = df.sort_values("ts").to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Student Questions", data=csv_bytes,
                       file_name=f"student_questions_{datetime.now().strftime('%Y-%m-%d')}.csv",
                       mime="text/csv", use_container_width=True)
