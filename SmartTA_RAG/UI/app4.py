# app.py — SmartTA RAG (AUB Edition — Final Secure + Feedback Version, qid fix)
# Student Chat + Feedback (no duplicates) + Instructor Login + PDF Export + Colored Diagnostics + Dashboard Design

import os
import sys
import re
import json
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt

# ---------- Optional PDF Export ----------
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# ---------- Optional: Contextual n-grams ----------
_HAS_SKLEARN = True
try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception:
    _HAS_SKLEARN = False

# ---------- Import rag_engine ----------
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))
import rag_engine1  # noqa


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="🧠 SmartTA — Multimodal RAG Assistant", layout="wide")



DATA_DIR = CURRENT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "chat_logs.ndjson"


# =============================================================================
# CSS (Design)
# =============================================================================
st.markdown("""
<style>
:root {
  --bg:#0D0E12; --text:#F3F3F3;
  --blue:#2e7cf6; --cyan:#00C6FF;
  --green:#22c55e; --amber:#f59e0b;
  --red:#ef4444; --purple:#6d28d9;
  --card:#171923; --muted:#9aa2b1;
}
html, body, .stApp { background:var(--bg); color:var(--text); font-family:'Inter', system-ui; }

/* Title */
.smartta-title {
  text-align:center; font-size:2.4rem; font-weight:900;
  margin:20px 0; background:linear-gradient(90deg,var(--cyan),var(--blue));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:0 0 20px rgba(0,198,255,0.3);
}

/* Chat bubbles */
.user-bubble {
  background:linear-gradient(135deg,#0066FF,#00BFFF); color:white;
  padding:12px 14px; border-radius:14px; margin:8px 0; max-width:75%;
}
.assistant-bubble {
  background:rgba(255,255,255,.07); color:var(--text);
  padding:12px 14px; border-radius:14px; margin:8px 0; max-width:85%;
  border-left:5px solid transparent;
}
.bubble-high { border-left-color:var(--green); }
.bubble-mid  { border-left-color:var(--amber); }
.bubble-low  { border-left-color:var(--red); }

/* Dashboard cards */
.metric-card {
  background:linear-gradient(135deg,#1e293b,#0f172a);
  border-radius:14px; padding:18px; text-align:center;
  box-shadow:0 0 14px rgba(0,0,0,.3);
  transition:transform .2s ease;
}
.metric-card:hover { transform:scale(1.04); }
.metric-card h2 { margin:0; font-size:1.8rem; color:var(--cyan); }
.metric-card p { margin:4px 0 0 0; color:var(--muted); }

/* Diagnostics tab colors */
div[role="tablist"] > button[role="tab"]:nth-child(1) > div > p { color:#60a5fa !important; font-weight:700; }
div[role="tablist"] > button[role="tab"]:nth-child(2) > div > p { color:#34d399 !important; font-weight:700; }
div[role="tablist"] > button[role="tab"]:nth-child(3) > div > p { color:#fb923c !important; font-weight:700; }
div[role="tablist"] > button[role="tab"]:nth-child(4) > div > p { color:#c084fc !important; font-weight:700; }

/* Model badges */
.model-badge {
  display:inline-block; font-weight:700; font-size:0.8rem;
  padding:3px 8px; border-radius:999px; margin-bottom:4px;
}
.model-text { background:#1d4ed8; color:white; }
.model-img  { background:#15803d; color:white; }
.model-clip { background:#b45309; color:white; }
.model-ctx  { background:#6d28d9; color:white; }

/* Login box */
.login-box {
  background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.15);
  padding:25px; border-radius:12px;
  width:350px; margin:auto; text-align:center;
  box-shadow:0 0 20px rgba(0,0,0,.4);
}

.chat-container { max-width:1200px; margin:0 auto; padding:8px 12px 40px 12px; }
.chat-input-row { display:flex; gap:12px; align-items:center; margin-top:8px; }
.chat-input { flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); padding:14px 16px; border-radius:14px; color:var(--text); }
.send-btn { background:linear-gradient(90deg,var(--cyan),var(--blue)); color:white; padding:10px 16px; border-radius:12px; border:none; cursor:pointer; box-shadow:0 6px 18px rgba(46,124,246,0.18); }
.action-btn { background:transparent; color:var(--muted); border:1px solid rgba(255,255,255,0.06); padding:8px 12px; border-radius:10px; }

/* Chat rows with avatars */
.chat-row { display:flex; gap:12px; align-items:flex-start; margin:8px 0; }
.avatar { width:36px; height:36px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center; font-weight:700; }
.avatar.user { background:linear-gradient(135deg,#0066FF,#00BFFF); color:white; }
.avatar.assistant { background:linear-gradient(135deg,#1f2937,#111827); color:#9fb0c8; border:1px solid rgba(255,255,255,0.03); }
.bubble { padding:12px 14px; border-radius:12px; max-width:78%; font-size:0.98rem; line-height:1.4; }
.bubble.user { background:linear-gradient(135deg,#0066FF,#00BFFF); color:white; border-bottom-right-radius:4px; }
.bubble.assistant { background:rgba(255,255,255,0.03); color:var(--text); border:1px solid rgba(255,255,255,0.04); }
.meta { font-size:0.78rem; color:var(--muted); margin-top:6px; }

/* Compact diagnostics panel */
.stExpander > .streamlit-expanderHeader { padding:0.5rem 0.75rem; }
 
/* Chat card and alignment variants */
.chat-card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:16px; padding:18px; box-shadow: 0 8px 30px rgba(2,6,23,0.6); }
.chat-row.user { flex-direction: row-reverse; }
.chat-row.user .bubble.user { border-bottom-left-radius:4px; border-bottom-right-radius:12px; }
.chat-row.user .avatar.user { margin-left:8px; }
.chat-row .avatar { flex: 0 0 36px; }
.chat-row .bubble { flex: 1 1 auto; }

/* Mode selector styling (interactive - horizontal, catchy design) */
div[role="radiogroup"] { 
  display: flex !important; 
  flex-direction: row !important; 
  gap: 12px !important;
  justify-content: center !important;
  align-items: center !important;
  width: 100% !important;
}
div[role="radio"] {
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03)) !important;
  border: 1.5px solid rgba(255,255,255,0.15) !important;
  border-radius: 16px !important;
  padding: 10px 16px !important;
  cursor: pointer !important;
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.3px !important;
  position: relative !important;
  overflow: hidden !important;
}
div[role="radio"]::before {
  content: "" !important;
  position: absolute !important;
  top: 0 !important;
  left: -100% !important;
  width: 100% !important;
  height: 100% !important;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
  transition: left 0.5s ease !important;
}
div[role="radio"]:hover {
  background: linear-gradient(135deg, rgba(46, 124, 246, 0.25), rgba(0, 198, 255, 0.15)) !important;
  border-color: #2e7cf6 !important;
  box-shadow: 0 8px 25px rgba(46, 124, 246, 0.25), inset 0 1px 0 rgba(255,255,255,0.1) !important;
  transform: translateY(-2px) !important;
}
div[role="radio"]:hover::before {
  left: 100% !important;
}
div[role="radio"][aria-checked="true"] {
  background: linear-gradient(135deg, rgba(46, 124, 246, 0.4), rgba(0, 198, 255, 0.25)) !important;
  border-color: #00C6FF !important;
  box-shadow: 0 12px 35px rgba(46, 124, 246, 0.35), 0 0 25px rgba(0, 198, 255, 0.2), inset 0 1px 0 rgba(255,255,255,0.15) !important;
  color: #00C6FF !important;
  font-weight: 700 !important;
  transform: translateY(-3px) scale(1.05) !important;
}

.chat-input-footer { 
    position: relative !important;
    margin-top: 24px !important;
    padding: 16px 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    z-index: 10;
}
.chat-input-section { max-width: 100%; margin: 0 auto; padding: 0 8px; }
.chat-footer { position: relative; background: transparent; padding: 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================
def find_aub_logo() -> str | None:
    path = CURRENT_DIR / "aub_logo.png"
    return str(path) if path.exists() else None


def _tuple_key_dict_to_str_key(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(k, tuple):
            out[f"{k[0]}:{k[1]}"] = v
        else:
            out[str(k)] = v
    return out


def sanitize_debug(debug: dict) -> dict:
    out = {}
    if debug is None:
        return out
    if "text_openai" in debug:
        out["text_openai"] = {str(k): float(v) for k, v in debug["text_openai"].items()}
    if "img2img" in debug:
        out["img2img"] = _tuple_key_dict_to_str_key(debug["img2img"])
    if "text2img" in debug:
        out["text2img"] = _tuple_key_dict_to_str_key(debug["text2img"])
    return out


def append_log(entry: dict):
    """Append a NEW Q&A row once (includes a unique qid)."""
    entry = dict(entry)
    # ensure qid exists
    entry.setdefault("qid", str(uuid.uuid4()))
    entry["timestamp"] = datetime.utcnow().isoformat()
    entry["debug"] = sanitize_debug(entry.get("debug", {}))
    entry["contexts"] = [
        {"pdf": c.get("pdf"), "page": c.get("page"), "score": float(c.get("score", 0.0))}
        for c in entry.get("contexts", []) or []
    ]
    if entry.get("img_path"):
        entry["img_path"] = True
    entry["feedback"] = entry.get("feedback", None)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_feedback(qid: str, feedback_value: str):
    """Update feedback for an existing row by qid (no duplicate rows)."""
    if not qid:
        return
    rows = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    # update match
    updated = False
    for r in rows:
        if r.get("qid") == qid:
            r["feedback"] = feedback_value
            updated = True
            break

    # (Optional legacy fallback) if no qid found, do nothing to avoid wrong merges.
    if not updated:
        return

    # rewrite file
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_logs_df() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # normalize qid for legacy rows (avoid accidental merging by text)
    if "qid" not in df.columns:
        df["qid"] = [str(uuid.uuid4()) for _ in range(len(df))]
    else:
        df["qid"] = df["qid"].fillna("").astype(str).replace({"": pd.NA})
        df.loc[df["qid"].isna(), "qid"] = [str(uuid.uuid4()) for _ in range(df["qid"].isna().sum())]

    df["ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["date"] = df["ts"].dt.date

    def _top_score(L):
        try:
            return max([float(c.get("score", 0.0)) for c in (L or [])] + [0.0])
        except Exception:
            return 0.0

    df["top_score"] = df.get("contexts", []).apply(_top_score)

    if "feedback" not in df.columns:
        df["feedback"] = None
    df["feedback"] = df["feedback"].fillna("none")

    # keep the latest per qid
    df = df.sort_values("ts").drop_duplicates(subset=["qid"], keep="last")
    return df


def _confidence_to_badge(conf: float) -> str:
    if conf < 0.4: return "bubble-low"
    if conf < 0.75: return "bubble-mid"
    return "bubble-high"


# =============================================================================
# INIT
# =============================================================================
if "engine_loaded" not in st.session_state:
    with st.spinner("Loading RAG indexes..."):
        rag_engine1.load_indexes()
        st.session_state.engine_loaded = True

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("submitted", False)
st.session_state.setdefault("authenticated", False)


# =============================================================================
# MODE SELECTOR (Top, horizontal, catchy design)
# =============================================================================
st.markdown("<div class='smartta-title'>🧠 SmartTA — Multimodal RAG Assistant</div>", unsafe_allow_html=True)

# Create a premium mode selector bar with buttons inside
st.markdown("""
<div style='background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01), rgba(255,255,255,0.02)); 
            border: 1px solid rgba(0,198,255,0.2); border-radius: 20px; padding: 24px 32px; 
            margin: 16px auto; max-width: 900px; box-shadow: 0 8px 32px rgba(46,124,246,0.1);
            backdrop-filter: blur(10px); display: flex; justify-content: center; align-items: center;'>
""", unsafe_allow_html=True)

# Mode selector centered in the middle
_, col_center, _ = st.columns([1, 1, 1])
with col_center:
    mode_choice = st.radio("", ["🎓 Student Chat", "📊 Instructor Dashboard"], horizontal=True, label_visibility="collapsed")
    # Extract mode value (remove emoji prefix)
    if "Student" in mode_choice:
        mode = "Student Chat"
    else:
        mode = "Instructor Dashboard"

st.markdown("</div>", unsafe_allow_html=True)

# Fixed default parameters
k = 5
min_conf = 0.25


# =============================================================================
# CONTENT
# =============================================================================

# =============================================================================
# LOGIN GATE FOR INSTRUCTOR
# =============================================================================
if mode == "Instructor Dashboard" and not st.session_state.authenticated:
    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("### 🔐 Instructor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    colA, colB = st.columns([1,1])
    if colA.button("Login"):
        if username == "eece" and password == "690":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials.")
    if colB.button("Cancel"):
        st.info("Login canceled.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# =============================================================================
# STUDENT CHAT
# =============================================================================
if mode == "Student Chat":
    def _mark_submit():
        st.session_state.submitted = True

    # Conversation area — render as chat rows with avatars and meta
    st.divider(); st.subheader("Conversation")
    st.markdown("<div class='chat-widget'><div class='chat-card'>", unsafe_allow_html=True)
    for idx, chat in enumerate(st.session_state.chat_history):
        # user row
        user_html = (
            f"<div class='chat-row user'><div class='avatar user'>You</div>"
            f"<div><div class='bubble user'>{chat['question']}</div></div></div>"
        )
        st.markdown(user_html, unsafe_allow_html=True)
        # optional image
        if isinstance(chat.get("img_path"), str) and os.path.exists(chat["img_path"]):
            st.image(chat["img_path"], caption="Referenced image", use_container_width=True)
        # assistant row
        badge = _confidence_to_badge(chat.get("confidence", 0.0))
        assist_html = (
            f"<div class='chat-row'><div class='avatar assistant'>TA</div>"
            f"<div><div class='bubble assistant {badge}'>{chat['answer']}</div>"
            f"<div class='meta'>Confidence: {chat.get('confidence',0.0):.2f}</div></div></div>"
        )
        st.markdown(assist_html, unsafe_allow_html=True)

        # Feedback buttons (compact)
        fb_col1, fb_col2 = st.columns([0.12, 0.12])
        with fb_col1:
            if st.button("👍 Helpful", key=f"helpful_{idx}"):
                chat["feedback"] = "helpful"
                update_feedback(chat.get("qid"), "helpful")
                st.success("Thanks for your feedback!")
        with fb_col2:
            if st.button("👎 Unclear", key=f"unclear_{idx}"):
                chat["feedback"] = "unclear"
                update_feedback(chat.get("qid"), "unclear")
                st.warning("Marked as unclear — noted.")

        # Diagnostics
        with st.expander("Retrieval Diagnostics"):
            t1, t2, t3, t4 = st.tabs(["openai→text", "img→img", "text↔img", "Contexts"])
            dbg = chat.get("debug", {}) or {}

            # text
            with t1:
                st.markdown("<span class='model-badge model-text'>text-embedding-3</span>", unsafe_allow_html=True)
                if dbg.get("text_openai"):
                    for k2, sc in list(dbg["text_openai"].items())[:5]:
                        meta = rag_engine1.chunk_meta[int(k2)]
                        st.markdown(
                            f"• **{meta['pdf']}** p.{meta['page']} → "
                            f"<span style='color:#22c55e'>`{float(sc):.3f}`</span>",
                            unsafe_allow_html=True)
                else:
                    st.caption("No text matches.")

            # img→img
            with t2:
                st.markdown("<span class='model-badge model-img'>CLIP (img→img)</span>", unsafe_allow_html=True)
                if dbg.get("img2img"):
                    for key2, sc in sorted(dbg["img2img"].items(), key=lambda x: -x[1])[:5]:
                        # Handle both tuple keys and string forms
                        if isinstance(key2, (tuple, list)):
                            pdf, page = key2[0], key2[1] if len(key2) > 1 else "?"
                        else:
                            parts = re.split(r"[:,()'\\s]+", str(key2))
                            pdf = next((p for p in parts if p.endswith(".pdf")), "?")
                            page = next((p for p in parts if p.isdigit()), "?")
                        st.markdown(
                            f"• **{pdf}** p.{page} → "
                            f"<span style='color:#22c55e'>`{float(sc):.3f}`</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("No image→image matches.")

            # text↔img
            with t3:
                st.markdown("<span class='model-badge model-clip'>CLIP (text↔img)</span>", unsafe_allow_html=True)
                if dbg.get("text2img"):
                    for key2, sc in sorted(dbg["text2img"].items(), key=lambda x: -x[1])[:5]:
                        # Handle tuple-style keys or strings
                        if isinstance(key2, (tuple, list)):
                            pdf, page = key2[0], key2[1] if len(key2) > 1 else "?"
                        else:
                            parts = re.split(r"[:,()'\s]+", str(key2))
                            pdf = next((p for p in parts if p.endswith(".pdf")), "?")
                            page = next((p for p in parts if p.isdigit()), "?")
                        st.markdown(
                            f"• **{pdf}** p.{page} → "
                            f"<span style='color:#22c55e'>`{float(sc):.3f}`</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("No text↔image matches.")

            # contexts
            with t4:
                st.markdown("<span class='model-badge model-ctx'>Context Chunks</span>", unsafe_allow_html=True)
                if chat.get("contexts"):
                    for ctx in chat["contexts"][:5]:
                        st.markdown(
                            f"• **{ctx['pdf']}** p.{ctx['page']} "
                            f"(score=<span style='color:#22c55e'>`{float(ctx['score']):.3f}`</span>)",
                            unsafe_allow_html=True)
                else:
                    st.caption("No chunks included.")

    # Show loading spinner INSIDE the conversation area ONLY when a non-empty question was submitted
    # This avoids showing the "SmartTA is thinking..." message when there is no question text.
    if st.session_state.submitted and st.session_state.get("chat_input", "").strip():
        st.markdown("<div style='text-align:center; padding: 20px; color: var(--cyan);'><b>💭 SmartTA is thinking...</b></div>", unsafe_allow_html=True)
    
    # close chat-widget + chat-card wrapper
    st.markdown("</div></div>", unsafe_allow_html=True)

    # (visualizations removed from Student Chat per request)
    st.markdown("---")

    # ===== INPUT SECTION - Rendered right after conversation (NOT fixed at bottom) =====
    # Simple container without fixed positioning
    st.markdown("<div class='chat-input-footer'>", unsafe_allow_html=True)
    
    # Input / upload / action controls
    st.markdown("### Ask a question:")
    q = st.text_input("", placeholder="What's your question?", key="chat_input", on_change=_mark_submit)
    
    img = st.file_uploader(
        "Optional: Upload slide image",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if img:
        st.image(img, caption="Uploaded preview", use_container_width=True)

    # action row: Ask, Clear, Export (styled)
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        ask_clicked = st.button("Ask SmartTA 🚀", key="ask_btn", help="Send the current question")
    with c2:
        clear = st.button("Clear Chat 🧹", key="clear_btn")
    with c3:
        export_txt = st.button("Export Chat (.txt)", key="export_txt_btn")
    with c4:
        export_pdf = st.button("Export PDF 📑", disabled=not REPORTLAB_OK, key="export_pdf_btn")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Add a small JS helper to ensure the page reserves space for the fixed footer
    # This measures the rendered footer height and sets padding on the Streamlit app
    # so the last messages are never hidden. Runs idempotently and on resize/mutations.
    try:
        components.html(r"""
        <script>
        (function(){
          function ensureFooterPad(){
            try{
              var footer = document.querySelector('.chat-input-footer');
              var app = document.querySelector('.stApp');
              if(!footer || !app) return;
              var h = footer.offsetHeight || 0;
              app.style.paddingBottom = (h + 28) + 'px';
            }catch(e){console.warn('footer-pad', e);} 
          }
          setTimeout(ensureFooterPad, 160);
          window.addEventListener('resize', ensureFooterPad);
          var mo = new MutationObserver(ensureFooterPad);
          mo.observe(document.body, { childList:true, subtree:true, attributes:true });
        })();
        </script>
        """, height=0)
    except Exception:
        pass

    if clear:
        st.session_state.chat_history = []
        st.session_state.uploader_key += 1
        st.session_state.submitted = False
        st.rerun()

    # Export TXT
    if export_txt and st.session_state.chat_history:
        transcript = []
        for c in st.session_state.chat_history:
            transcript.append(f"You: {c['question']}")
            if c.get("img_path"): transcript.append("[image attached]")
            transcript.append(f"SmartTA: {c['answer']}\n")
        st.download_button("⬇️ Download Chat (.txt)",
                           ("\n".join(transcript)).encode("utf-8"),
                           file_name="SmartTA_chat.txt", mime="text/plain")

    # Export PDF
    if export_pdf and st.session_state.chat_history and REPORTLAB_OK:
        aub_logo = find_aub_logo()
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        # set sensible margins for a nicer print layout
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.6*inch, bottomMargin=0.6*inch)
        styles = getSampleStyleSheet()
        # create centered title/subtitle styles
        try:
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.enums import TA_CENTER
        except Exception:
            ParagraphStyle = None
            TA_CENTER = None

        title_style = ParagraphStyle('TitleCenter', parent=styles.get('Title', styles['Normal']), alignment=TA_CENTER or 1, fontSize=26, leading=30, spaceAfter=12)
        subtitle_style = ParagraphStyle('SubCenter', parent=styles.get('BodyText', styles['Normal']), alignment=TA_CENTER or 1, fontSize=14, leading=18, spaceAfter=8)
        meta_style = ParagraphStyle('MetaCenter', parent=styles.get('Italic', styles['Normal']), alignment=TA_CENTER or 1, fontSize=10, leading=12, spaceAfter=12)

        story = []
        if aub_logo and os.path.exists(aub_logo):
            # wider, centered logo
            logo = RLImage(aub_logo, width=2.6*inch, height=2.6*inch)
            logo.hAlign = 'CENTER'
            story.append(logo)
        story.append(Spacer(1,0.12*inch))
        story.append(Paragraph("<b>American University of Beirut – EECE Department</b>", title_style))
        story.append(Paragraph("EECE 490/690 — Intro to Machine Learning", subtitle_style))
        story.append(Paragraph(f"<i>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", meta_style))
        story.append(Spacer(1,0.25*inch))

        # add each Q&A with clear separations and consistent styling
        qa_q_style = ParagraphStyle('QAQ', parent=styles.get('BodyText', styles['Normal']), fontSize=11, leading=14, spaceAfter=4)
        qa_a_style = ParagraphStyle('QAA', parent=styles.get('BodyText', styles['Normal']), fontSize=11, leading=16, spaceAfter=10)

        for c in st.session_state.chat_history:
            story.append(Paragraph(f"<b>Question:</b> {c.get('question','')}", qa_q_style))
            if isinstance(c.get("img_path"), str) and os.path.exists(c["img_path"]):
                img_obj = RLImage(c["img_path"], width=5.8*inch, height=3.6*inch)
                img_obj.hAlign = 'CENTER'
                story.append(img_obj)
                story.append(Spacer(1,0.08*inch))
            story.append(Paragraph(f"<b>Answer:</b> {c.get('answer','')}", qa_a_style))
            story.append(Spacer(1,0.12*inch))
            # small divider line
            try:
                from reportlab.lib import colors
                from reportlab.platypus import HRFlowable
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e6e6e6')))
                story.append(Spacer(1,0.12*inch))
            except Exception:
                pass

        doc.build(story)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f.read(),
                               "SmartTA_Session_Report.pdf", mime="application/pdf")

    # Ask
    if (ask_clicked or st.session_state.submitted) and q.strip():
        tmp_img = None
        if img:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img.read()); tmp.flush()
            tmp_img = tmp.name

        out = rag_engine1.answer_with_citations(
            question=q, screenshot_path=tmp_img, k=k, min_conf=min_conf,
            model="gpt-4o-mini", debug=True
        )

        conf = max([c["score"] for c in out.get("contexts", [])], default=0.0)
        record = {
            "qid": str(uuid.uuid4()),  # ← unique id per Q&A
            "question": q,
            "answer": out.get("answer", ""),
            "debug": out.get("page_scores", {}),
            "contexts": out.get("contexts", []),
            "img_path": tmp_img,
            "confidence": conf,
            "feedback": None,   # initially no feedback
        }
        st.session_state.chat_history.append(record)
        append_log({**record, "img_path": bool(tmp_img)})
        st.session_state.uploader_key += 1
        st.session_state.submitted = False
        st.rerun()

# =============================================================================
# INSTRUCTOR DASHBOARD (with Feedback Analytics)
# =============================================================================
elif mode == "Instructor Dashboard" and st.session_state.authenticated:
    st.markdown("## 🧑‍🏫 Instructor Dashboard")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

    df = read_logs_df()
    if df.empty:
        st.info("No logs yet — ask some questions first.")
        st.stop()

    # Filters
    st.markdown("### 📅 Filters")
    min_d, max_d = df["date"].min(), df["date"].max()
    d_range = st.date_input("Select Date Range", value=(min_d, max_d))
    d_from, d_to = d_range if isinstance(d_range, tuple) else (min_d, max_d)
    conf_filter = st.slider("Min confidence filter", 0.0, 1.0, 0.0, 0.01)
    fb_filter = st.selectbox("Feedback filter", ["all", "helpful only", "unclear only"])

    # Apply filters
    df = df[(df["date"] >= d_from) & (df["date"] <= d_to) & (df["top_score"] >= conf_filter)]
    if fb_filter == "helpful only":
        df = df[df["feedback"] == "helpful"]
    elif fb_filter == "unclear only":
        df = df[df["feedback"] == "unclear"]

    if df.empty:
        st.warning("No data after filters.")
        st.stop()

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
                    .encode(
                        x="Count:Q",
                        y=alt.Y("Phrase:N", sort='-x', axis=alt.Axis(labelFontSize=12, labelLimit=300)),
                        tooltip=["Phrase", "Count"]
                    )
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
            .encode(x="Mentions:Q", y=alt.Y("Slide:N", sort='-x'),
                    tooltip=["Slide", "Mentions"])
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
                if m:
                    all_refs.append((m.group(1), page))
    if all_refs:
        refs_df = pd.DataFrame(all_refs, columns=["Chapter", "Page"]).drop_duplicates()
        slide_counts = (refs_df["Chapter"]
                        .value_counts()
                        .reindex(["C1","C2","C3","C4","C5","C6"])
                        .fillna(0).astype(int).reset_index())
        slide_counts.columns = ["Chapter", "UniqueSlides"]
        st.altair_chart(
            alt.Chart(slide_counts)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#3b82f6")
            .encode(
                x=alt.X("Chapter:N", sort=["C1","C2","C3","C4","C5","C6"], title="Chapter"),
                y=alt.Y("UniqueSlides:Q", title="Unique Slides"),
                tooltip=["Chapter", "UniqueSlides"]
            )
            .properties(height=350, title="Unique Slides Referenced per Chapter")
            .configure_axis(labelColor="#e5e7eb", titleColor="#e5e7eb"),
            use_container_width=True
        )
    else:
        st.caption("No slide references yet.")

    st.markdown("---")

    # ❗ Hardest / Low-rated Questions
    st.markdown("### ❗ Low-Rated & Low-Confidence Responses")
    tabA, tabB = st.tabs(["👎 Unclear (by time)", "Lowest Confidence (top 10)"])
    with tabA:
        unclear = df[df["feedback"] == "unclear"][["ts","question","top_score","answer"]]
        if unclear.empty:
            st.caption("No 'unclear' feedback yet.")
        else:
            unclear = unclear.sort_values("ts", ascending=False)
            unclear = unclear.rename(columns={"ts":"Time","top_score":"Confidence"})
            st.dataframe(unclear, use_container_width=True)
    with tabB:
        low_conf = df.sort_values("top_score").head(10)[["question","top_score","answer","date"]]
        st.dataframe(low_conf.rename(columns={"top_score":"Confidence"}), use_container_width=True)

    st.markdown("---")

    # 👍/👎 Feedback distribution
    st.markdown("### 👍/👎 Feedback Distribution")
    fb_counts = (df["feedback"]
                 .value_counts()
                 .reindex(["helpful","unclear","none"])
                 .fillna(0).astype(int)
                 .reset_index())
    fb_counts.columns = ["Feedback","Count"]
    st.altair_chart(
        alt.Chart(fb_counts)
        .mark_bar(color="#34d399")
        .encode(x=alt.X("Feedback:N", title="Type"),
                y=alt.Y("Count:Q", title="Count"),
                tooltip=["Feedback","Count"])
        .properties(height=300),
        use_container_width=True
    )

    st.markdown("---")
    
        # 🧠 Word Cloud — Most Frequent Terms in Questions
    st.markdown("### ☁️ Common Words in Student Questions")
    st.caption("Shows most frequent keywords from all student questions.")

    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        text_data = " ".join(df["question"].dropna().astype(str))
        if text_data.strip():
            wc = WordCloud(
                width=1000,
                height=500,
                background_color="black",
                colormap="cool",
                max_words=150
            ).generate(text_data)

            st.image(wc.to_array(), caption="Top Keywords from Student Questions", use_container_width=True)
        else:
            st.info("No questions available yet to generate a word cloud.")
    except Exception as e:
        st.warning(f"WordCloud generation skipped (install 'wordcloud' & 'matplotlib'): {e}")

    # 🕒 Engagement Over Time
    st.markdown("### 🕒 Engagement Over Time")
    daily = df.groupby("date").size().reset_index(name="Questions")
    st.altair_chart(
        alt.Chart(daily).mark_area(color="#20c997")
        .encode(x="date:T", y="Questions:Q", tooltip=["date:T", "Questions:Q"])
        .properties(height=300),
        use_container_width=True
    )