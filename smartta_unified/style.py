"""Static CSS for the SmartTA app."""

BASE_STYLES = """
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
.typing { display:inline-block; padding:10px 14px; border-radius:12px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.06);}
.dot{ height:6px; width:6px; background:#9aa2b1; border-radius:999px; display:inline-block; margin:0 2px;
  animation: blink 1.3s infinite;}
.dot:nth-child(2){ animation-delay:.2s;} .dot:nth-child(3){ animation-delay:.4s;}
@keyframes blink { 0%{opacity:.25} 50%{opacity:1} 100%{opacity:.25} }
.metric-card{ background:linear-gradient(135deg,#1e293b,#0f172a); border-radius:14px; padding:18px; text-align:center; box-shadow:0 0 14px rgba(0,0,0,.3); }
.metric-card h2{ margin:0; font-size:1.8rem; color:#00C6FF;}
.metric-card p{ margin:4px 0 0; color:var(--muted);}
.login-box{ background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.15); padding:25px; border-radius:12px; width:350px; margin:auto; text-align:center;}
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
.retrieval-row{display:flex;align-items:center;gap:.65rem;margin:8px 0;font-size:.93rem;color:#e2e8f0;}
.pdf-pill{background:linear-gradient(120deg,#fbbf24,#f97316);color:#111827;padding:2px 10px;border-radius:10px;font-weight:700;}
.page-pill{color:#c4cedd;font-weight:600;}
.arrow-icon{color:#f87171;font-weight:700;}
.score-pill{background:linear-gradient(120deg,#059669,#22c55e);color:#052e16;padding:2px 12px;border-radius:999px;font-weight:700;min-width:64px;text-align:center;}
.score-context{color:#9da8c5;font-size:.88rem;}
.score-inline{color:#22c55e;font-weight:700;}
.bullet-dot{color:#f87171;font-size:1.2rem;}
.feedback-btn{display:inline-block;width:100%;}
.feedback-btn button{width:100%;border:none;border-radius:999px;font-weight:700;font-size:.95rem;padding:6px 14px;color:#fff;box-shadow:0 10px 20px rgba(0,0,0,.25);}
.feedback-btn.helpful button{background:linear-gradient(120deg,#4338ca,#8b5cf6);}
.feedback-btn.unclear button{background:linear-gradient(120deg,#be185d,#f43f5e);}
.feedback-status{margin-top:.5rem;font-weight:600;font-size:.85rem;padding:4px 12px;display:inline-block;border-radius:999px;}
.feedback-status.helpful-tag{background:rgba(79,70,229,.2);color:#a5b4fc;}
.feedback-status.unclear-tag{background:rgba(248,113,113,.2);color:#fecdd3;}
.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid rgba(255,255,255,.08);gap:1rem;}
.stTabs [data-baseweb="tab-list"] button{color:#94a3b8;font-weight:600;font-size:.92rem;}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"]{color:#fb7185;border-bottom:2px solid #fb7185;border-radius:0;}
.stTabs [data-baseweb="tab-list"] button:focus{outline:none;box-shadow:none;}
.retrieval-row .pdf-pill strong{color:#0f172a;}
</style>
"""
