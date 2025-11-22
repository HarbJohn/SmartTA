"""Frontend CSS styles for SmartTA.
Centralizes all custom CSS/styling to keep app.py clean.
No logic changes; extracted verbatim from app.py inline markdown blocks.
"""
import streamlit as st


def apply_base_styles():
    """Apply base responsive CSS and component styling (mobile, buttons, containers)."""
    st.markdown("""
<style>
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        iframe {
            width: 100% !important;
            height: 300px !important;
        }
    }
    
    /* Improved button styling */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Better download buttons */
    .stDownloadButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 6px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1em;
    }
    
    /* Keyboard shortcut hints */
    .shortcut-hint {
        background: #f0f2f6;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    
    /* Better video container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def apply_search_ui_styles():
    """Apply search UI styles (dark theme, inputs, cards, animations, confetti, toasts, skeletons)."""
    st.markdown(r"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* FORCE COMPLETE DARK THEME - NO WHITE BLOCKS */
    :root{
        --bg-primary: #0d1117;
        --bg-secondary: #0f1724;
        --bg-card: rgba(18,25,35,0.95);
        --text-primary: #e6eef8;
        --text-secondary: #b9c6d8;
        --border-color: rgba(255,255,255,0.06);
        --accent-1: #6d28d9;
        --accent-2: #2e7cf6;
    }

    .stApp{ background:var(--bg-primary); color:var(--text-primary); font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Force all markdown divs to dark */
    div[data-testid="stMarkdownContainer"] { background: transparent !important; }

    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stTextInput textarea {
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px; padding:0.8rem 1rem;
    }
    input::placeholder, textarea::placeholder { color: rgba(230,238,248,0.45) !important; }

    /* Expanders / cards */
    .stExpander{ background:var(--bg-card) !important; color:var(--text-primary) !important; border:1px solid rgba(255,255,255,0.03); }
    .stExpander[open]{ background: rgba(25,30,40,0.98) !important; border-color: rgba(110,84,255,0.08) !important; }

    /* Buttons */
    .stButton>button{ background: linear-gradient(135deg,var(--accent-2),var(--accent-1)) !important; color:white !important; border:none !important; }

    /* Confetti animation */
    @keyframes confetti-fall {
        0% { transform: translateY(-100vh) rotate(0deg); opacity: 1; }
        100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
    }
    .confetti { position: fixed; width: 10px; height: 10px; background: #667eea; top: -10px; z-index: 9999; animation: confetti-fall 3s linear forwards; pointer-events: none; }

    /* Toast notifications */
    @keyframes toast-slide-in { from { transform: translateX(400px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
    @keyframes toast-slide-out { from { transform: translateX(0); opacity: 1; } to { transform: translateX(400px); opacity: 0; } }
    .toast { position: fixed; top: 80px; right: 20px; background: #1e293b; padding: 1rem 1.5rem; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); z-index: 999999; animation: toast-slide-in 0.3s ease; display: flex; align-items: center; gap: 0.75rem; min-width: 300px; max-width: 400px; color: #e6eef8; }
    .toast.success { border-left: 4px solid #10b981; }
    .toast.info { border-left: 4px solid #3b82f6; }
    .toast.warning { border-left: 4px solid #f59e0b; }

    /* Keyboard shortcut hints */
    .kbd { background: rgba(45,85,180,0.2); padding: 0.25rem 0.5rem; border-radius: 6px; font-family: 'SF Mono', 'Monaco', 'Consolas', monospace; font-size: 0.875rem; font-weight: 600; color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); box-shadow: 0 2px 4px rgba(0,0,0,0.3); }

    /* Loading skeletons */
    @keyframes skeleton-pulse { 0%,100%{ opacity:1; } 50%{ opacity:0.5; } }
    .skeleton { background: linear-gradient(90deg,#1e293b 25%,#334155 50%,#1e293b 75%); background-size:200% 100%; animation: skeleton-pulse 1.5s ease-in-out infinite; border-radius:8px; }

    /* Pulse effect for new results */
    @keyframes pulse-highlight { 0%{ box-shadow:0 0 0 0 rgba(102,126,234,0.7);} 70%{ box-shadow:0 0 0 10px rgba(102,126,234,0);} 100%{ box-shadow:0 0 0 0 rgba(102,126,234,0);} }
    .new-result { animation: pulse-highlight 2s; }

    /* Custom header styling */
    .ta-header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:2rem 2rem 1.5rem 2rem; border-radius:16px; margin-bottom:1.5rem; box-shadow:0 8px 32px rgba(102,126,234,0.25); position:relative; overflow:hidden; }
    .ta-header::before { content:''; position:absolute; top:0; left:0; right:0; bottom:0; background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%); pointer-events:none; }
    .ta-header h2 { color: white; font-weight:700; font-size:1.75rem; margin:0; text-shadow:0 2px 10px rgba(0,0,0,0.15); position:relative; z-index:1; }
    .ta-header p { color: rgba(255,255,255,0.95); margin:0.5rem 0 0 0; font-size:0.95rem; position:relative; z-index:1; }

    /* Enhanced form inputs */
    .stTextInput input { border-radius:12px; border:2px solid #334155; padding:0.875rem 1rem; font-size:1rem; transition:all 0.3s ease; background:var(--bg-secondary); color: var(--text-primary); }
    .stTextInput input:focus { border-color:#667eea; box-shadow:0 0 0 3px rgba(102,126,234,0.2); outline:none; }

    /* Modern button styling */
    .stButton button { border-radius:10px; font-weight:600; padding:0.625rem 1.25rem; transition:all 0.2s ease; border:none; }
    .stButton button[kind="primary"] { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); box-shadow:0 4px 12px rgba(102,126,234,0.3); }
    .stButton button[kind="primary"]:hover { transform: translateY(-2px); box-shadow:0 6px 20px rgba(102,126,234,0.4); }
    .stButton button[kind="secondary"] { background:var(--bg-card); border:2px solid #334155; color:#667eea; }
    .stButton button[kind="secondary"]:hover { background:rgba(102,126,234,0.1); border-color:#667eea; }

    /* Glassmorphism result cards */
    .stExpander { background: rgba(30,41,59,0.8); backdrop-filter: blur(10px); border:1px solid rgba(255,255,255,0.03); border-radius:12px; margin:0.75rem 0; box-shadow:0 4px 16px rgba(0,0,0,0.3); transition:all 0.3s ease; }
    .stExpander:hover { box-shadow:0 8px 24px rgba(0,0,0,0.5); transform: translateY(-2px); }
    .stExpander > summary { font-weight:600; padding:1rem 1.25rem; color:var(--text-primary); }
    .stExpander[open] { background:rgba(30,41,59,0.95); border-color: rgba(102,126,234,0.2); }

    /* Radio button styling */
    .stRadio > div { background: rgba(102,126,234,0.08); padding:0.5rem; border-radius:10px; }
    .stRadio label { background:var(--bg-card); padding:0.5rem 1rem; border-radius:8px; font-weight:500; transition:all 0.2s ease; cursor:pointer; color: var(--text-primary); }
    .stRadio label:hover { background: rgba(102,126,234,0.15); }

    /* Metrics styling */
    .stMetric { background: rgba(102,126,234,0.08); padding:0.75rem; border-radius:8px; border-left:3px solid #667eea; }
    .stMetric label { font-weight:600; color:var(--text-secondary); }

    /* Progress indicators */
    .stProgress > div > div { background: linear-gradient(90deg,#667eea 0%,#764ba2 100%); border-radius:10px; }

    /* Info/Warning boxes */
    .stAlert { border-radius:12px; border-left:4px solid; background: rgba(30,41,59,0.8) !important; }

    /* Sidebar improvements */
    .css-1d391kg { background: var(--bg-primary); }

    /* Checkbox styling */
    .stCheckbox label { font-weight:500; color:var(--text-primary); }

    /* Caption text */
    .stCaptionContainer { color:var(--text-secondary); font-size:0.875rem; }

    /* Divider */
    hr { border:none; height:2px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent); margin:2rem 0; }

    /* Markdown text in results - highlight styling */
    .stMarkdown strong { background: rgba(245,158,11,0.2); color: #fbbf24; padding:0.125rem 0.25rem; border-radius:3px; font-weight:600; }

    /* Animation for score badges */
    @keyframes slideIn { from { opacity:0; transform: translateX(-10px); } to { opacity:1; transform: translateX(0); } }
    .stMetric { animation: slideIn 0.3s ease; }

    </style>
    """, unsafe_allow_html=True)


def render_search_header():
    """Render styled 'Ask the TA' header with gradient background."""
    st.markdown("""
<div class="ta-header">
    <h2>💬 Ask the TA</h2>
</div>
""", unsafe_allow_html=True)


def apply_all_styles():
    """Convenience function to apply all SmartTA styles at once."""
    apply_base_styles()
    apply_search_ui_styles()
