"""Central configuration and dependency wiring for the SmartTA unified app."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# Environment defaults

_ENV_DEFAULTS = {
    "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _key, _val in _ENV_DEFAULTS.items():
    os.environ.setdefault(_key, _val)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMARTTA_RAG_DIR = PROJECT_ROOT / "SmartTA_RAG"
SMARTTA_EXT_DIR = PROJECT_ROOT / "SmartTA_Extensions"

for _path in (SMARTTA_RAG_DIR, SMARTTA_EXT_DIR):
    if _path.exists():
        sys.path.insert(0, str(_path))


def _load_env_file(path: Path) -> None:
    """Best-effort .env loader."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=str(path), override=False)
        return
    except Exception:
        pass

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(PROJECT_ROOT / ".env")
_load_env_file(SMARTTA_RAG_DIR / ".env")

# External modules
try:  # rag engine (new package)
    from smartta_unified import rag as rag_engine1  # type: ignore
    RAG_IMPORT_ERROR = None
except Exception as exc:  #   handled at runtime
    try:
        import rag_engine1  # type: ignore
        RAG_IMPORT_ERROR = exc
    except Exception as legacy_exc:
        rag_engine1 = None
        RAG_IMPORT_ERROR = legacy_exc

try:  # Streamlit extensions
    from SmartTA_Extensions.frontend.tabs.student import render_student_tab  # type: ignore
    from SmartTA_Extensions.frontend.tabs.professor import render_professor_tab  # type: ignore
    from SmartTA_Extensions.utils.helpers import (  # type: ignore
        get_youtube_lectures_categorized,
        load_youtube_lectures,
    )
    from SmartTA_Extensions.backend.indexing import reindex_if_needed  # type: ignore
except Exception as exc:  #  no cover - runtime feedback only
    render_student_tab = None
    render_professor_tab = None
    load_youtube_lectures = None
    get_youtube_lectures_categorized = None
    reindex_if_needed = None
    EXTENSION_IMPORT_ERROR = exc
else:
    EXTENSION_IMPORT_ERROR = None

try:  # Optional deps
    import psutil  # type: ignore
except Exception:  #  no cover - optional
    psutil = None

try:
    from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
except Exception:  #  no cover - optional
    CountVectorizer = None  # type: ignore


@dataclass(frozen=True)
class ReportLabDeps:
    SimpleDocTemplate: Any
    Paragraph: Any
    Spacer: Any
    RLImage: Any
    getSampleStyleSheet: Any
    ParagraphStyle: Any
    TA_CENTER: Any
    A4: Any
    inch: Any
    colors: Any
    HRFlowable: Any


try:
    from reportlab.platypus import (  # type: ignore
        HRFlowable,
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )
    from reportlab.lib import colors  # type: ignore
    from reportlab.lib.enums import TA_CENTER  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
    from reportlab.lib.units import inch  # type: ignore

    REPORTLAB_DEPS: Optional[ReportLabDeps] = ReportLabDeps(
        SimpleDocTemplate=SimpleDocTemplate,
        Paragraph=Paragraph,
        Spacer=Spacer,
        RLImage=RLImage,
        getSampleStyleSheet=getSampleStyleSheet,
        ParagraphStyle=ParagraphStyle,
        TA_CENTER=TA_CENTER,
        A4=A4,
        inch=inch,
        colors=colors,
        HRFlowable=HRFlowable,
    )
except Exception:  #  no cover - optional
    REPORTLAB_DEPS = None

# Data directories
APP_DATA_DIR = PROJECT_ROOT / "data"
APP_DATA_DIR.mkdir(exist_ok=True)
LOCAL_CHAT_LOG = APP_DATA_DIR / "chat_logs.ndjson"

RAG_UI_DIR = SMARTTA_RAG_DIR / "UI" / "data"
RAG_UI_DIR.mkdir(parents=True, exist_ok=True)
RAG_UI_CHAT_LOG = RAG_UI_DIR / "chat_logs.ndjson"

EXT_DATA_DIR = SMARTTA_EXT_DIR / "data"
EXT_DATA_DIR.mkdir(exist_ok=True)
EXT_META_SEG = EXT_DATA_DIR / "segments_metadata.json"
EXT_META_PATH = EXT_DATA_DIR / "metadata.json"
EXT_INDEX_PATH = EXT_DATA_DIR / "course.index"
EXT_LOG_PATH = APP_DATA_DIR / "youtube_queries_log.json"
EXT_UPLOADS = EXT_DATA_DIR / "uploaded_files"
EXT_TRANS = EXT_DATA_DIR / "transcripts"
EXT_MODELS = EXT_DATA_DIR / "models"
EXT_LECTURES = EXT_DATA_DIR / "lectures"
for _dir in (EXT_UPLOADS, EXT_TRANS, EXT_MODELS, EXT_LECTURES):
    _dir.mkdir(exist_ok=True)

# Feature flags
ENABLE_RAG = os.getenv("SMARTTA_ENABLE_RAG", "1") not in {"0", "false", "False"}
ENABLE_YOUTUBE = os.getenv("SMARTTA_ENABLE_YOUTUBE", "1") not in {"0", "false", "False"}
RAG_QUERY_LIMIT = int(os.getenv("SMARTTA_RAG_QUERY_LIMIT", "50"))

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "SMARTTA_RAG_DIR",
    "SMARTTA_EXT_DIR",
    "APP_DATA_DIR",
    "LOCAL_CHAT_LOG",
    "RAG_UI_CHAT_LOG",
    "EXT_DATA_DIR",
    "EXT_META_SEG",
    "EXT_META_PATH",
    "EXT_INDEX_PATH",
    "EXT_LOG_PATH",
    "EXT_UPLOADS",
    "EXT_TRANS",
    "EXT_MODELS",
    "EXT_LECTURES",
    # Flags
    "ENABLE_RAG",
    "ENABLE_YOUTUBE",
    "RAG_QUERY_LIMIT",
    # Deps
    "rag_engine1",
    "render_student_tab",
    "render_professor_tab",
    "load_youtube_lectures",
    "get_youtube_lectures_categorized",
    "reindex_if_needed",
    "psutil",
    "CountVectorizer",
    "REPORTLAB_DEPS",
    "RAG_IMPORT_ERROR",
    "EXTENSION_IMPORT_ERROR",
]
