"""Shared utilities for SmartTA unified app."""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from . import config


def aub_logo_path() -> Optional[str]:
    """Return the preferred AUB logo path if available."""
    candidates = [
        config.SMARTTA_RAG_DIR / "UI" / "aub_logo.png",
        config.SMARTTA_RAG_DIR / "aub_logo.png",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _tuple_key_dict_to_str_key(data: Dict[Any, Any]) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    for key, value in (data or {}).items():
        if isinstance(key, (tuple, list)):
            key_str = f"{key[0]}:{key[1]}" if len(key) > 1 else str(key[0])
        else:
            key_str = str(key)
        converted[key_str] = value
    return converted


def _sanitize_debug(payload: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    if not payload:
        return cleaned
    if "text_openai" in payload:
        cleaned["text_openai"] = {
            str(k): float(v) for k, v in payload["text_openai"].items()
        }
    if "img2img" in payload:
        cleaned["img2img"] = _tuple_key_dict_to_str_key(payload["img2img"])
    if "text2img" in payload:
        cleaned["text2img"] = _tuple_key_dict_to_str_key(payload["text2img"])
    return cleaned


def _write_log_line(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def append_chat_log(entry: Dict[str, Any]) -> None:
    """Persist chat payload to both the local and legacy log locations."""
    payload = dict(entry)
    payload.setdefault("qid", str(uuid.uuid4()))
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["debug"] = _sanitize_debug(payload.get("debug", {}))
    payload["contexts"] = [
        {
            "pdf": ctx.get("pdf"),
            "page": ctx.get("page"),
            "score": float(ctx.get("score", 0.0)),
        }
        for ctx in payload.get("contexts", []) or []
    ]
    if payload.get("img_path"):
        payload["img_path"] = True
    _write_log_line(config.LOCAL_CHAT_LOG, payload)


def update_feedback_entry(qid: str, feedback: str) -> bool:
    """Update feedback field for a stored chat entry across logs."""
    if not qid:
        return False
    updated = False
    path = config.LOCAL_CHAT_LOG
    if not path.exists():
        return False
    try:
        lines = []
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if data.get("qid") == qid:
                data["feedback"] = feedback
                changed = True
            lines.append(json.dumps(data, ensure_ascii=False))
        if changed:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            updated = True
    except Exception:
        return False
    return updated


def read_logs_df() -> pd.DataFrame:
    """Load chat logs as a dataframe with derived columns."""
    rows = []
    path = config.LOCAL_CHAT_LOG
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "qid" not in df.columns:
        df["qid"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["ts"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    tz_mask = df["ts"].isna() & df["timestamp"].notna()
    if tz_mask.any():
        tz_parsed = pd.to_datetime(df.loc[tz_mask, "timestamp"], errors="coerce", utc=True)
        try:
            tz_parsed = tz_parsed.dt.tz_convert(None)
        except Exception:
            pass
        df.loc[tz_mask, "ts"] = tz_parsed
    df["date"] = df["ts"].dt.date

    def _top(entries: Iterable[Dict[str, Any]]) -> float:
        try:
            return max([float(c.get("score", 0.0)) for c in (entries or [])] + [0.0])
        except Exception:
            return 0.0

    df["top_score"] = df.get("contexts", []).apply(_top)
    if "feedback" not in df.columns:
        df["feedback"] = None
    df["feedback"] = df["feedback"].fillna("none")
    df = df.sort_values("ts").drop_duplicates(subset=["qid"], keep="last")
    return df


def sanitize_question(question: str) -> str:
    """Normalize whitespace and length for questions."""
    if not isinstance(question, str):
        return ""
    question = question.replace("\r", " ")
    question = "".join(ch for ch in question if ch.isprintable() or ch in "\n\t ")
    import re as _re

    return _re.sub(r"\s+", " ", question).strip()[:1000]


@contextmanager
def ext_cwd():
    """Temporarily switch to the SmartTA extensions directory."""
    old = Path.cwd()
    os.chdir(config.SMARTTA_EXT_DIR)
    try:
        yield
    finally:
        os.chdir(old)
