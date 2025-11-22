"""Common helper functions for SmartTA.
Includes YouTube lecture DB utilities, file operations, and UI helpers.
Logic mirrors previous inline implementations in app.py with no behavior changes intended.
"""
import os
import re
import json
import time
import datetime
from typing import Any, Dict, List, Callable, Iterable, Tuple

import streamlit as st

from utils.config import (
    DATA_DIR,
    LECTURE_DIR,
    LECTURES_DIR,
    COURSE_MATERIALS_PATH,
)


def safe_json_load(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def safe_json_write(path: str, data: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def load_youtube_lectures() -> Dict[str, Any]:
    """Load YouTube lectures from data/lectures/*.json and course materials.
    Returns a dict: {"lectures": {id: info, ...}, "course_materials": [...]}.
    """
    lectures: Dict[str, Any] = {}
    try:
        os.makedirs(LECTURES_DIR, exist_ok=True)
        for fname in os.listdir(LECTURES_DIR):
            if fname.endswith(".json"):
                lec_id = os.path.splitext(fname)[0]
                path = os.path.join(LECTURES_DIR, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        # Ensure id present
                        if isinstance(info, dict):
                            info.setdefault("id", lec_id)
                            lectures[lec_id] = info
                except Exception:
                    continue
    except Exception:
        pass

    course_materials = safe_json_load(COURSE_MATERIALS_PATH, [])
    return {"lectures": lectures, "course_materials": course_materials}


def save_youtube_lectures(yt_data: Dict[str, Any]) -> bool:
    """Persist YouTube lectures and course materials to disk.
    - Lectures: data/lectures/<id>.json
    - Course materials: data/course_materials.json
    """
    ok = True
    try:
        os.makedirs(LECTURES_DIR, exist_ok=True)
        for lec_id, info in yt_data.get("lectures", {}).items():
            # Normalize id
            lid = lec_id or info.get("id") or re.sub(r"\s+", "_", info.get("title", "")).strip("_")
            path = os.path.join(LECTURES_DIR, f"{lid}.json")
            data = dict(info)
            data["id"] = lid
            ok = safe_json_write(path, data) and ok
    except Exception:
        ok = False

    ok = safe_json_write(COURSE_MATERIALS_PATH, yt_data.get("course_materials", [])) and ok
    return ok


def get_youtube_lectures_categorized() -> Dict[str, List[str]]:
    """Return mapping: category -> list of lecture IDs."""
    yt = load_youtube_lectures()
    cats: Dict[str, List[str]] = {}
    for lec_id, info in yt.get("lectures", {}).items():
        cat = info.get("category", "Uncategorized") or "Uncategorized"
        cats.setdefault(cat, []).append(lec_id)
    for k in list(cats.keys()):
        cats[k] = sorted(cats[k])
    return cats


def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return ""
    # Patterns: watch?v=ID, youtu.be/ID, embed/ID, shorts/ID
    patterns = [
        r"v=([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"embed/([\w-]{11})",
        r"shorts/([\w-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    # Fallback: last 11-char token
    m = re.search(r"([\w-]{11})(?:\?|&|$)", url)
    return m.group(1) if m else ""


def get_video_id_from_title(title: str) -> str:
    """Find YouTube video ID by lecture title."""
    try:
        yt = load_youtube_lectures()
        for _, info in yt.get("lectures", {}).items():
            if info.get("title") == title:
                return info.get("video_id") or extract_youtube_id(info.get("youtube_url", ""))
    except Exception:
        pass
    return ""


def delete_lecture_file(lecture_id: str) -> bool:
    """Delete a lecture metadata JSON file by id."""
    try:
        path = os.path.join(LECTURES_DIR, f"{lecture_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


def get_all_lectures() -> List[str]:
    """List all .mp4 files in lectures directory (filenames)."""
    try:
        if not os.path.exists(LECTURE_DIR):
            return []
        return sorted([f for f in os.listdir(LECTURE_DIR) if f.endswith(".mp4")])
    except Exception:
        return []


def process_with_progress(
    items: Iterable[Any],
    func: Callable[[Any], Any],
    message: str = "Processing",
) -> List[Any]:
    """Run a function over items with a Streamlit progress bar and status text."""
    items_list = list(items)
    if not items_list:
        return []
    progress = st.progress(0)
    status = st.empty()
    total = len(items_list)
    start = time.time()
    results: List[Any] = []
    for i, it in enumerate(items_list, 1):
        try:
            status.info(f"{message}: {i}/{total}")
            res = func(it)
            results.append(res)
        except Exception as e:
            st.warning(f"Failed on item {i}: {e}")
            results.append(None)
        finally:
            progress.progress(i / total)
    elapsed = time.time() - start
    status.success(f"{message} complete in {elapsed:.1f}s")
    time.sleep(0.2)
    status.empty()
    return results
