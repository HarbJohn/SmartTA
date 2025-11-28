"""Lecture processing operations: transcription, indexing, and data persistence.

This module handles the backend operations for processing new lectures,
including YouTube audio download, Whisper transcription, and FAISS indexing.
"""
import os
import json
import time
import datetime
import logging

import streamlit as st
import faiss

from SmartTA_Extensions.utils.helpers import (
    load_youtube_lectures,
    save_youtube_lectures,
    extract_youtube_id,
    delete_lecture_file,
)
from SmartTA_Extensions.backend.transcription import (
    download_youtube_audio,
    load_whisper_model,
)
from SmartTA_Extensions.backend.indexing import load_sentence_transformer
from SmartTA_Extensions.frontend.professor.lecture_forms import normalize_lecture_id

logger = logging.getLogger(__name__)

# DATA PERSISTENCE

def save_lecture_metadata(
    lecture_id: str,
    lecture_title: str,
    youtube_url: str,
    video_id: str,
    selected_category: str,
    professor_note: str
) -> None:
    """Save lecture metadata to database."""
    yt_data = load_youtube_lectures()
    if "lectures" not in yt_data:
        yt_data["lectures"] = {}
    
    yt_data["lectures"][lecture_id] = {
        "title": lecture_title,
        "youtube_url": youtube_url,
        "video_id": video_id,
        "category": selected_category or "Uncategorized",
        "professor_note": professor_note if professor_note else "",
        "added_date": datetime.datetime.now().isoformat(),
        "id": lecture_id,
    }
    save_youtube_lectures(yt_data)


def save_transcription_results(
    lecture_title: str,
    result: dict,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str
) -> list[dict]:
    """Save transcription results and return segments."""
    segments = [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg["text"]).strip()
        }
        for seg in result["segments"]
        if seg.get("text")
    ]
    
    # Save full transcript
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{lecture_title}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result.get("text", "").strip())
    
    # Save segments
    if os.path.exists(META_SEGMENTS):
        with open(META_SEGMENTS, "r", encoding="utf-8") as f:
            all_segments = json.load(f)
    else:
        all_segments = {}
    
    all_segments[lecture_title] = segments
    with open(META_SEGMENTS, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2)
    
    return segments

# TRANSCRIPTION WORKFLOW

def transcribe_audio(
    audio_path: str,
    lecture_title: str,
    progress_container,
    progress_bar
) -> dict | None:
    """Transcribe audio using Whisper model."""
    import warnings
    
    with progress_container:
        st.info(f"Step 2/3: Transcribing '{lecture_title}' (this may take 2-5 minutes)...")
        st.caption("Processing audio with Whisper AI...")
    
    model = load_whisper_model()
    if not model:
        st.error("Failed to load Whisper model")
        return None
    
    try:
        # Suppress CUDA availability warning - we intentionally use CPU for portability
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Performing inference on CPU when CUDA is available.*")
            result = model.transcribe(
                audio_path,
                verbose=False,
                fp16=False,  
                temperature=0,
                best_of=1,
                beam_size=1,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True,
                word_timestamps=False,
            )
        progress_bar.progress(66)
        return result
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        logger.error(f"Transcription error: {e}")
        return None


def index_new_lecture(
    segments: list,
    lecture_title: str,
    category: str,
    META_PATH: str,
    INDEX_PATH: str,
) -> None:
    """Index newly added lecture segments."""
    embedder = load_sentence_transformer()
    if not embedder:
        return
    try:
        if os.path.exists(INDEX_PATH):
            idx = faiss.read_index(INDEX_PATH)
        else:
            quantizer = faiss.IndexFlatL2(384)
            idx = faiss.IndexIVFFlat(quantizer, 384, min(100, len(segments) + 1))
            texts = [s["text"] for s in segments]
            embeddings = embedder.encode(texts, convert_to_numpy=True)
            idx.train(embeddings)
        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = []
        texts = [s["text"] for s in segments]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        idx.add(embeddings)
        for s in segments:
            meta.append({
                "lecture": lecture_title,
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
                "category": category or "Uncategorized",
            })
        faiss.write_index(idx, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.error(f"Error indexing new lecture: {e}")
        st.warning("Lecture added but indexing will complete on next page load")

# MAIN PROCESSING PIPELINE

def process_new_lecture(
    lecture_title: str,
    youtube_url: str,
    selected_category: str,
    custom_lecture_id: str,
    professor_note: str,
    DATA_DIR: str,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str,
    META_PATH: str,
    INDEX_PATH: str,
) -> None:
    """Process and transcribe a new lecture."""
    # Validation
    if not lecture_title:
        st.error("Please enter a lecture title")
        return
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        return
    if not selected_category:
        st.error("Please select or create a category")
        return

    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please use a valid YouTube link (e.g., https://youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID)")
        return

    with st.spinner(f"Processing '{lecture_title}'..."):
        try:
            # Generate lecture ID
            lecture_id = normalize_lecture_id(
                custom_lecture_id.strip() if custom_lecture_id and custom_lecture_id.strip() else lecture_title
            )
            
            # Check conflicts again (defensive)
            yt_data = load_youtube_lectures()
            if lecture_id in yt_data.get("lectures", {}):
                st.error(f"❌ Lecture ID '{lecture_id}' already exists. Choose a different Custom Lecture ID or title.")
                return
            if any(info.get("title", "").lower().strip() == lecture_title.lower().strip() 
                    for info in yt_data.get("lectures", {}).values()):
                st.error(f"❌ A lecture with title '{lecture_title}' already exists. Choose a different title.")
                return

            # Save metadata
            save_lecture_metadata(
                lecture_id, lecture_title, youtube_url, 
                video_id, selected_category, professor_note
            )

            # Progress tracking
            progress_container = st.container()
            with progress_container:
                st.info("Step 1/3: Downloading audio from YouTube...")
                progress_bar = st.progress(0)
                progress_bar.progress(10)

            # Download audio
            audio_path = os.path.join(DATA_DIR, f"temp_{lecture_id}.mp3")
            if not download_youtube_audio(youtube_url, audio_path):
                st.error("Failed to download audio from YouTube. Check the URL and try again.")
                delete_lecture_file(lecture_id)
                return
            
            progress_bar.progress(33)
            
            # Transcribe
            result = transcribe_audio(audio_path, lecture_title, progress_container, progress_bar)
            if not result or "segments" not in result:
                st.error("Transcription failed - no segments produced")
                os.remove(audio_path)
                return
            
            # Save results
            with progress_container:
                st.info("Step 3/3: Indexing for search...")
            
            segments = save_transcription_results(
                lecture_title, result, TRANSCRIPT_DIR, META_SEGMENTS
            )
            progress_bar.progress(90)
            
            # Cleanup and index
            os.remove(audio_path)
            progress_bar.progress(95)
            
            st.info("Indexing new lecture for search...")
            index_new_lecture(
                segments, lecture_title, selected_category, 
                META_PATH, INDEX_PATH
            )
            progress_bar.progress(100)
            
            # Success
            st.success(f"✅ Lecture '{lecture_title}' added and indexed! ({len(segments)} segments)")
            st.info("Reloading page...")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"Error adding lecture: {str(e)}")
            logger.error(f"Error adding YouTube lecture: {e}")
