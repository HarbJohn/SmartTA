"""
Transcription Module
Copies transcription-related functions from app.py without modification.
No logic changes. Pending later extraction of shared constants/utilities into utils/config.py.
"""

import os
import json
import time
import random
import logging
import streamlit as st
"""Whisper import is handled inside load_whisper_model to avoid module import failure
when the package isn't installed. This enables the app to continue running and show
an actionable error instead of crashing at import time.
"""
import yt_dlp  # for YouTube downloads
from utils.config import (
    DATA_DIR,
    MODEL_DIR,
    TRANSCRIPT_DIR,
    LECTURE_DIR,
    META_SEGMENTS,
    MAX_RETRIES,
    RETRY_DELAY,
    WHISPER_MODEL_SIZE,
)
from utils.helpers import get_all_lectures, process_with_progress
from backend.video import get_video_duration

# Constants centralized via utils/config.py

logger = logging.getLogger(__name__)

# --------------------
# TRANSCRIPTION FUNCTIONS (Copied verbatim from app.py)
# --------------------

def download_youtube_audio(youtube_url, output_path):
    """Download audio from YouTube video using yt-dlp with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': output_path.replace('.mp3', ''),
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            return True
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"YouTube download failed after {MAX_RETRIES} attempts: {e}")
                return False
            wait_time = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"YouTube download attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load and cache the Whisper model."""
    model_path = os.path.join(MODEL_DIR, f"whisper_{WHISPER_MODEL_SIZE}")
    try:
        # Import whisper lazily to avoid module import errors at app startup
        try:
            import whisper as _whisper  # type: ignore
        except Exception as import_err:
            st.error(
                "Whisper is not available. Please install it first.\n\n"
                "If installation failed on Python 3.13, we switched to installing from GitHub in requirements.txt.\n"
                "Try: pip install git+https://github.com/openai/whisper.git#egg=openai-whisper"
            )
            logging.getLogger(__name__).error(f"Whisper import error: {import_err}")
            return None

        with st.spinner("Loading Whisper model..."):
            # Force CPU usage for Whisper due to CUDA compatibility issues
            model = _whisper.load_model(WHISPER_MODEL_SIZE, download_root=model_path, device="cpu")
            return model
    except Exception as e:
        st.error(f"Whisper model failed to load: {e}")
        st.info("Tip: Check internet connection for first-time model download, or verify 'data/models' directory permissions.")
        logger.error(f"Whisper model load error: {e}")
        return None

def safe_operation(func):
    """Decorator for safe operation execution with retry logic."""
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    st.error(f"Operation failed after {MAX_RETRIES} attempts: {str(e)}")
                    raise e
                wait_time = RETRY_DELAY * (2 ** attempt)
                st.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
    return wrapper


def transcribe_new_videos():
    """Transcribes new lectures (with timestamps) using Whisper."""
    model = load_whisper_model()
    if not model:
        return {}, 0

    if os.path.exists(META_SEGMENTS):
        with open(META_SEGMENTS, "r", encoding="utf-8") as f:
            all_segments = json.load(f)
    else:
        all_segments = {}

    # get_all_lectures is expected to be available in the calling context (app.py)
    lectures = get_all_lectures()  # noqa: F821

    # Check both segments metadata and existing transcript files
    transcribed_lectures = set()
    if os.path.exists(META_SEGMENTS):
        transcribed_lectures.update(all_segments.keys())

    # Check existing transcripts (handle both old flat structure and new subfolder structure)
    for transcript in os.listdir(TRANSCRIPT_DIR):
        if transcript.endswith('.txt'):
            base_name = os.path.splitext(transcript)[0]
            transcribed_lectures.add(base_name)

    # Filter new videos: check if basename (filename without path) matches existing transcripts
    new_videos = []
    for lec in lectures:
        lec_name = os.path.splitext(lec)[0]
        lec_basename = os.path.basename(lec_name)
        if lec_name not in transcribed_lectures and lec_basename not in transcribed_lectures:
            new_videos.append(lec)

    if not new_videos:
        if transcribed_lectures and not all_segments:
            st.info("🔄 Found existing transcripts but missing metadata. Rebuilding...")
            for lec_name in transcribed_lectures:
                transcript_path = os.path.join(TRANSCRIPT_DIR, f"{lec_name}.txt")
                if os.path.exists(transcript_path):
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        all_segments[lec_name] = [{"start": 0, "end": 100, "text": text}]
            with open(META_SEGMENTS, "w", encoding="utf-8") as f:
                json.dump(all_segments, f, indent=2)
        return all_segments, 0

    st.info(f"Found {len(new_videos)} new lecture(s). Starting transcription...")

    def retry_with_backoff(func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1} of {MAX_RETRIES}")
                result = func(*args, **kwargs)
                if result is None or (isinstance(result, tuple) and any(r is None for r in result)):
                    raise ValueError("Function returned invalid result")
                return result
            except Exception as e:
                last_error = e
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise last_error
                base_wait = RETRY_DELAY * (2 ** attempt)
                jitter = base_wait * 0.1 * random.random()
                wait_time = base_wait + jitter
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                st.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

    def transcribe_lecture(lec):
        """Transcribe a lecture with retry logic and progress tracking."""
        name = os.path.splitext(lec)[0]
        logger.info(f"Starting transcription of {lec}")
        video_path = os.path.join(LECTURE_DIR, lec)
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        try:
            def _transcribe():
                with st.spinner(f"Transcribing {lec}..."):
                    try:
                        duration = get_video_duration(video_path)  # noqa: F821
                        model.to("cpu")
                        result = model.transcribe(
                            video_path,
                            initial_prompt="This is a lecture video transcription.",
                            verbose=False,
                            fp16=False,
                            temperature=0,
                            best_of=1,
                            task="transcribe"
                        )
                        if not result or "segments" not in result:
                            raise ValueError("Transcription failed to produce segments")
                        segments = [
                            {
                                "start": float(seg["start"]),
                                "end": float(seg["end"]),
                                "text": str(seg["text"]).strip()
                            }
                            for seg in result["segments"]
                            if seg.get("text") and seg.get("start") is not None and seg.get("end") is not None
                        ]
                        if not segments:
                            raise ValueError("No valid segments produced")
                        return result.get("text", "").strip(), segments
                    except Exception as e:
                        logger.error(f"Transcription error: {str(e)}")
                        raise
            transcript_text, segments = retry_with_backoff(_transcribe)
            try:
                transcript_path = os.path.join(TRANSCRIPT_DIR, f"{name}.txt")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                logger.info(f"Successfully transcribed {lec}")
                st.success(f"Transcribed {lec}")
            except Exception as e:
                logger.error(f"Failed to save transcript/subtitles for {lec}: {e}")
                st.error(f"Failed to save transcript for {lec}")
            return name, segments
        except Exception as e:
            logger.error(f"Failed to transcribe {lec}: {e}")
            st.error(f"❌ Transcription failed for {lec}: {str(e)[:100]}")
            st.info("💡 Common fixes:\n- Check internet connection for YouTube downloads\n- Ensure FFmpeg is installed\n- Verify sufficient disk space in data/ folder\n- Check data/app.log for detailed error")
            return None

    # process_with_progress is expected to be available in calling context (app.py)
    results = process_with_progress(  # noqa: F821
        new_videos,
        transcribe_lecture,
        "Transcribing lectures"
    )
    for result in results:
        if result:
            name, segments = result
            all_segments[name] = segments
    with open(META_SEGMENTS, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, indent=2)
    st.success("Transcription complete.")
    return all_segments, len(new_videos)
