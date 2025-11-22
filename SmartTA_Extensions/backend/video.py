"""
Video Processing Module
Handles FFmpeg operations, video clips, playback, and subtitles.
Copied from app.py without modifications.
"""

import os
import re
import time
import datetime
import subprocess
import logging
import streamlit as st
from utils.config import LECTURE_DIR, CLIPS_DIR, FFMPEG_PATH_DEFAULT
from utils.helpers import get_video_id_from_title

# Constants centralized via utils/config.py; app.py may override FFMPEG_PATH
FFMPEG_PATH = FFMPEG_PATH_DEFAULT  

logger = logging.getLogger(__name__)

# VIDEO UTILITY FUNCTIONS 

def safe_ffmpeg_run(command, error_msg="FFmpeg operation failed"):
    """Run ffmpeg commands safely with error handling."""
    if FFMPEG_PATH is None:
        st.warning("FFmpeg not available - clip generation disabled")
        return False, None
    
    # Replace 'ffmpeg' with actual path
    if command[0] == "ffmpeg":
        command[0] = FFMPEG_PATH
    
    try:
        result = subprocess.run(command, check=True, capture_output=True)
        return True, result
    except subprocess.CalledProcessError as e:
        st.error(f"{error_msg}: {e.stderr.decode()}")
        return False, None

def validate_video_format(file_path):
    """Validate video format and convert if needed."""
    command = ["ffmpeg", "-i", file_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if "Invalid data found" in result.stderr:
            st.error("Invalid video file format")
            return False
        if not result.stderr.strip():
            st.error("Could not read video file")
            return False
        return True
    except Exception as e:
        st.error(f"Error validating video: {str(e)}")
        return False

def convert_to_mp4(input_path, output_path):
    """Convert video to MP4 format."""
    command = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        output_path, "-y"
    ]
    success, _ = safe_ffmpeg_run(command, "Video conversion failed")
    return success

def get_video_duration(video_path):
    """Get video duration using FFmpeg with enhanced error handling."""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
        
    try:
        cmd = ['ffmpeg', '-i', video_path]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=10)
        
        if result.stderr:
            duration_match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})", result.stderr)
            if duration_match:
                try:
                    h, m, s = map(int, duration_match.groups())
                    duration = h * 3600 + m * 60 + s
                    if duration > 0:
                        return duration
                except ValueError:
                    logger.error("Invalid duration format in FFmpeg output")
            else:
                logger.warning(f"No duration found in FFmpeg output for {video_path}")
        else:
            logger.warning(f"No FFmpeg output for {video_path}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg timeout while getting duration for {video_path}")
    except Exception as e:
        logger.error(f"Error getting video duration for {video_path}: {str(e)}")
    
    return None

