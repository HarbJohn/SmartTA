"""Type definitions for SmartTA frontend components."""

from typing import TypedDict, Optional, List


class Segment(TypedDict):
    """Transcript segment with timing and metadata."""
    lecture: str
    text: str
    start: float
    end: float
    category: Optional[str]


class SearchResult(Segment):
    """Search result segment with optional score fields."""
    score: Optional[float]
    kw_overlap: Optional[float]


class MaterialFile(TypedDict):
    """File metadata for course materials."""
    filename: str
    size: int
    type: str
    note: str


class MaterialSection(TypedDict):
    """Course material section with multiple files."""
    id: str
    title: str
    description: str
    order: int
    files: List[MaterialFile]
    created_date: str


class LectureInfo(TypedDict):
    """YouTube lecture metadata."""
    title: str
    youtube_url: str
    video_id: str
    category: str
    professor_note: str
    added_date: str
    id: str


class YouTubeData(TypedDict):
    """Complete YouTube lectures database structure."""
    lectures: dict[str, LectureInfo]
    course_materials: List[MaterialSection]
