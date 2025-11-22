import os
import re
import json
import time
import faiss
import datetime
import logging
from typing import Callable
import streamlit as st

from utils.helpers import (
    load_youtube_lectures,
    save_youtube_lectures,
    extract_youtube_id,
    delete_lecture_file,
    get_all_lectures as helpers_get_all_lectures,
)
from backend.transcription import (
    download_youtube_audio,
    load_whisper_model,
)
from backend.indexing import load_sentence_transformer
from backend.analytics import (
    load_analytics_data,
    render_lecture_frequency_chart,
    render_query_timeline_analysis,
    render_query_patterns_over_time,
)

logger = logging.getLogger(__name__)


def render_professor_tab(
    DATA_DIR: str,
    LECTURES_DIR: str,
    TRANSCRIPT_DIR: str,
    META_SEGMENTS: str,
    META_PATH: str,
    INDEX_PATH: str,
    LOG_PATH: str,
    UPLOADS_DIR: str,
    lecture_categories: dict[str, list[str]],
) -> None:
    """Render the Professor Dashboard tab.
    
    Args:
        DATA_DIR: Root data directory path
        LECTURES_DIR: Directory for individual lecture JSON files
        TRANSCRIPT_DIR: Directory for transcript text files
        META_SEGMENTS: Path to segments metadata JSON
        META_PATH: Path to search metadata JSON
        INDEX_PATH: Path to FAISS index file
        LOG_PATH: Path to query analytics log
        UPLOADS_DIR: Path to uploaded course materials
        lecture_categories: Map of category names to lecture IDs
    """

   
    # Control Panel - Quick Actions
    st.markdown("### ⚙️ Control Panel")
    st.caption("Quick access to system maintenance tasks")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("📊 Clear Analytics", use_container_width=True):
            if os.path.exists(LOG_PATH):
                try:
                    os.remove(LOG_PATH)
                    st.success("✅ Analytics cleared")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear analytics: {e}")
            else:
                st.info("No analytics data to clear")
    
    with col3:
        if st.button("🧹 Clean Index", use_container_width=True):
            try:
                # Remove index entries for deleted lectures
                if os.path.exists(META_PATH):
                    with open(META_PATH, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Get list of existing transcript files
                    existing_transcripts = set()
                    if os.path.exists(TRANSCRIPT_DIR):
                        existing_transcripts = {os.path.splitext(f)[0] for f in os.listdir(TRANSCRIPT_DIR) if f.endswith('.txt')}
                    
                    # Filter metadata to only include existing lectures
                    original_count = len(metadata)
                    metadata = [m for m in metadata if m.get('lecture', '').replace('.txt', '') in existing_transcripts]
                    removed_count = original_count - len(metadata)
                    
                    if removed_count > 0:
                        # Save cleaned metadata
                        with open(META_PATH, 'w', encoding='utf-8') as f:
                            json.dump(metadata, f, indent=2)
                        
                        # Rebuild index
                        embedder = load_sentence_transformer()
                        if embedder and metadata:
                            texts = [m['text'] for m in metadata]
                            embeddings = embedder.encode(texts, show_progress_bar=True)
                            
                            dimension = embeddings.shape[1]
                            index = faiss.IndexFlatL2(dimension)
                            index.add(embeddings)
                            faiss.write_index(index, INDEX_PATH)
                        
                        st.success(f"✅ Cleaned index: removed {removed_count} deleted lecture(s)")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.info("Index is clean - no deleted lectures found")
                else:
                    st.info("No index to clean")
            except Exception as e:
                st.error(f"Failed to clean index: {e}")
                logger.error(f"Clean index error: {e}")
    
    with col4:
        if st.button("⚠️ Clear All Data", use_container_width=True):
            st.session_state['confirm_clear_all'] = True

    
    # Smart Recommendations - Top Confused Topics
    st.markdown("---")
    st.subheader(" Smart Recommendations - Top Confused Topics")

    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                logs = json.load(f)

            if logs:
                topic_analysis = {}
                for log in logs:
                    query = log.get('question', '').lower()
                    results = log.get('results', [])
                    if query and results:
                        words = re.findall(r'\b\w{4,}\b', query)
                        topic = ' '.join(words[:3]) if words else query[:30]
                        if topic not in topic_analysis:
                            topic_analysis[topic] = {'count': 0, 'questions': [], 'locations': []}
                        topic_analysis[topic]['count'] += 1
                        if query not in topic_analysis[topic]['questions']:
                            topic_analysis[topic]['questions'].append(query)
                        for r in results[:3]:
                            location = {
                                'lecture': r['lecture'],
                                'start': r['start'],
                                'end': r['end'],
                                'text': r['text'][:100],
                            }
                            if location not in topic_analysis[topic]['locations']:
                                topic_analysis[topic]['locations'].append(location)

                top_topics = sorted(topic_analysis.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
                if top_topics:
                    st.info(f"📊 Showing {len(top_topics)} most-asked topics from {len(logs)} total queries")
                    for i, (topic, data) in enumerate(top_topics, 1):
                        with st.expander(f" #{i}: {topic.title()} ({data['count']} queries)", expanded=(i <= 2)):
                            st.markdown("**Students asked:**")
                            for q in data['questions'][:5]:
                                st.markdown(f"- *\"{q}\"*")
                            if len(data['questions']) > 5:
                                st.caption(f"... and {len(data['questions']) - 5} more variations")
                            st.markdown("---")
                            st.markdown("** Where it's explained:**")
                            locations_text = ""
                            for loc in data['locations'][:5]:
                                time_str = f"{datetime.timedelta(seconds=int(loc['start']))} - {datetime.timedelta(seconds=int(loc['end']))}"
                                locations_text += f"• **{loc['lecture']}** @ {time_str}\n"
                                locations_text += f"  _{loc['text']}_\n\n"
                            st.markdown(locations_text)
                            share_text = f" {topic.title()}\n\n"
                            share_text += f"Asked by students {data['count']} times\n\n"
                            share_text += "Watch these sections:\n"
                            for loc in data['locations'][:5]:
                                time_str = f"{datetime.timedelta(seconds=int(loc['start']))}"
                                share_text += f"- {loc['lecture']} at {time_str}\n"
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text_area("Copy & Share with Students:", share_text, height=150, key=f"share_{i}")
                            with col2:
                                st.markdown("<br>", unsafe_allow_html=True)
                                st.caption("👆 Copy this text to share with students")
                else:
                    st.info("No queries yet. Students will start asking questions soon!")
            else:
                st.info("No queries logged yet.")
        except Exception as e:
            logger.error(f"Error loading recommendations: {e}")
            st.error("Could not load recommendations")
    else:
        st.info("No query data available yet. Start using the Student Assistant to see recommendations!")

    # --------------------
    # Add YouTube Lecture
    # --------------------
    st.markdown("---")
    st.markdown('<div id="add_lecture"></div>', unsafe_allow_html=True)
    st.subheader("📺 Add YouTube Lecture")
    st.info("Upload your lecture to YouTube as Unlisted, then paste the link here.")

    upload_categories = sorted(lecture_categories.keys()) if lecture_categories else []
    if upload_categories:
        category_options = ["— Select Category —"] + upload_categories + ["+ Create New Category"]
        category_choice = st.selectbox(
            "Category:", options=category_options, index=0, help="Choose which category for this lecture or create a new one"
        )
        if category_choice == "— Select Category —":
            selected_category = None
        elif category_choice == "+ Create New Category":
            new_category = st.text_input("Enter new category name:", placeholder="e.g., Deep Learning")
            selected_category = new_category.strip() if new_category and new_category.strip() else None
        else:
            selected_category = category_choice
    else:
        selected_category = st.text_input("Create first category:", placeholder="e.g., Supervised Learning")
        selected_category = selected_category.strip() if selected_category else None

    lecture_title = st.text_input("Lecture Title:", placeholder="e.g., 01. Introduction to Machine Learning")
    suggested_id = re.sub(r'[^a-zA-Z0-9\s\-]', '', lecture_title).strip()
    suggested_id = re.sub(r'\s+', '_', suggested_id) if suggested_id else ""
    custom_lecture_id = st.text_input(
        "Optional: Custom Lecture ID (filename)",
        value=suggested_id,
        placeholder="e.g., 01_introduction_to_ml",
        help="Auto-suggested from the title. You can override. Used as filename under data/lectures.",
    )
    youtube_url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    # Real-time URL validation feedback
    if youtube_url and not extract_youtube_id(youtube_url):
        st.warning("⚠️ Invalid YouTube URL format. Please use a valid link (e.g., youtube.com/watch?v=... or youtu.be/...)")
    
    professor_note = st.text_area(
        "Professor's Note (optional):",
        placeholder="Key topics: gradient descent, learning rate...",
        help="This note will be shown to students when they view this lecture",
    )

    yt_data_for_validation = load_youtube_lectures()
    existing_ids = set(yt_data_for_validation.get("lectures", {}).keys())
    existing_titles = set(info.get("title", "").lower().strip() for info in yt_data_for_validation.get("lectures", {}).values())
    
    normalized_id = re.sub(r'[^a-zA-Z0-9\s\-]', '', (custom_lecture_id or lecture_title)).strip()
    normalized_id = re.sub(r'\s+', '_', normalized_id).strip('_') if normalized_id else ""
    id_conflict = normalized_id in existing_ids if normalized_id else False
    title_conflict = lecture_title.lower().strip() in existing_titles if lecture_title else False

    col_submit, col_warn = st.columns([1, 2])
    with col_submit:
        can_submit = bool(lecture_title and youtube_url and selected_category and not id_conflict and not title_conflict)
        submit_help = None if can_submit else "Enter title, URL, choose category, ensure unique ID and title"
        trigger_submit = st.button("Add & Transcribe Lecture", type="primary", disabled=not can_submit, help=submit_help)
    with col_warn:
        if id_conflict:
            st.warning(f"⚠️ Lecture ID '{normalized_id}' already exists. Change the Custom Lecture ID.")
        elif title_conflict:
            st.warning(f"⚠️ A lecture with title '{lecture_title}' already exists. Choose a different title.")

    if trigger_submit:
        if not lecture_title:
            st.error("Please enter a lecture title")
        elif not youtube_url:
            st.error("Please enter a YouTube URL")
        elif not selected_category:
            st.error("Please select or create a category")
        else:
            video_id = extract_youtube_id(youtube_url)
            if not video_id:
                st.error("❌ Invalid YouTube URL. Please use a valid YouTube link (e.g., https://youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID)")
            else:
                with st.spinner(f"Processing '{lecture_title}'..."):
                    try:
                        yt_data = load_youtube_lectures()
                        base_id = custom_lecture_id.strip() if custom_lecture_id and custom_lecture_id.strip() else lecture_title
                        lecture_id = re.sub(r'[^a-zA-Z0-9\s\-]', '', base_id)
                        lecture_id = re.sub(r'\s+', '_', lecture_id).strip('_')
                        
                        # Check for duplicate ID
                        if lecture_id in yt_data.get("lectures", {}):
                            st.error(
                                f"❌ Lecture ID '{lecture_id}' already exists. Choose a different Custom Lecture ID or title."
                            )
                        # Check for duplicate title
                        elif any(info.get("title", "").lower().strip() == lecture_title.lower().strip() 
                                for info in yt_data.get("lectures", {}).values()):
                            st.error(
                                f"❌ A lecture with title '{lecture_title}' already exists. Choose a different title."
                            )
                        else:
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

                            progress_container = st.container()
                            with progress_container:
                                st.info("Step 1/3: Downloading audio from YouTube...")
                                progress_bar = st.progress(0)
                                progress_bar.progress(10)

                            audio_path = os.path.join(DATA_DIR, f"temp_{lecture_id}.mp3")
                            if download_youtube_audio(youtube_url, audio_path):
                                progress_bar.progress(33)
                                with progress_container:
                                    st.info(f"Step 2/3: Transcribing '{lecture_title}' (this may take 2-5 minutes)...")
                                    st.caption("Processing audio with Whisper AI...")
                                model = load_whisper_model()
                                if model:
                                    try:
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
                                        if result and "segments" in result:
                                            with progress_container:
                                                st.info("Step 3/3: Indexing for search...")
                                            segments = [
                                                {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg["text"]).strip()}
                                                for seg in result["segments"]
                                                if seg.get("text")
                                            ]
                                            progress_bar.progress(80)
                                            transcript_path = os.path.join(TRANSCRIPT_DIR, f"{lecture_title}.txt")
                                            with open(transcript_path, "w", encoding="utf-8") as f:
                                                f.write(result.get("text", "").strip())
                                            if os.path.exists(META_SEGMENTS):
                                                with open(META_SEGMENTS, "r", encoding="utf-8") as f:
                                                    all_segments = json.load(f)
                                            else:
                                                all_segments = {}
                                            all_segments[lecture_title] = segments
                                            progress_bar.progress(90)
                                            with open(META_SEGMENTS, "w", encoding="utf-8") as f:
                                                json.dump(all_segments, f, indent=2)
                                            os.remove(audio_path)
                                            progress_bar.progress(100)
                                            st.info("Indexing new lecture for search...")
                                            embedder = load_sentence_transformer()
                                            if embedder:
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
                                                            "category": selected_category or "Uncategorized",
                                                        })
                                                    faiss.write_index(idx, INDEX_PATH)
                                                    with open(META_PATH, "w", encoding="utf-8") as f:
                                                        json.dump(meta, f, indent=2)
                                                    st.success(
                                                        f"✅ Lecture '{lecture_title}' added and indexed! ({len(segments)} segments)"
                                                    )
                                                except Exception as e:
                                                    logger.error(f"Error indexing new lecture: {e}")
                                                    st.warning("Lecture added but indexing will complete on next page load")
                                            st.info("Reloading page...")
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("Transcription failed - no segments produced")
                                    except Exception as e:
                                        st.error(f"Transcription error: {str(e)}")
                                        logger.error(f"Transcription error: {e}")
                                else:
                                    st.error("Failed to load Whisper model")
                            else:
                                st.error("Failed to download audio from YouTube. Check the URL and try again.")
                                delete_lecture_file(lecture_id)
                    except Exception as e:
                        st.error(f"Error adding lecture: {str(e)}")
                        logger.error(f"Error adding YouTube lecture: {e}")

    # --------------------
    # Manage Course Materials
    # --------------------
    st.markdown("---")
    st.markdown('<div id="course_materials"></div>', unsafe_allow_html=True)
    st.subheader("📚 Manage Course Materials")
    st.info("Create organized material sections (e.g., 'Week 1 Resources', 'Homework', 'Additional Reading') with multiple files in each section.")

    yt_data = load_youtube_lectures()
    if "course_materials" not in yt_data:
        yt_data["course_materials"] = []

    with st.expander("Create New Material Section", expanded=False):
        with st.form("new_material_form"):
            section_title = st.text_input("Section Title:", placeholder="e.g., Week 1 - Introduction Materials")
            section_description = st.text_area("Description:", placeholder="e.g., All resources for Week 1 including slides and readings")
            section_order = st.number_input("Display Order:", min_value=1, value=len(yt_data["course_materials"]) + 1, help="Lower numbers appear first")
            st.markdown("**Upload files for this section:**")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["pdf", "pptx", "ppt", "docx", "doc", "png", "jpg", "jpeg", "txt", "zip"],
                accept_multiple_files=True,
                help="You can select multiple files at once",
            )
            file_notes = {}
            if uploaded_files:
                st.markdown("**Add descriptions for each file:**")
                for uploaded_file in uploaded_files:
                    file_notes[uploaded_file.name] = st.text_input(
                        f"{uploaded_file.name}:", placeholder="Description for students...", key=f"note_{uploaded_file.name}"
                    )
            if st.form_submit_button("Create Section", type="primary"):
                if not section_title:
                    st.error("Please enter a section title")
                elif not uploaded_files:
                    st.error("Please upload at least one file")
                else:
                    try:
                        files_metadata = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            files_metadata.append({
                                "filename": uploaded_file.name,
                                "size": uploaded_file.size,
                                "type": uploaded_file.type,
                                "note": file_notes.get(uploaded_file.name, ""),
                            })
                        material_id = f"material_{len(yt_data['course_materials']) + 1}_{int(time.time())}"
                        new_material = {
                            "id": material_id,
                            "title": section_title,
                            "description": section_description,
                            "order": section_order,
                            "files": files_metadata,
                            "created_date": datetime.datetime.now().isoformat(),
                        }
                        yt_data["course_materials"].append(new_material)
                        save_youtube_lectures(yt_data)
                        st.success(f"Created '{section_title}' with {len(files_metadata)} file(s)!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create material section: {e}")
                        logger.error(f"Material creation error: {e}")

    if yt_data["course_materials"]:
        st.markdown("### 📂 Existing Material Sections")
        for material in sorted(yt_data["course_materials"], key=lambda x: x.get('order', 999)):
            with st.expander(f"📁 {material['title']} ({len(material.get('files', []))} files)", expanded=False):
                st.markdown(f"**Description:** {material['description']}")
                st.caption(f"Order: {material['order']} | Created: {material.get('created_date', 'N/A')[:10]}")
                st.markdown("---")
                st.markdown("**Files:**")
                for file_info in material.get('files', []):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{file_info['filename']}** ({file_info['size'] / 1024:.1f} KB)")
                        if file_info.get('note'):
                            st.caption(f"Note: {file_info['note']}")
                    with col2:
                        st.caption("")
                st.markdown("---")
                col_edit, col_delete = st.columns(2)
                with col_edit:
                    if st.button("Edit Section", key=f"edit_mat_{material['id']}"):
                        st.session_state[f"editing_material_{material['id']}"] = True
                with col_delete:
                    if st.button("Delete Section", key=f"del_mat_{material['id']}"):
                        st.session_state[f"confirm_delete_material_{material['id']}"] = True
                if st.session_state.get(f"editing_material_{material['id']}", False):
                    with st.form(key=f"edit_material_form_{material['id']}"):
                        st.markdown("#### Edit Section")
                        new_title = st.text_input("Title:", value=material['title'])
                        new_description = st.text_area("Description:", value=material['description'])
                        new_order = st.number_input("Order:", min_value=1, value=material['order'])
                        st.markdown("**Add more files (optional):**")
                        additional_files = st.file_uploader(
                            "Choose additional files",
                            type=["pdf", "pptx", "ppt", "docx", "doc", "png", "jpg", "jpeg", "txt", "zip"],
                            accept_multiple_files=True,
                            key=f"add_files_{material['id']}",
                        )
                        additional_notes = {}
                        if additional_files:
                            for add_file in additional_files:
                                additional_notes[add_file.name] = st.text_input(
                                    f"{add_file.name}:", placeholder="Description...", key=f"add_note_{material['id']}_{add_file.name}"
                                )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.form_submit_button("Save Changes", type="primary"):
                                material['title'] = new_title
                                material['description'] = new_description
                                material['order'] = new_order
                                if additional_files:
                                    for add_file in additional_files:
                                        file_path = os.path.join(UPLOADS_DIR, add_file.name)
                                        with open(file_path, "wb") as f:
                                            f.write(add_file.getbuffer())
                                        material['files'].append({
                                            "filename": add_file.name,
                                            "size": add_file.size,
                                            "type": add_file.type,
                                            "note": additional_notes.get(add_file.name, ""),
                                        })
                                save_youtube_lectures(yt_data)
                                st.session_state[f"editing_material_{material['id']}"] = False
                                st.success("Changes saved!")
                                time.sleep(1)
                                st.rerun()
                        with col_cancel:
                            if st.form_submit_button("Cancel"):
                                st.session_state[f"editing_material_{material['id']}"] = False
                                st.rerun()
                if st.session_state.get(f"confirm_delete_material_{material['id']}", False):
                    st.warning(
                        f"Delete section '{material['title']}'? This will also delete all {len(material.get('files', []))} file(s)."
                    )
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, Delete", key=f"yes_del_mat_{material['id']}"):
                            for file_info in material.get('files', []):
                                file_path = os.path.join(UPLOADS_DIR, file_info['filename'])
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            yt_data["course_materials"] = [m for m in yt_data["course_materials"] if m['id'] != material['id']]
                            save_youtube_lectures(yt_data)
                            st.session_state[f"confirm_delete_material_{material['id']}"] = False
                            st.success("Section deleted!")
                            time.sleep(1)
                            st.rerun()
                    with col_no:
                        if st.button("Cancel", key=f"no_del_mat_{material['id']}"):
                            st.session_state[f"confirm_delete_material_{material['id']}"] = False
                            st.rerun()
    else:
        st.info("No material sections created yet. Create your first section above!")

    # --------------------
    # Manage Existing Lectures
    # --------------------
    st.markdown("---")
    st.markdown('<div id="manage_lectures"></div>', unsafe_allow_html=True)
    st.subheader("🗂️ Manage Existing Lectures")

    yt_data = load_youtube_lectures()
    all_lecture_data = yt_data.get("lectures", {})
    if all_lecture_data:
        categories = {}
        for lec_id, lec_info in all_lecture_data.items():
            cat = lec_info.get("category", "Uncategorized")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((lec_id, lec_info))
        for category in sorted(categories.keys()):
            with st.expander(f"📂 {category} ({len(categories[category])} lectures)"):
                for lec_id, lec_info in sorted(categories[category], key=lambda x: x[1].get("title", "")):
                    title = lec_info.get("title", "Unknown")
                    note = lec_info.get("professor_note", "")
                    youtube_url = lec_info.get("youtube_url", "")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{title}**")
                        if note:
                            st.caption(f"📝 Note: {note}")
                        st.caption(f"🔗 [{youtube_url}]({youtube_url})")
                    with col2:
                        if st.button("Edit", key=f"edit_{lec_id}"):
                            st.session_state[f"editing_{lec_id}"] = True
                        if st.button("Delete", key=f"delete_{lec_id}"):
                            st.session_state[f"confirm_delete_{lec_id}"] = True
                    if st.session_state.get(f"editing_{lec_id}", False):
                        with st.form(key=f"edit_form_{lec_id}"):
                            st.markdown("#### Edit Lecture")
                            new_title = st.text_input("Title:", value=title)
                            new_note = st.text_area("Professor's Note:", value=note)
                            new_category = st.selectbox(
                                "Category:", options=sorted(categories.keys()) + ["+ New Category"],
                                index=sorted(categories.keys()).index(category) if category in categories else 0,
                            )
                            if new_category == "+ New Category":
                                new_category = st.text_input("New category name:")
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.form_submit_button("Save Changes", type="primary"):
                                    yt_data["lectures"][lec_id]["title"] = new_title
                                    yt_data["lectures"][lec_id]["professor_note"] = new_note
                                    yt_data["lectures"][lec_id]["category"] = new_category
                                    if new_title != title:
                                        if os.path.exists(META_SEGMENTS):
                                            with open(META_SEGMENTS, 'r', encoding='utf-8') as f:
                                                all_segments = json.load(f)
                                            if title in all_segments:
                                                all_segments[new_title] = all_segments.pop(title)
                                                with open(META_SEGMENTS, 'w', encoding='utf-8') as f:
                                                    json.dump(all_segments, f, indent=2)
                                        old_transcript = os.path.join(TRANSCRIPT_DIR, f"{title}.txt")
                                        new_transcript = os.path.join(TRANSCRIPT_DIR, f"{new_title}.txt")
                                        if os.path.exists(old_transcript):
                                            os.rename(old_transcript, new_transcript)
                                    save_youtube_lectures(yt_data)
                                    st.session_state[f"editing_{lec_id}"] = False
                                    st.success("Changes saved!")
                                    time.sleep(1)
                                    st.rerun()
                            with col_cancel:
                                if st.form_submit_button("Cancel"):
                                    st.session_state[f"editing_{lec_id}"] = False
                                    st.rerun()
                    if st.session_state.get(f"confirm_delete_{lec_id}", False):
                        st.warning(f"Are you sure you want to delete **{title}**?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Yes, Delete", key=f"confirm_yes_{lec_id}"):
                                delete_lecture_file(lec_id)
                                if os.path.exists(META_SEGMENTS):
                                    with open(META_SEGMENTS, 'r', encoding='utf-8') as f:
                                        all_segments = json.load(f)
                                    if title in all_segments:
                                        del all_segments[title]
                                        with open(META_SEGMENTS, 'w', encoding='utf-8') as f:
                                            json.dump(all_segments, f, indent=2)
                                transcript_path = os.path.join(TRANSCRIPT_DIR, f"{title}.txt")
                                if os.path.exists(transcript_path):
                                    os.remove(transcript_path)
                                st.session_state[f"confirm_delete_{lec_id}"] = False
                                st.success(f"Deleted '{title}'")
                                time.sleep(1)
                                st.rerun()
                        with col_no:
                            if st.button("Cancel", key=f"confirm_no_{lec_id}"):
                                st.session_state[f"confirm_delete_{lec_id}"] = False
                                st.rerun()
                    st.markdown("---")
    else:
        st.info("No lectures added yet. Add your first lecture above!")

    # --------------------
    # Danger Zone: Clear All Data
    # --------------------
    st.markdown("---")
    st.subheader("Danger Zone")
    if st.button("Clear All Data"):
        st.session_state['confirm_clear_all'] = True

    if st.session_state.get('confirm_clear_all', False):
        st.error("🚨 **DANGER ZONE** - This will delete:")
        st.markdown(
            """
        - All lectures from YouTube database
        - All transcripts
        - All segments and search index
        - All analytics data
        
        **This action cannot be undone!**
        """
        )
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, Delete Everything", type="primary"):
                if os.path.exists(LECTURES_DIR):
                    for file in os.listdir(LECTURES_DIR):
                        if file.endswith('.json'):
                            os.remove(os.path.join(LECTURES_DIR, file))
                materials_path = os.path.join(DATA_DIR, "course_materials.json")
                if os.path.exists(materials_path):
                    os.remove(materials_path)
                if os.path.exists(META_SEGMENTS):
                    os.remove(META_SEGMENTS)
                if os.path.exists(META_PATH):
                    os.remove(META_PATH)
                if os.path.exists(INDEX_PATH):
                    os.remove(INDEX_PATH)
                if os.path.exists(TRANSCRIPT_DIR):
                    for file in os.listdir(TRANSCRIPT_DIR):
                        if file.endswith('.txt'):
                            os.remove(os.path.join(TRANSCRIPT_DIR, file))
                if os.path.exists(LOG_PATH):
                    os.remove(LOG_PATH)
                st.session_state['confirm_clear_all'] = False
                st.success("All data cleared!")
                time.sleep(2)
                st.rerun()
        with col_no:
            if st.button("Cancel"):
                st.session_state['confirm_clear_all'] = False
                st.rerun()

    # --------------------
    # Analytics Dashboard
    # --------------------
    st.markdown("---")
    st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
    st.subheader("📊 SmartTA Analytics Dashboard")
    if not os.path.exists(LOG_PATH):
        st.info("No queries yet.")
    else:
        df = load_analytics_data()
        render_lecture_frequency_chart(df)
        render_query_timeline_analysis(df)
        st.markdown("### Advanced Analytics")
        render_query_patterns_over_time(df)
