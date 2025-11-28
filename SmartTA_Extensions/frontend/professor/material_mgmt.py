"""Course materials management UI components for Professor Dashboard.
"""
import os
import time
import datetime
import logging
from typing import Dict, Any

import streamlit as st

from SmartTA_Extensions.utils.helpers import load_youtube_lectures, save_youtube_lectures

logger = logging.getLogger(__name__)


def render_material_management(UPLOADS_DIR: str) -> None:
    """Render the course materials management section."""
    st.markdown("---")
    st.markdown('<div id="course_materials"></div>', unsafe_allow_html=True)
    st.subheader("📚 Manage Course Materials")
    st.info("Create organized material sections (e.g., 'Week 1 Resources', 'Homework', 'Additional Reading') with multiple files in each section.")

    yt_data = load_youtube_lectures()
    if "course_materials" not in yt_data:
        yt_data["course_materials"] = []

    # Create new section form
    with st.expander("Create New Material Section", expanded=False):
        _render_create_material_form(yt_data, UPLOADS_DIR)

    # Display existing sections
    if yt_data["course_materials"]:
        st.markdown("### 📂 Existing Material Sections")
        for material in sorted(yt_data["course_materials"], key=lambda x: x.get('order', 999)):
            _render_material_section(material, UPLOADS_DIR)
    else:
        st.info("No material sections created yet. Create your first section above!")


def _render_create_material_form(yt_data: Dict[str, Any], UPLOADS_DIR: str) -> None:
    """Render form to create a new material section."""
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


def _render_material_section(material: Dict[str, Any], UPLOADS_DIR: str) -> None:
    """Render a single material section with files and edit/delete controls."""
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
        
        # Edit form
        if st.session_state.get(f"editing_material_{material['id']}", False):
            _render_edit_material_form(material, UPLOADS_DIR)
        
        # Delete confirmation
        if st.session_state.get(f"confirm_delete_material_{material['id']}", False):
            _render_delete_material_confirmation(material, UPLOADS_DIR)


def _render_edit_material_form(material: Dict[str, Any], UPLOADS_DIR: str) -> None:
    """Render edit form for a material section."""
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
                yt_data = load_youtube_lectures()
                save_youtube_lectures(yt_data)
                st.session_state[f"editing_material_{material['id']}"] = False
                st.success("Changes saved!")
                time.sleep(1)
                st.rerun()
        with col_cancel:
            if st.form_submit_button("Cancel"):
                st.session_state[f"editing_material_{material['id']}"] = False
                st.rerun()


def _render_delete_material_confirmation(material: Dict[str, Any], UPLOADS_DIR: str) -> None:
    """Render delete confirmation for a material section."""
    st.warning(f"Delete section '{material['title']}'? This will also delete all {len(material.get('files', []))} file(s).")
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Yes, Delete", key=f"yes_del_mat_{material['id']}"):
            for file_info in material.get('files', []):
                file_path = os.path.join(UPLOADS_DIR, file_info['filename'])
                if os.path.exists(file_path):
                    os.remove(file_path)
            yt_data = load_youtube_lectures()
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
