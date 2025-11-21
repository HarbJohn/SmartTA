"""Course feedback survey page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from . import config

FEEDBACK_FILE = Path(getattr(config, "APP_DATA_DIR", Path("data"))) / "course_feedback.csv"


def _load_feedback_data() -> pd.DataFrame:
    if FEEDBACK_FILE.exists():
        try:
            return pd.read_csv(FEEDBACK_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _append_feedback(entry: dict) -> None:
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = not FEEDBACK_FILE.exists()
    pd.DataFrame([entry]).to_csv(FEEDBACK_FILE, mode="a", index=False, header=header)


def render_course_feedback() -> None:
    st.markdown("## 📝 Course Feedback Survey")
    st.caption("Anonymous survey results are stored locally in data/course_feedback.csv.")

    feedback_df = _load_feedback_data()
    with st.form("course_feedback"):
        col_a, col_b = st.columns(2)
        with col_a:
            q_satisfied = st.radio("Overall, are you satisfied with your course experience so far?", ["Yes", "No"], horizontal=True)
            q_engaging = st.radio("Do you find the in-class interactions engaging and helpful?", ["Yes", "No"], horizontal=True)
            q_intro = st.radio("Do you think the 5 minutes class intro is helpful?", ["Yes", "No"], horizontal=True)
            q_outro = st.radio("Do you think the 5 minutes class outro is helpful?", ["Yes", "No"], horizontal=True)
        with col_b:
            q_project = st.radio("Is the project-driven approach effective in enhancing your learning?", ["Yes", "No"], horizontal=True)
            q_paths = st.radio(
                "How effective do you find the diverse learning paths used in this course?",
                ["Very effective", "Moderately effective", "Slightly effective", "Not effective"],
            )
            q_preference = st.radio(
                "Which would you have preferred?",
                ["2 Hackathons (30% of grade)", "1 Hackathon (10%) - 1 Midterm (20%)"],
            )
        q_balance = st.text_area(
            "If you could adjust the balance between these components, what would you change? "
            "(e.g., “More project work”, “Fewer theoretical assignments”, “More in-class demos”, etc.)"
        )
        q_favorite = st.text_area("What do you like most about the in-class interactions?")
        q_suggestions = st.text_area("What suggestions do you have to make the coursework and activities more engaging or manageable?")
        q_comments = st.text_area("Any additional comments or suggestions?")

        submitted = st.form_submit_button("Submit Survey")
        if submitted:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "satisfied": q_satisfied,
                "engaging": q_engaging,
                "intro_helpful": q_intro,
                "outro_helpful": q_outro,
                "project_effective": q_project,
                "paths_effective": q_paths,
                "preference": q_preference,
                "balance_change": q_balance.strip(),
                "favorite_interaction": q_favorite.strip(),
                "suggestions": q_suggestions.strip(),
                "comments": q_comments.strip(),
            }
            _append_feedback(entry)
            st.success("✅ Thank you! Your feedback has been recorded.")
            feedback_df = pd.concat([feedback_df, pd.DataFrame([entry])], ignore_index=True)

    if not feedback_df.empty:
        st.caption(f"Collected responses: {len(feedback_df)}")
        satisfied_rate = (feedback_df["satisfied"] == "Yes").mean() if "satisfied" in feedback_df else 0
        engaging_rate = (feedback_df["engaging"] == "Yes").mean() if "engaging" in feedback_df else 0
        project_rate = (feedback_df["project_effective"] == "Yes").mean() if "project_effective" in feedback_df else 0
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Satisfied", f"{satisfied_rate * 100:0.0f}%")
        fc2.metric("Engaged in class", f"{engaging_rate * 100:0.0f}%")
        fc3.metric("Project approach effective", f"{project_rate * 100:0.0f}%")
        with st.expander("Latest responses"):
            st.dataframe(feedback_df.tail(10), use_container_width=True)
    else:
        st.caption("No survey responses recorded yet.")
