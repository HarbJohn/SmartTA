from __future__ import annotations

import re
from datetime import date, datetime

import altair as alt
import pandas as pd
import streamlit as st

from . import config
from .helpers import read_logs_df


def _canon_q(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


ALIAS_MAP: dict[str, str] = {}


def _alias_q(text: str) -> str:
    key = _canon_q(text)
    return ALIAS_MAP.get(key, key)


def render_instructor_dashboard() -> None:
    """Render instructor analytics for the RAG assistant."""
    df = read_logs_df()
    if df.empty:
        st.info("No logs yet — ask some questions first.")
        return

    st.markdown("### 📅 Filters")
    df = df[df["date"].notna()].copy()
    if df.empty:
        st.info("No valid logs with dates yet.")
        return

    min_date, max_date = df["date"].min(), df["date"].max()
    today = date.today()
    if not isinstance(min_date, date):
        min_date = today
    if not isinstance(max_date, date):
        max_date = today

    date_range = st.date_input("Select Date Range", value=(min_date, max_date))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_from, d_to = date_range
    else:
        d_from, d_to = (min_date, max_date)

    conf_filter = st.slider("Min confidence filter", 0.0, 1.0, 0.0, 0.01)
    fb_filter = st.selectbox("Feedback filter", ["all", "helpful only", "unclear only"])

    mask = (df["date"] >= d_from) & (df["date"] <= d_to) & (df["top_score"] >= conf_filter)
    df = df[mask].copy()
    if fb_filter == "helpful only":
        df = df[df["feedback"] == "helpful"]
    elif fb_filter == "unclear only":
        df = df[df["feedback"] == "unclear"]
    if df.empty:
        st.warning("No data after filters.")
        return

    df["q_norm"] = df["question"].fillna("").astype(str).map(_alias_q)

    total_q = int(len(df))
    avg_conf = float(df["top_score"].mean())
    active_days = int(df["date"].nunique())
    helpful_cnt = int((df["feedback"] == "helpful").sum())
    unclear_cnt = int((df["feedback"] == "unclear").sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(
        f"<div class='metric-card'><h2>{total_q}</h2><p>Total Q&A Rows</p></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='metric-card'><h2>{avg_conf:.3f}</h2><p>Average Confidence</p></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='metric-card'><h2>{active_days}</h2><p>Active Days</p></div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='metric-card'><h2>{helpful_cnt}</h2><p>👍 Helpful</p></div>",
        unsafe_allow_html=True,
    )
    c5.markdown(
        f"<div class='metric-card'><h2>{unclear_cnt}</h2><p>👎 Unclear</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🔝 Top 10 Topics")
    st.caption(
        "Frequent 2–5 word phrases from student questions (stop-words removed) — provides deeper context."
    )
    questions = df["question"].dropna().astype(str)
    if not questions.empty and config.CountVectorizer:
        try:
            vectorizer = config.CountVectorizer(
                ngram_range=(2, 5),
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            )
            matrix = vectorizer.fit_transform(questions)
            freqs = matrix.toarray().sum(axis=0)
            vocab = vectorizer.get_feature_names_out()
            topic_df = (
                pd.DataFrame({"Phrase": vocab, "Count": freqs})
                .sort_values("Count", ascending=False)
                .head(10)
            )
            st.altair_chart(
                alt.Chart(topic_df)
                .mark_bar(color="#2e7cf6")
                .encode(
                    x="Count:Q",
                    y=alt.Y("Phrase:N", sort="-x", axis=alt.Axis(labelFontSize=12, labelLimit=300)),
                    tooltip=["Phrase", "Count"],
                )
                .properties(height=450, width=900),
                use_container_width=True,
            )
        except Exception:
            st.info("Not enough signal to compute topics yet.")
    else:
        st.caption("No questions to analyze or scikit-learn missing.")

    st.markdown("---")
    st.markdown("### 📄 Most Referenced Slides")
    mentioned: list[str] = []
    for contexts in df["contexts"].dropna():
        if isinstance(contexts, list):
            for ctx in contexts:
                try:
                    mentioned.append(f"{ctx.get('pdf')} p.{ctx.get('page')}")
                except Exception:
                    continue
    if mentioned:
        slide_df = pd.Series(mentioned).value_counts().head(12).reset_index()
        slide_df.columns = ["Slide", "Mentions"]
        st.altair_chart(
            alt.Chart(slide_df)
            .mark_bar(color="#ef4444")
            .encode(
                x="Mentions:Q",
                y=alt.Y("Slide:N", sort="-x"),
                tooltip=["Slide", "Mentions"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    else:
        st.caption("No slides referenced yet.")

    st.markdown("---")
    st.markdown("### 🧩 Slide References per Chapter (C1–C6)")
    st.caption("Counts unique slides referenced per chapter.")
    chapter_refs: list[tuple[str, str | None]] = []
    pattern = re.compile(r"(C[1-6])", re.IGNORECASE)
    for contexts in df["contexts"].dropna():
        if isinstance(contexts, list):
            for ctx in contexts:
                pdf = str(ctx.get("pdf", ""))
                page = ctx.get("page")
                match = pattern.search(pdf)
                if match:
                    chapter_refs.append((match.group(1).upper(), page))
    if chapter_refs:
        refs_df = pd.DataFrame(chapter_refs, columns=["Chapter", "Page"]).drop_duplicates()
        slide_counts = (
            refs_df["Chapter"]
            .value_counts()
            .reindex(["C1", "C2", "C3", "C4", "C5", "C6"])
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        slide_counts.columns = ["Chapter", "UniqueSlides"]
        st.altair_chart(
            alt.Chart(slide_counts)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, color="#3b82f6")
            .encode(
                x=alt.X("Chapter:N", sort=["C1", "C2", "C3", "C4", "C5", "C6"], title="Chapter"),
                y=alt.Y("UniqueSlides:Q", title="Unique Slides"),
                tooltip=["Chapter", "UniqueSlides"],
            )
            .properties(height=350, title="Unique Slides Referenced per Chapter")
            .configure_axis(labelColor="#e5e7eb", titleColor="#e5e7eb"),
            use_container_width=True,
        )
    else:
        st.caption("No slide references yet.")

    st.markdown("---")
    st.markdown("### ❗ Low-Rated & Low-Confidence Responses")
    tab_unclear, tab_lowconf = st.tabs(["👎 Unclear (by time)", "Lowest Confidence (top 10)"])
    with tab_unclear:
        unclear = df[df["feedback"] == "unclear"][["ts", "question", "top_score", "answer"]]
        if unclear.empty:
            st.caption("No 'unclear' feedback yet.")
        else:
            unclear = unclear.sort_values("ts", ascending=False).rename(
                columns={"ts": "Time", "top_score": "Confidence"}
            )
            st.dataframe(unclear, use_container_width=True)
    with tab_lowconf:
        low_conf = df.sort_values("top_score").head(10)[["question", "top_score", "answer", "date"]]
        st.dataframe(
            low_conf.rename(columns={"top_score": "Confidence"}),
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### 👍/👎 Feedback Distribution")
    fb_counts = (
        df["feedback"]
        .value_counts()
        .reindex(["helpful", "unclear", "none"])
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    fb_counts.columns = ["Feedback", "Count"]
    st.altair_chart(
        alt.Chart(fb_counts)
        .mark_bar(color="#34d399")
        .encode(
            x=alt.X("Feedback:N", title="Type"),
            y=alt.Y("Count:Q", title="Count"),
            tooltip=["Feedback", "Count"],
        )
        .properties(height=300),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### ♻️ Repeated Questions")
    repeats = (
        df[df["q_norm"] != ""]
        .groupby("q_norm")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="Times")
        .rename(columns={"q_norm": "Question"})
    )
    if not repeats.empty:
        st.dataframe(repeats.head(15), use_container_width=True, height=300)
    else:
        st.caption("No repeated questions yet.")

    st.markdown("### ⏱️ Engagement Heatmap (Hour × Day)")
    ts = df["ts"].dropna()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grid = pd.MultiIndex.from_product([day_order, list(range(24))], names=["dow", "hour"]).to_frame(
        index=False
    )
    if not ts.empty:
        counts = (
            pd.DataFrame({"dow": ts.dt.day_name(), "hour": ts.dt.hour})
            .value_counts(["dow", "hour"])
            .rename("count")
            .reset_index()
        )
        heat = grid.merge(counts, on=["dow", "hour"], how="left").fillna({"count": 0})
    else:
        heat = grid.copy()
        heat["count"] = 0

    def _fmt_hour(hour: int) -> str:
        if hour == 0:
            return "12 am"
        if hour < 12:
            return f"{hour} am"
        if hour == 12:
            return "12 pm"
        return f"{hour - 12} pm"

    heat["hour_label"] = heat["hour"].map(_fmt_hour)
    heatmap = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X(
                "hour:O",
                sort=list(range(24)),
                title="Hour",
                axis=alt.Axis(
                    values=list(range(24)),
                    labelExpr=(
                        'datum.value==0 ? "12 am" : '
                        'datum.value<12 ? datum.value + " am" : '
                        'datum.value==12 ? "12 pm" : (datum.value-12) + " pm"'
                    ),
                ),
            ),
            y=alt.Y("dow:N", sort=day_order, title="Day"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
            tooltip=[
                alt.Tooltip("dow:N", title="Day"),
                alt.Tooltip("hour_label:N", title="Hour"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(heatmap, use_container_width=True)

    st.markdown("### 🕒 Engagement Over Time")
    daily = df.groupby("date").size().reset_index(name="Questions")
    st.altair_chart(
        alt.Chart(daily)
        .mark_area(color="#20c997")
        .encode(x="date:T", y="Questions:Q", tooltip=["date:T", "Questions:Q"])
        .properties(height=300),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 📥 Student Questions Database")
    csv_bytes = df.sort_values("ts").to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Student Questions",
        data=csv_bytes,
        file_name=f"student_questions_{datetime.now().strftime('%Y-%m-%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
