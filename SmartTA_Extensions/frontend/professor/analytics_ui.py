"""Analytics and recommendations UI components for Professor Dashboard.
"""
import os
import re
import json
import datetime
import logging
from typing import Dict, Any

import streamlit as st

from SmartTA_Extensions.utils.helpers import load_youtube_lectures
from SmartTA_Extensions.backend.analytics import (
    load_analytics_data,
    render_lecture_frequency_chart,
    render_query_timeline_analysis,
    render_query_patterns_over_time,
)

logger = logging.getLogger(__name__)


def render_confused_topics(LOG_PATH: str) -> None:
    """Render the 'Top Confused Topics' section with recommendations."""
    st.markdown("---")
    st.subheader("Top Confused Topics")

    if not os.path.exists(LOG_PATH):
        st.info("No query data available yet. Start using the Student Assistant to see recommendations!")
        return

    try:
        # Load current lectures to validate references
        yt_data = load_youtube_lectures()
        valid_lecture_titles = {info.get('title') for info in yt_data.get('lectures', {}).values() if info.get('title')}
        
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        if not logs:
            st.info("No queries logged yet.")
            return

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
                    # Skip segments from deleted lectures
                    lecture_title = r.get('lecture', '')
                    if lecture_title not in valid_lecture_titles:
                        continue
                    location = {
                        'lecture': lecture_title,
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
                with st.expander(f" #{i}: {topic.title()} ({data['count']} queries)", expanded=False):
                    st.markdown("**Students asked:**")
                    for q in data['questions'][:5]:
                        st.markdown(f"- *\"{q}\"*")
                    if len(data['questions']) > 5:
                        st.caption(f"... and {len(data['questions']) - 5} more variations")
                    st.markdown("---")
                    st.markdown("**Relevant lecture segments:**")
                    locations_text = ""
                    for loc in data['locations'][:5]:
                        time_str = f"{datetime.timedelta(seconds=int(loc['start']))} - {datetime.timedelta(seconds=int(loc['end']))}"
                        locations_text += f"• **{loc['lecture']}** @ {time_str}\n"
                        locations_text += f"  _{loc['text']}_\n\n"
                    st.markdown(locations_text)
        else:
            st.info("No queries yet. Students will start asking questions soon!")
    except Exception as e:
        logger.error(f"Error loading recommendations: {e}")
        st.error("Could not load recommendations")


def render_analytics_dashboard(LOG_PATH: str) -> None:
    """Render the analytics dashboard section."""
    st.markdown("---")
    st.markdown('<div id="analytics"></div>', unsafe_allow_html=True)
    st.subheader("📊 SmartTA Analytics Dashboard")
    
    if not os.path.exists(LOG_PATH):
        st.info("No queries yet.")
        return
    
    df = load_analytics_data()
    render_lecture_frequency_chart(df)
    render_query_timeline_analysis(df)
    st.markdown("### Advanced Analytics")
    df = render_query_patterns_over_time(df)
