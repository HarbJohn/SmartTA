"""
Analytics Module
Handles query logging, analytics data loading, and visualization generation.
Copied from app.py without modifications.
"""

import os
import json
import logging
import datetime
import re
from collections import Counter
import pandas as pd
import streamlit as st
from SmartTA_Extensions.utils.config import DATA_DIR, LOG_PATH

# Constants centralized via utils/config.py

logger = logging.getLogger(__name__)

# Optional imports for enhanced visualizations
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

# --------------------
# ANALYTICS FUNCTIONS (Copied verbatim from app.py)
# --------------------

def log_query(question, lecture_filter, results):
    """Log a search query to the analytics file."""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "lecture_filter": lecture_filter,
            "results": [
                {
                    "lecture": r.get('lecture'),
                    "start": r.get('start'),
                    "end": r.get('end'),
                    "text": r.get('text', '')[:200]
                } for r in results
            ]
        }
        existing = []
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'r', encoding='utf-8') as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
                except Exception:
                    existing = []
        existing.append(log_entry)
        with open(LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to append analytics log: {e}")


@st.cache_data(ttl=60)
def load_analytics_data():
    """Load analytics log safely and return DataFrame with expected columns.
    Returns empty DataFrame if file missing or malformed.
    Cached for 60 seconds to improve dashboard performance.
    """
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=["timestamp", "question", "lecture_filter", "results"])
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect a list of dicts
        if not isinstance(data, list):
            logger.warning("Analytics log is not a list; returning empty DataFrame")
            return pd.DataFrame(columns=["timestamp", "question", "lecture_filter", "results"])
        df_local = pd.DataFrame(data)
        # Ensure expected columns exist
        for col in ["timestamp", "question", "lecture_filter", "results"]:
            if col not in df_local.columns:
                df_local[col] = None if col != 'results' else [[] for _ in range(len(df_local))]
        return df_local
    except Exception as e:
        logger.error(f"Failed to load analytics data: {e}")
        return pd.DataFrame(columns=["timestamp", "question", "lecture_filter", "results"])


def render_lecture_frequency_chart(df):
    """Render the most queried lectures chart."""
    lec_counts = Counter([r["lecture"] for _, row in df.iterrows() 
                        for r in row["results"]])
    st.markdown("### 🎥 Most Queried Lectures")
    
    if PLOTLY_AVAILABLE:
        # Create interactive bar chart
        fig_lectures = go.Figure()
        
        # Sort lectures by count for better visualization
        sorted_lectures = sorted(lec_counts.items(), key=lambda x: x[1], reverse=True)
        lectures, counts = zip(*sorted_lectures) if sorted_lectures else ([], [])
        
        fig_lectures.add_trace(go.Bar(
            x=lectures,
            y=counts,
            marker_color='royalblue',
            text=counts,  # Add value labels
            textposition='auto',
        ))
        
        fig_lectures.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=100),
            xaxis_title="Lecture Name",
            yaxis_title="Number of Queries",
            xaxis_tickangle=-45,  # Angle the x-axis labels
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_lectures, config={"responsive": True})
    else:
        st.bar_chart(pd.DataFrame.from_dict(lec_counts, orient="index"))


def render_query_timeline_analysis(df):
    """Render the query timeline analysis with carousel navigation."""
    st.markdown("### ⏱ Query Timeline Analysis")
    
    hotspot_data = {}
    for _, row in df.iterrows():
        for r in row["results"]:
            lec = r["lecture"]
            time_point = (r["start"] + r["end"]) / 2
            if lec not in hotspot_data:
                hotspot_data[lec] = []
            hotspot_data[lec].append(time_point)
    
    # Initialize session state for carousel index
    if 'timeline_graph_index' not in st.session_state:
        st.session_state.timeline_graph_index = 0
    
    lectures_list = list(hotspot_data.keys())
    
    if lectures_list and PLOTLY_AVAILABLE:
        # Navigation controls
        col1, col2, col3 = st.columns([1, 6, 1])
        
        with col1:
            if st.button("◀", key="timeline_prev", width='stretch'):
                st.session_state.timeline_graph_index = (st.session_state.timeline_graph_index - 1) % len(lectures_list)
        
        with col2:
            st.markdown(f"<h4 style='text-align: center;'>Graph {st.session_state.timeline_graph_index + 1} of {len(lectures_list)}</h4>", unsafe_allow_html=True)
        
        
        with col3:
            if st.button("▶", key="timeline_next", width='stretch'):
                st.session_state.timeline_graph_index = (st.session_state.timeline_graph_index + 1) % len(lectures_list)
        
        # Display current lecture graph
        current_lec = lectures_list[st.session_state.timeline_graph_index]
        times = hotspot_data[current_lec]
        
        fig_hotspot = go.Figure()
        
        # Convert to minutes for better readability
        times_minutes = [t/60 for t in times]
        
        fig_hotspot.add_trace(go.Histogram(
            x=times_minutes,
            nbinsx=20,
            name='Query Count',
            marker_color='rgba(65, 105, 225, 0.7)'
        ))
        
        fig_hotspot.update_layout(
            title=f"Query Distribution for {current_lec}",
            height=400,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis_title="Time in Video (minutes)",
            yaxis_title="Number of Queries",
            bargap=0.1,
            plot_bgcolor='white'
        )
        
        # Add rangeslider for easy navigation
        fig_hotspot.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig_hotspot, config={"responsive": True})
    elif lectures_list:
        # Fallback to simple bar chart for non-plotly environments
        avg_hotspots = {lec: sum(v)/len(v) for lec, v in hotspot_data.items() if v}
        st.bar_chart(pd.DataFrame.from_dict(avg_hotspots, orient="index"))


def render_query_patterns_over_time(df):
    """Render query patterns over time analysis."""
    st.subheader("Query Patterns Over Time")
    # Parse timestamps robustly to avoid crashes from mixed ISO formats
    try:
        # Use permissive parsing
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as e:
        logger.warning(f"Timestamp parse error: {e}. Falling back to permissive parsing.")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Drop rows with invalid timestamps for time-based charts
    if df['timestamp'].isnull().any():
        logger.warning("Some analytics entries have invalid timestamps and will be ignored in time-based charts.")
    df = df.dropna(subset=['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    hourly_counts = df.groupby('hour').size()
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_counts.index,
            y=hourly_counts.values,
            mode='lines+markers',
            name='Queries per Hour'
        ))
        fig.update_layout(
            title='Query Distribution by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Queries'
        )
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.bar_chart(hourly_counts)
    
    return df  # Return modified df for subsequent use
