
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
from utils.config import DATA_DIR, LOG_PATH

# Constants centralized via utils/config.py

logger = logging.getLogger(__name__)

# Optional imports for enhanced visualizations
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

# ANALYTICS FUNCTIONS 

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
        # Create interactive bar chart with professional styling
        fig_lectures = go.Figure()
        
        # Sort lectures by count for better visualization
        sorted_lectures = sorted(lec_counts.items(), key=lambda x: x[1], reverse=True)
        lectures, counts = zip(*sorted_lectures) if sorted_lectures else ([], [])
        
        fig_lectures.add_trace(go.Bar(
            x=lectures,
            y=counts,
            marker=dict(
                color='rgba(102, 126, 234, 0.9)',
                line=dict(color='rgba(102, 126, 234, 1)', width=1.5)
            ),
            text=counts,
            textposition='auto',
            textfont=dict(color='white', size=11, family='Arial'),
            hovertemplate='<b>%{x}</b><br>Queries: %{y}<extra></extra>',
        ))
        
        fig_lectures.update_layout(
            height=450,
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=100),
            xaxis_title="Lecture Name",
            yaxis_title="Number of Queries",
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            paper_bgcolor='rgba(13, 17, 23, 0)',
            font=dict(color='#e6eef8', family='Arial', size=11),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255, 255, 255, 0.1)'
            ),
            hovermode='x unified'
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
            if st.button("◀", key="timeline_prev", use_container_width=True):
                st.session_state.timeline_graph_index = (st.session_state.timeline_graph_index - 1) % len(lectures_list)
        
        with col2:
            st.markdown(f"<h4 style='text-align: center; color: #e6eef8;'>Graph {st.session_state.timeline_graph_index + 1} of {len(lectures_list)}</h4>", unsafe_allow_html=True)
        
        
        with col3:
            if st.button("▶", key="timeline_next", use_container_width=True):
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
            marker=dict(
                color='rgba(46, 204, 113, 0.8)',
                line=dict(color='rgba(46, 204, 113, 1)', width=1)
            ),
            hovertemplate='<b>Time Range</b><br>%{x:.1f} min<br>Queries: %{y}<extra></extra>'
        ))
        
        fig_hotspot.update_layout(
            title=f"<b>Query Distribution: {current_lec}</b>",
            height=420,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=60),
            xaxis_title="Time in Video (minutes)",
            yaxis_title="Number of Queries",
            bargap=0.1,
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            paper_bgcolor='rgba(13, 17, 23, 0)',
            font=dict(color='#e6eef8', family='Arial', size=11),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255, 255, 255, 0.1)',
                rangeslider_visible=True,
                rangeslider_thickness=0.08
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255, 255, 255, 0.1)'
            ),
            hovermode='x unified'
        )
        
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
            name='Queries per Hour',
            line=dict(
                color='rgba(58, 134, 255, 0.9)',
                width=3,
                dash='solid'
            ),
            marker=dict(
                size=10,
                color='rgba(58, 134, 255, 1)',
                line=dict(width=2, color='#1e293b')
            ),
            fill='tozeroy',
            fillcolor='rgba(58, 134, 255, 0.2)',
            hovertemplate='<b>Hour %{x}:00</b><br>Queries: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title='<b>Query Distribution by Hour</b>',
            height=420,
            xaxis_title='Hour of Day',
            yaxis_title='Number of Queries',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            paper_bgcolor='rgba(13, 17, 23, 0)',
            font=dict(color='#e6eef8', family='Arial', size=11),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255, 255, 255, 0.1)',
                dtick=1
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.08)',
                showline=True,
                linewidth=1,
                linecolor='rgba(255, 255, 255, 0.1)'
            ),
            margin=dict(l=20, r=20, t=50, b=50),
            hovermode='x unified'
        )
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.bar_chart(hourly_counts)
    
    return df  # Return modified df for subsequent use


def render_student_engagement_analysis(df, get_all_lectures_func):
    """Render student engagement analysis."""
    st.subheader("Student Engagement Analysis")
    all_lectures = get_all_lectures_func()  # noqa: F821
    if all_lectures:
        engagement_data = {}
        for lec in all_lectures:
            lecture_name = os.path.splitext(lec)[0]
            lec_queries = df[df['lecture_filter'] == lecture_name]
            if not lec_queries.empty:
                engagement_data[lec] = {
                    'total_queries': len(lec_queries),
                    'avg_results': len(lec_queries['results'].explode()),
                    'unique_segments': len(set(r['start'] for results in lec_queries['results'] for r in results))
                }
        
        if engagement_data:
            engagement_df = pd.DataFrame(engagement_data).T
            engagement_df.columns = ['Total Queries', 'Results Viewed', 'Unique Segments']
            st.dataframe(engagement_df)
            
            # Engagement visualization
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                for col in engagement_df.columns:
                    fig.add_trace(go.Bar(
                        name=col,
                        x=engagement_df.index,
                        y=engagement_df[col],
                        text=engagement_df[col],
                        textposition='auto',
                    ))
                fig.update_layout(
                    title='Lecture Engagement Metrics',
                    barmode='group'
                )
                st.plotly_chart(fig, config={"responsive": True})


def render_learning_path_analysis(df):
    """Render learning path analysis."""
    st.subheader("Learning Path Analysis")
    if len(df) > 1:
        # Analyze query sequences with better deduplication
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        df_sorted['prev_query'] = df_sorted['question'].shift(1)
        df_sorted['time_gap'] = (df_sorted['timestamp'] - df_sorted['timestamp'].shift(1)).dt.total_seconds()
        
        # Filter for meaningful sequences (within 10 minutes, different questions)
        meaningful_sequences = df_sorted[
            (df_sorted['time_gap'] < 600) & 
            (df_sorted['time_gap'] > 5) &  # At least 5 seconds apart
            (df_sorted['prev_query'] != df_sorted['question'])  # Different questions
        ][['prev_query', 'question']].dropna()
        
        if not meaningful_sequences.empty:
            # Extract topic transitions (simplified keyword extraction)
            def extract_key_topic(question):
                """Extract main topic from question"""
                # Remove common question words
                stopwords = {'what', 'is', 'how', 'does', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
                words = re.findall(r'\b\w+\b', question.lower())
                keywords = [w for w in words if len(w) > 3 and w not in stopwords]
                return ' '.join(keywords[:3]) if keywords else question[:30]
            
            # Build topic transitions
            topic_transitions = []
            for _, row in meaningful_sequences.iterrows():
                prev_topic = extract_key_topic(row['prev_query'])
                curr_topic = extract_key_topic(row['question'])
                
                # Only add if topics are actually different
                if prev_topic != curr_topic:
                    topic_transitions.append(f"{prev_topic} → {curr_topic}")
            
            if topic_transitions:
                st.markdown("**🔄 Learning Progressions** (different topics within 10 minutes)")
                st.caption("Shows how students naturally progress from one concept to another")
                
                transition_counts = Counter(topic_transitions).most_common(10)
                transition_df = pd.DataFrame(transition_counts, columns=['Topic Flow', 'Frequency'])
                st.dataframe(transition_df)
                
                # Visualization of top transitions
                if PLOTLY_AVAILABLE and len(transition_counts) > 0:
                    top_transitions = transition_df.head(5)
                    fig = go.Figure(go.Bar(
                        x=top_transitions['Frequency'],
                        y=top_transitions['Topic Flow'],
                        orientation='h',
                        marker=dict(
                            color='rgba(168, 85, 247, 0.85)',
                            line=dict(color='rgba(168, 85, 247, 1)', width=1.5)
                        ),
                        text=top_transitions['Frequency'],
                        textposition='auto',
                        textfont=dict(color='white', size=11),
                        hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
                    ))
                    fig.update_layout(
                        title="<b>Top 5 Learning Progressions</b>",
                        height=340,
                        xaxis_title="Frequency",
                        yaxis_title="",
                        plot_bgcolor='rgba(30, 41, 59, 0.5)',
                        paper_bgcolor='rgba(13, 17, 23, 0)',
                        font=dict(color='#e6eef8', family='Arial', size=10),
                        margin=dict(l=20, r=20, t=50, b=40),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor='rgba(255, 255, 255, 0.08)',
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(255, 255, 255, 0.1)'
                        ),
                        yaxis=dict(
                            showgrid=False,
                            showline=True,
                            linewidth=1,
                            linecolor='rgba(255, 255, 255, 0.1)'
                        ),
                        hovermode='y unified'
                    )
                    st.plotly_chart(fig)
            else:
                st.info("💡 Students are mostly asking isolated questions. No clear learning progressions detected yet.")
        else:
            st.info("💡 Not enough sequential queries to detect learning patterns. Students may be using the system sporadically.")
        
        # Show repeated queries separately (indicates confusion)
        repeated_queries = df_sorted[
            (df_sorted['time_gap'] < 600) & 
            (df_sorted['prev_query'] == df_sorted['question'])
        ]
        
        if not repeated_queries.empty:
            st.markdown("---")
            st.markdown("**⚠️ Repeated Questions** (same question asked multiple times)")
            st.caption("These topics may need clearer explanations")
            
            repeat_counts = repeated_queries['question'].value_counts().head(5)
            repeat_df = pd.DataFrame({
                'Question': repeat_counts.index,
                'Times Repeated': repeat_counts.values
            })
            st.dataframe(repeat_df)
