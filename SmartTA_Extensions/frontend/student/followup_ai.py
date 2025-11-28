"""AI-powered follow-up question handler.
Extracted from student.py to improve maintainability.
"""
import os
import json
import datetime
import streamlit as st

from SmartTA_Extensions.utils.helpers import get_video_id_from_title


def render_followup_question_section(META_SEGMENTS: str) -> None:
    """Render the AI-powered follow-up question expander."""
    if not st.session_state.get("last_search_results") or not st.session_state.get("last_search_query"):
        return
    
    st.markdown("")
    with st.expander("💬 Ask Follow-up Question (AI-powered)", expanded=False):
        st.caption("Ask a clarifying question based on the search results above. GPT will synthesize an answer from the retrieved segments.")
        
        followup_query = st.text_input(
            "Follow-up question:",
            placeholder="e.g., Why is the learning rate important?",
            key="followup_query_input",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            ask_followup = st.button("🤖 Ask AI", key="ask_followup_btn", width='stretch', type="primary")
        with col2:
            use_top_n = st.selectbox("Use top", [3, 5, 8], index=0, key="followup_top_n")
        
        if ask_followup and followup_query:
            _handle_followup_query(followup_query, use_top_n, META_SEGMENTS)


def _handle_followup_query(followup_query: str, use_top_n: int, META_SEGMENTS: str) -> None:
    """Handle the AI follow-up query execution."""
    with st.spinner("🧠 AI is thinking..."):
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                st.error("⚠️ OPENAI_API_KEY not set. Please configure your API key.")
                return
            
            # Load all segments for context expansion
            all_segments_data = {}
            if os.path.exists(META_SEGMENTS):
                with open(META_SEGMENTS, "r", encoding="utf-8") as f:
                    all_segments_data = json.load(f)
            
            # Build context from search results with neighboring segments
            rs = st.session_state["last_search_results"][:use_top_n]
            context_parts = []
            for idx, r in enumerate(rs, 1):
                lec = r.get("lecture", "Unknown")
                ts_start = r.get("start", 0)
                ts_end = r.get("end", 0)
                text = r.get("text", "")
                
                # Get neighboring segments for fuller context (±1 segment before/after)
                lecture_segments = all_segments_data.get(lec, [])
                expanded_text = text
                
                if lecture_segments:
                    # Find current segment index
                    current_idx = None
                    for seg_idx, seg in enumerate(lecture_segments):
                        if abs(seg.get("start", 0) - ts_start) < 1:  # Match by timestamp
                            current_idx = seg_idx
                            break
                    
                    if current_idx is not None:
                        # Add context: 1 segment before + current + 1 segment after
                        context_segments = []
                        if current_idx > 0:
                            context_segments.append(lecture_segments[current_idx - 1].get("text", ""))
                        context_segments.append(text)
                        if current_idx < len(lecture_segments) - 1:
                            context_segments.append(lecture_segments[current_idx + 1].get("text", ""))
                        
                        expanded_text = " ".join(context_segments)
                
                context_parts.append(f"[{idx}] From '{lec}' at {datetime.timedelta(seconds=int(ts_start))}:\n{expanded_text}")
            
            context = "\n\n".join(context_parts)
            original_q = st.session_state["last_search_query"]
            
            # Call OpenAI
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful teaching assistant. Answer the student's follow-up question using this priority:\n1. PRIMARY: Use information from the provided lecture segments and cite them with [1], [2], etc.\n2. FALLBACK: If the segments don't contain enough information, provide a correct general explanation but clearly mark it as 'Note: This information is not from your lecture materials - it's general knowledge about [topic].'\nBe concise, accurate, and educational. Always distinguish between lecture content and general knowledge."
                    },
                    {
                        "role": "user",
                        "content": f"Original question: {original_q}\n\nLecture segments:\n{context}\n\nFollow-up question: {followup_query}\n\nAnswer using lecture segments if possible (cite [1], [2], etc.). If not, provide accurate general knowledge but label it clearly."
                    }
                ],
                temperature=0.5,
                max_tokens=600
            )
            
            answer = response.choices[0].message.content
            
            st.markdown("### 🎓 AI Answer")
            st.markdown(answer)
            
            st.markdown("---")
            st.markdown("**📚 Source Segments:**")
            for idx, r in enumerate(rs, 1):
                lec = r.get("lecture")
                vid = get_video_id_from_title(lec)
                ts = int(r.get("start", 0))
                tstr = str(datetime.timedelta(seconds=ts))
                if vid:
                    url = f"https://youtu.be/{vid}?t={ts}"
                    st.markdown(f"**[{idx}]** [{lec} @ {tstr}]({url})")
                else:
                    st.markdown(f"**[{idx}]** {lec} @ {tstr}")
            
        except ImportError:
            st.error("❌ OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Make sure your OPENAI_API_KEY is valid and you have API credits.")
