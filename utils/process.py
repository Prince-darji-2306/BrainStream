import streamlit as st
from video.video_processing import get_subtitles

def session_state(start = True, reset = False):
    if start:
        if 'page' not in st.session_state:
            st.session_state["page"] = "search"
    for i in ["selected_video", 'selected_id', "vectorstore", "chat_chain", "memory", 'results', 'flag']:
        if i not in st.session_state:
            st.session_state[i] = None
    if reset:
        for i in ["memory",'chat_chain']:
            st.session_state[i] = None


def process_video(id, title, rerun = True):
    with st.spinner('Processing Transcript...'):
        vectorstore = get_subtitles(id)
        if not vectorstore:
            st.error("No subtitles found for this video.")
        else:
            st.session_state["selected_video"] = title
            st.session_state["vectorstore"] = vectorstore
            session_state(reset=True)

            if rerun:
                st.session_state["page"] = "chat"
                st.rerun()
                

def show_videos(videos):
    cols = st.columns(3) 
    for i, video in enumerate(videos):
        with cols[i % 3]:
            st.markdown(f"""
                <div class='card'>
                    <img src="{video['thumbnail']}" style="width:100%" />
                    <div class ='card-text'>{video['title']}</div>
                </div>
            """, unsafe_allow_html=True)
            if st.button("▶ Select", key=f"{video['id']}"):
                st.session_state['selected_video'] = video['title']
                st.session_state['selected_id'] = video['id']
                st.session_state["page"] = "chat"
                st.rerun()