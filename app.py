import streamlit as st
from video.youtube_search import search_youtube_videos, youtube_id
from llm_engine import get_conversational_chain, get_conversation
from utils.process import session_state, process_video, show_videos
from langchain.schema import HumanMessage, AIMessage


st.set_page_config(page_title="BrainStrem | RAG Assistnt for YouTube Video", layout="wide", page_icon='static/img/icon.png')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("static/css/ystyle.css")

# Sidebar navigation buttons
with st.sidebar:
    st.title("BrainStream")
    if st.button("Search Video"):
        st.session_state["page"] = "search"
    if st.button("Chat with Video"):
        if "selected_video" in st.session_state:
            st.session_state["page"] = "chat"
        else:
            st.warning("Please select a video.")

session_state()

if st.session_state["page"] == "search":
    st.header("Search YouTube Videos")
    query = st.text_input("Enter YouTube Video Name or Paste Link")
    if query:
        if "youtube.com" in query or "youtu.be" in query:
            st.session_state['flag'] = False
            id, title = youtube_id(query)
            st.session_state.results = None
            process_video(id, title)
        else:
            with st.spinner('Fetching Videos.'):
                videos = search_youtube_videos(query, limit=6)
            if not videos:
                st.info("No videos found.")
            else:
                st.session_state['results'] = videos
                st.session_state['flag'] = True
                show_videos(videos)
    elif st.session_state['results']:
        show_videos(st.session_state['results'])

elif st.session_state["page"] == "chat":
    st.header("Chat with Video")

    video = st.session_state.get("selected_video")
    
    if not video:
        st.warning("No video selected, please go to Search Video first.")
    else:
        st.markdown('Video : ' + video)
        if st.session_state['flag']:
            st.session_state.flag = False
            process_video(st.session_state['selected_id'],
                          st.session_state['selected_video'], False)

        if not st.session_state['chat_chain']:
            chat_chain, memory = get_conversational_chain(st.session_state['vectorstore'])
            st.session_state["chat_chain"] = chat_chain
            st.session_state["memory"] = memory
        else:
            chat_chain = st.session_state["chat_chain"]
            memory = st.session_state["memory"]
        
        chat = get_conversation(memory)

        for msg in chat:
            if isinstance(msg, HumanMessage):
                st.markdown(f'<div class="user-msg">{msg.content}</div>', unsafe_allow_html=True)
            elif isinstance(msg, AIMessage):
                st.markdown(msg.content, unsafe_allow_html=True)     

        query = st.chat_input("Ask anything about the video:")
        if query:
            st.markdown(f'<div class="user-msg">{query}</div>', unsafe_allow_html=True)
            
            with st.spinner("Getting response..."):
                result = chat_chain.run(question=query)
            st.markdown(result, unsafe_allow_html=True)


        
