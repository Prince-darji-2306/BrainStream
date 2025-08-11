import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")