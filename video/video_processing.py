from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from embedding import load_model

def get_subtitles(video_id):
    try:
        transcript = YouTubeTranscriptApi()
        transcript = transcript.fetch(video_id, languages=['en'])

        full_text = " ".join([snippet.text for snippet in transcript])
        if full_text.strip() != '':
            return split_text(full_text)
        else:
            return None
    
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

# Simple text splitter for RAG - chunk size 500 words with 100 overlap
def split_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    
    return create_vectorstore(chunks)

def create_vectorstore(text_chunks):
    docs = [Document(page_content=t) for t in text_chunks]

    vectorstore = FAISS.from_documents(docs, load_model())
    return vectorstore