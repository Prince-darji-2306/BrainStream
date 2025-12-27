import logging
from langchain.schema import Document
from utils.embedding import load_model
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, CouldNotRetrieveTranscript, TranslationLanguageNotAvailable, NotTranslatable

# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def get_subtitles(video_id):
    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)

        english_codes = ['en', 'en-US', 'en-GB']
        
        def fetch_text(transcript):
            subtitles = transcript.fetch()
            text = " ".join([snippet.text for snippet in subtitles])
            if transcript.language.lower().startswith('en'):
                return split_text(text)
            
            return split_text(text, 300, 50)
        
        try:
            transcript = transcript_list.find_manually_created_transcript(english_codes)
            return fetch_text(transcript)
        
        except CouldNotRetrieveTranscript:
            pass
        
        # Step 2: Check any manually created transcript, translate if possible
        manual_transcripts = [t for t in transcript_list if not t.is_generated]
        if manual_transcripts:

            orig_transcript = manual_transcripts[0]
            if orig_transcript.is_translatable:
                try:
                    translated_transcript = orig_transcript.translate('en')
                    return fetch_text(translated_transcript)

                except (TranslationLanguageNotAvailable, NotTranslatable):
                    pass
            return fetch_text(orig_transcript)
        
        # Step 3: Check generated transcripts in English
        try:
            transcript = transcript_list.find_generated_transcript(english_codes)
            return fetch_text(transcript)
        except CouldNotRetrieveTranscript:
            pass
        
        # Step 4: Any generated transcript
        generated_transcripts = [t for t in transcript_list if t.is_generated]
        if generated_transcripts:
            orig_transcript = generated_transcripts[0]
            if orig_transcript.is_translatable:
                try:
                    translated_transcript = orig_transcript.translate('en')
                    return fetch_text(translated_transcript)
                except (TranslationLanguageNotAvailable, NotTranslatable):
                    pass

            return fetch_text(orig_transcript)
        
        raise NoTranscriptFound("No subtitles available for this video.")
    except Exception as e:
        logger.info(e)
        return None
    

# Simple text splitter for RAG - chunk size 500 words with 100 overlap
def split_text(text, chunk_size=600, overlap=80):
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