from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

API = os.getenv('GROQ')

# Initialize LLM
llm = ChatGroq(api_key=API, 
               model_name="openai/gpt-oss-20b", 
               temperature=0.1)


def get_conversational_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )
    return qa_chain, memory

def get_conversation(memory):
    return memory.load_memory_variables({})["chat_history"]