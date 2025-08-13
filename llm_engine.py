from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

API = os.getenv('GROQ')

llm = ChatGroq( api_key=API, 
                model_name="qwen/qwen3-32b", 
                temperature=0.1,
                reasoning_format = 'hidden',
                reasoning_effort = 'none'
               )

SPROMPT = """
You are an AI assistant that writes answers in a visually engaging, Markdown‑rich format.  
Your responses should:
-> Be clear and concise, Explain only from given context and don't explain too much.
1. **Use Markdown** for headings, bold, italics, bullet lists, numbered lists, and tables.  
2. **Use inline HTML** (example: `<span style="color:#ff5733;">text</span>`) to add colored text when appropriate.  
3. Answer in English only, regardless of the language of the context or input.
4. Do not mention that you are an AI or that you are following a prompt.  
"""

def get_conversational_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_prompt = PromptTemplate.from_template(
        SPROMPT + "\n\nUse context to answer the question"
        "\n\n {context}\n\nQuestion: {question}\n Helpful Answer:"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return qa_chain, memory

def get_conversation(memory):
    return memory.load_memory_variables({})["chat_history"]