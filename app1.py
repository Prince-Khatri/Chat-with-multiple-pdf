import os
import asyncio
import nest_asyncio

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI

def set_event_loop():
    nest_asyncio.apply()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # new instance
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks
    
def get_vs(chunks,api_key):
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-')
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
    vs = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vs



def get_cc(vs,api_key):
    # llm = ChatOpenAI()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=api_key)
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    cc = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vs.as_retriever(),
        memory=memory
    )
    return cc

def chat(question):
    

def main():
    load_dotenv()
    api_key = os.getenv('GOOGLE_STUDIO_API_KEY')
    set_event_loop() # for asyncio gemini and st conflict

    if "cc" not in st.session_state:
        st.session_state.cc = None

    st.set_page_config(page_title="Chat with books",page_icon=":books:")
    st.header("Chat with books")

    
    question = st.text_input("Ask your questions here")
    if(st.button()):
        chat(question)



    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Enter your files",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # test extraction
                raw_text = get_pdf_text(pdf_docs)

                # chunk creation
                chunks = get_text_chunks(raw_text)


                # creating vector space
                vs = get_vs(chunks,api_key)


                # converstaion chain
                st.session_state.cc = get_cc(vs,api_key)


                





if __name__=='__main__':
    main()