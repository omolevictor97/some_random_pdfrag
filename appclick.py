import streamlit as st
import re
import io
import base64
import pygame
import time

from pathlib import Path
from bs4 import BeautifulSoup

from langchain_community.document_loaders import (
    WebBaseLoader, SitemapLoader, RecursiveUrlLoader,
    PyPDFDirectoryLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from universal_docs import url, pdf_folder, db_dir
from dotenv import load_dotenv, find_dotenv
from voice_generation import generate_voice

import os

if os.path.exists('.env'):
    load_dotenv(find_dotenv('.env'))




groq_api_key = os.getenv('GROQ_API_KEY')

model = ChatGroq(
    model="openai/gpt-oss-120b", 
    groq_api_key=groq_api_key, 
    temperature=1
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a genius and can answer question based on context only.
    Please, provide the most accurate response based on the question 
    And do not start with 'According to the provided context or anything that 
    looks like it, it is too boring'. Infact Act formally like an official assistant
    <context>
    {context}
    Question : {input}
    """
)

for existing_file in Path(__file__).parent.glob('speech_*.wav'):
    try:
        os.remove(existing_file)
    except PermissionError:
        pass


# embeddings come first
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def load_data(url, pdf_folder):
    web_loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor= lambda x: BeautifulSoup(x, 'html.parser').text
    )

    pdf_loader = PyPDFDirectoryLoader(path=os.path.join(os.getcwd(), pdf_folder))

    return web_loader.load() + pdf_loader.load()

def split_doc(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)
    return splits




def get_retriever():
    # check if the vector database exists
    if os.path.exists(db_dir) and os.listdir(db_dir):
        print('Using existing vector database >>>')
        vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        final_chunks = split_doc(load_data(url, pdf_folder))
        vector_store = Chroma.from_documents(
            documents=final_chunks, embedding=embeddings,
            persist_directory=db_dir
        )
    return vector_store.as_retriever(search_kwargs={'k' : 5})

user_prompt = st.text_input("Hey, this appclick AI, what would you want to know about us?")


if user_prompt:
    unique_id = int(time.time())

    speech_file_path = Path(__file__).parent / f"speech_{unique_id}.wav"

    document_chain = create_stuff_documents_chain(
        llm= model, prompt = prompt
    )
    retriever = get_retriever()
    retriever_chain = create_retrieval_chain(
        retriever, document_chain
    )
    start = time.process_time()
    response = retriever_chain.invoke({'input':user_prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])



    response = generate_voice(
        input_text= response['answer'],
        speech_file_path=speech_file_path
    )
    audio_bytes = response.read()
    b64_audio = base64.b64encode(audio_bytes).decode()

    md = f"""
        <audio autoplay>
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

    # pygame.mixer.init()
    # pygame.mixer.music.load(str(speech_file_path))
    # pygame.mixer.music.play()

    # while pygame.mixer.music.get_busy():
    #     time.sleep(0.1)
