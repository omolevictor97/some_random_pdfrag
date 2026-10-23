import streamlit as st
import time
import os
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv, find_dotenv

if os.path.exists('.env'):
    load_dotenv(find_dotenv('.env'))

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(
    model= "qwen/qwen3.8-27b",
    groq_api_key = groq_api_key, temperature=0.6
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a genius and can answer question based on context only.
    Please, provide the most accurate response based on the question <context>
    {context}
    Question : {input}
    """
)

def create_vector_embeddings():
    research_folder_path = os.path.join(os.getcwd(), "research_papers")
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader(research_folder_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

user_prompt = st.text_input('Enter your query from the research paper: ')

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database Is Ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(
        llm = model, prompt = prompt
    )
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retriever_chain.invoke({'input':user_prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])

    # with streamlit expander
    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)