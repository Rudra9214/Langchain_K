import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_objectbox.vectorstores import ObjectBox
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

st.title("Search in Directory using ObjectBox")

prompt = ChatPromptTemplate.from_template(
      """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}

    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.pdfloader = PyPDFDirectoryLoader('./pdf files').load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200).split_documents(st.session_state.pdfloader)
        st.session_state.vectors = ObjectBox.from_documents(st.session_state.splitter,st.session_state.embeddings,embedding_dimensions=768)

input_prompt=st.text_input("Enter Your Question From Documents")

if st.button("Document Embedding"):
    vector_embedding()    
    st.write("ReadY!")

if input_prompt:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    ret_chain = create_retrieval_chain(retriever,doc_chain)
    response = ret_chain.invoke({'input': input_prompt})
    st.write(response['answer'])