from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are an assistant'),
        ("user" ,"Question {question}" )
    ]
)

llm = ChatOpenAI()
st.title("Simple chat bot App")
input_text = st.text_input("Search the topic you require help: ")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question',input_text}))