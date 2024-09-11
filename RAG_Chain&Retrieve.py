from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader('Attention.pdf').load()

text_split = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100 ).split_documents(loader)

db = Chroma.from_documents(text_split, OpenAIEmbeddings())

query = "who are the author of this book?"
results = db.similarity_search(query)

print(results[0].page_content)