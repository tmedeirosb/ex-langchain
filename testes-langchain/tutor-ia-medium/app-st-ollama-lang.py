#https://ai.gopubby.com/improving-llms-with-ollama-and-rag-508fad3f841f

import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# load the PDF
pdf_loader = PyPDFLoader('/Users/tmedeirosb/Desktop/DEV/LANGCHAIN/tutor-ia-medium/data/lattes.pdf')
doc = pdf_loader.load()

#chunk it
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3:latest")

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Create the retriever
retriever = vectorstore.as_retriever()

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3:latest', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = doc
    return ollama_llm(question, formatted_context)

# Use the RAG chain
result = rag_chain("""
What is this document about?
""")

st.write(result)