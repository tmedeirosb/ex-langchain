#https://blog.gopenai.com/building-a-multi-pdf-rag-chatbot-langchain-streamlit-with-code-d21d0a1cf9e5
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.chat_models import ChatOllama 
#from langchain.llms import Ollama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

embeddings = OllamaEmbeddings(model="mxbai-embed-large") 

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(retriever, ques):

    #TESTE DO RETRIEVER
    # test_query = "Diga o nome da universidade"
    # try:
    #     test_results = retriever.get_relevant_documents(test_query)
    #     st.write("Teste de funcionamento do retriever:", test_results)
    #     if not test_results:
    #         st.write("O retriever não retornou resultados. Verifique se a base está corretamente populada e o retriever configurado.")
    #         return
    # except Exception as e:
    #     st.write("Erro ao verificar o retriever:", str(e))
    #     return
    
    #llm = ChatOllama(model='llama3:latest') 
    llm = ChatOpenAI(
        api_key="ollama",
        model="llama3:latest",
        #model="splitpierre/bode-alpaca-pt-br:latest", 
        #model="splitpierre/bode-alpaca-pt-br:13b-Q4_0", 
        base_url="http://localhost:11434/v1",
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é um assistente prestativo. Responda à pergunta com o máximo 
            de detalhes possível a partir do contexto fornecido. Certifique-se de 
            fornecer todos os detalhes. Se a resposta não estiver no contexto fornecido, 
            apenas diga “a resposta não está disponível no contexto”. Não forneça uma resposta errada.""",
        ),
        #("placeholder", "{chat_history}"),
        ("human", "{input}"),
        #("placeholder", "{agent_scratchpad}"),
    ])
    
    chain = retriever | prompt | llm | StrOutputParser()
    response = chain.invoke({"input": ques})

    st.write("Reply: ", response)

def user_input(user_question):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    
    retriever = new_db.as_retriever()

    #TESTE DO RETRIEVER
    # test_query = "Teste de funcionamento"
    # try:
    #     test_results = retriever.get_relevant_documents(test_query)
    #     st.write("Teste de funcionamento do retriever:", test_results)
    #     if not test_results:
    #         st.write("O retriever não retornou resultados. Verifique se a base está corretamente populada e o retriever configurado.")
    #         return
    # except Exception as e:
    #     st.write("Erro ao verificar o retriever:", str(e))
    #     return

    #retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
    
    # Testar se o retrieval_chain está funcionando
    # try:
    #     response = get_conversational_chain(retrieval_chain, user_question)
    #     print("Resposta da cadeia de recuperação:", response)
    # except Exception as e:
    #     print("Erro ao executar a cadeia de recuperação:", str(e))

    get_conversational_chain(retriever, user_question)

def main():
    st.set_page_config("Chat PDF")
    st.title("Chat PDF: Ollama + Llama 3:8b + FAISS + Langchain + Streamlit")

    user_question = st.text_input("Faça uma Pergunta a partir dos Arquivos PDF")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Envie seus arquivos PDF e clique no botão Enviar e Processar.", accept_multiple_files=True)
        if st.button("Enviar e Processar"):
            with st.spinner("Processando..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Processado")

if __name__ == "__main__":
    main()