from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

#função para ler o pdf
def pdf_read(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

#função para segmentar o texto
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

#path do pdf
pdf_path = 'data/OrganizacaoDidatica_2012_versaoFINAL_20mai2012 (4).pdf'

#transforma pdf em texto corrido
raw_text = pdf_read(pdf_path)

#segmenta o texto em chunks
text_chunks = get_chunks(raw_text)

#set o modelo de embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large") 

#cria o vetor store
db = FAISS.from_texts(text_chunks, embedding=embeddings)
#db.save_local("faiss_db")

#cria o LLM de conversação
llm = ChatOpenAI(
        api_key="ollama",
        #model="llama3:latest",
        model="gemma2:latest",
        #model="splitpierre/bode-alpaca-pt-br:latest", 
        #model="splitpierre/bode-alpaca-pt-br:13b-Q4_0", 
        base_url="http://localhost:11434/v1",
    )

#cria o objeto de recuperação de perguntas e respostas
qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

#faz uma pergunta
res = qa.invoke("Quais são os critérios de reprovação do curso?")

#imprime a resposta
print(res['result'])



