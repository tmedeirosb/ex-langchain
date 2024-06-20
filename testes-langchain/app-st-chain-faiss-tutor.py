import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document

# Preparar o Corpus de Documentos
documents = [
    "O gato está no telhado.",
    "O cachorro está no jardim.",
    "Os pássaros estão cantando nas árvores.",
    "O peixe está nadando no aquário."
]

# Configurar a Recuperação de Documentos

# Criar embeddings para os documentos
embeddings = OllamaEmbeddings(model="mxbai-embed-large") 

# Obter embeddings dos documentos
doc_embeddings = [embeddings.embed_query(doc) for doc in documents]

# Inicializar o índice FAISS
dimension = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dimension)

# Adicionar embeddings ao índice
index.add(np.array(doc_embeddings))

# Criar um docstore em memória
docstore = InMemoryDocstore({i: Document(page_content=doc) for i, doc in enumerate(documents)})

# Criar um mapeamento de index para docstore_id
index_to_docstore_id = {i: i for i in range(len(documents))}

# Criar a store FAISS usando o índice criado manualmente
vector_store = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Recuperação de Documentos
def retrieve_documents(query, k=2):
    results = vector_store.similarity_search(query, k=k)
    # Acessar o atributo 'page_content' dos resultados
    return [result.page_content for result in results]

query = "Onde está o gato?"
retrieved_docs = retrieve_documents(query)
st.write("Documentos recuperados:", retrieved_docs)

# Geração de Resposta

# Criar o modelo de linguagem
llm = Ollama(model='llama3:latest')

# Definir o template do prompt
prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Query: {query}\nContext: {context}\nAnswer:"
)

# Criar a cadeia de geração de texto
chain = LLMChain(llm=llm, prompt=prompt)

def generate_answer(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    response = chain.run(query=query, context=context)
    return response

answer = generate_answer(query, retrieved_docs)
st.write("Resposta gerada:", answer)
