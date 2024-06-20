import streamlit as st

from langchain_community.embeddings import OllamaEmbeddings

embeddings = (
    #OllamaEmbeddings(model='llama3:latest')
    OllamaEmbeddings(model="mxbai-embed-large")    
)  # by default, uses llama2. Run `ollama pull llama2` to pull down the model

text = "This is a test document."

query_result = embeddings.embed_query(text)

st.write(query_result[:5])
st.write(len(query_result))

words = ["cat", "dog", "computer", "animal"]
doc_vectors = embeddings.embed_documents(words)

st.write(doc_vectors[:5])
st.write(len(doc_vectors))

from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
X = np.array(doc_vectors)
dists = squareform(pdist(X))

import pandas as pd
df = pd.DataFrame(
    data=dists,
    index=words,
    columns=words
)

st.dataframe(df.style.background_gradient(cmap='coolwarm'))

from langchain.vectorstores import Chroma
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import CharacterTextSplitter

loader = ArxivLoader(query="2310.06825")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(documents=docs, embedding=OllamaEmbeddings(model="mxbai-embed-large"))

k = 2
query_vector = embeddings.embed_query("study")
similar_vectors = vectorstore.query(query_vector, k)

st.write(similar_vectors)