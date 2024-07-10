# Import necessary modules 
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Initialize language model
llm = ChatOpenAI(
        api_key="ollama",
        model="llama3:latest",
        #model="gemma2:latest", #respostas sempre em ingles
        #model="splitpierre/bode-alpaca-pt-br:latest", 
        #model="splitpierre/bode-alpaca-pt-br:13b-Q4_0", 
        base_url="http://localhost:11434/v1",
        temperature=0, 
    )

# Load the summarization chain
summarize_chain = load_summarize_chain(llm)

# Load the document using PyPDFLoader
document_loader = PyPDFLoader(file_path="data/OrganizacaoDidatica_2012_versaoFINAL_20mai2012 (4).pdf")
document = document_loader.load()

# Summarize the document
summary = summarize_chain(document)
print(summary['output_text'])

