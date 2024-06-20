import streamlit as st
from langchain.llms import Ollama
from typing import Optional
from langchain.pydantic_v1 import BaseModel, Field

# Definindo o LLM
llm = Ollama(model='llama3:latest')

class Experience(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class Study(Experience):
    degree: Optional[str] = None
    university: Optional[str] = None
    country: Optional[str] = None
    grade: Optional[str] = None

class WorkExperience(Experience):
    company: str
    job_title: str

class Resume(BaseModel):
    first_name: str
    last_name: str
    linkedin_url: Optional[str] = None
    email_address: Optional[str] = None
    nationality: Optional[str] = None
    skill: Optional[str] = None
    study: Optional[Study] = None
    work_experience: Optional[WorkExperience] = None
    hobby: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

from langchain.chains import create_extraction_chain_pydantic
from langchain.document_loaders import PyPDFLoader

# Carregando e dividindo o PDF
pdf_file_path = "data/lattes.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

# Criando e executando a cadeia de extração
chain = create_extraction_chain_pydantic(pydantic_schema=Resume, llm=llm)
try:
    res = chain.run(docs)
    st.write(res)
except Exception as e:
    st.write(f"An error occurred: {e}")