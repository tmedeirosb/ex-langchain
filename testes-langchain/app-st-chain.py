import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOllama 
from langchain.schema import StrOutputParser
from langchain.chains import OpenAIModerationChain
   
cot_prompt = PromptTemplate.from_template(
    "{question} \nLet's think step by step!"
)

#moderation_chain = OpenAIModerationChain()
#llm_chain = cot_prompt | ChatOllama(model='llama3:latest') | StrOutputParser() | moderation_chain

llm_chain = cot_prompt | ChatOllama(model='llama3:latest') | StrOutputParser()

response = llm_chain.invoke({"question": "What is the future of programming?"})

st.write(response)