import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
#.agents.create_pandas_dataframe_agent
from langchain import PromptTemplate
from langchain.llms import Ollama

from sklearn.datasets import load_iris

df = load_iris(as_frame=True)["data"]

PROMPT = (
    "If you do not know the answer, say you don't know.\n"
    "Think step by step.\n"
    "\n"
    "Below is the query.\n"
    "Query: {query}\n"
)
prompt = PromptTemplate(template=PROMPT, input_variables=["query"])
llm = Ollama(model='llama3:latest')
agent = create_pandas_dataframe_agent(llm, df, verbose=True)

messages = st.container()
if prompt_user := st.chat_input("Say something"):

    response = agent.run(prompt.format(query=prompt_user))

    messages.chat_message("user").write(prompt_user)
    messages.chat_message("assistant").write(f"Echo: {response}")

st.dataframe(df)