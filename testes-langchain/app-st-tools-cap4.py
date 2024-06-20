import streamlit as st

#from langchain.llms import Ollama
#llm = Ollama(model='llama3:latest')

from langchain.agents import (
    AgentExecutor, AgentType, initialize_agent, load_tools
)

from langchain.chat_models import ChatOllama

def load_agent() -> AgentExecutor:
    llm = ChatOllama(model='llama3:latest', streaming=True)
    # DuckDuckGoSearchRun, wolfram alpha, arxiv search, wikipedia 
    # TODO: try wolfram-alpha!
    tools = load_tools(
                #tool_names=["ddg-search", "wolfram-alpha", "arxiv", "wikipedia"],
                tool_names=["ddg-search", "arxiv", "wikipedia"],
                llm=llm 
            )
    
    return initialize_agent(
        tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

from langchain.callbacks import StreamlitCallbackHandler

chain = load_agent()

st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        #response = chain.run(prompt, callbacks=[st_callback])
        #st.write(response)
        try:
            #response = chain.run(prompt, callbacks=[st_callback])
            response = chain.run(prompt, callbacks=[st_callback])
            st.write(response)
        except Exception as e:
            st.write(f"Error: {e}")